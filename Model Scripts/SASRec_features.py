import numpy as np
import torch
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
import argparse
import random
import os

class SessionDataset(Dataset):
    def __init__(self, parquet_path, features, target):
        df = pd.read_parquet(parquet_path)
        df = df.dropna(subset=features)
        self.sessions = []
        self.labels = []
        for session_id, session in df.groupby("session_id"):
            session = session.sort_values("ts")
            x = torch.tensor(session[features].values, dtype=torch.float32)
            y = torch.tensor(session[target].values, dtype=torch.float32)
            self.sessions.append(x)
            self.labels.append(y)
    def __len__(self):
        return len(self.sessions)
    def __getitem__(self, idx):
        return self.sessions[idx], self.labels[idx]


def collate_fn(batch):
    sessions, labels = zip(*batch)
    sessions_padded = nn.utils.rnn.pad_sequence(sessions, batch_first=True)
    labels_padded = nn.utils.rnn.pad_sequence(labels, batch_first=True)
    lengths = torch.tensor([len(s) for s in sessions])
    return sessions_padded, labels_padded, lengths


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()
        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)
        return outputs


class SASRec(torch.nn.Module):
    def __init__(self, feature_no, args):
        super(SASRec, self).__init__()
        self.feature_no = feature_no
        self.dev = args.device
        self.norm_first = args.norm_first
        self.feature_proj = torch.nn.Linear(self.feature_no, args.hidden_units)
        self.pos_emb = torch.nn.Embedding(args.maxlen + 1, args.hidden_units, padding_idx=0)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.attention_layernorms = torch.nn.ModuleList()
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()
        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
        for _ in range(args.num_blocks):
            self.attention_layernorms.append(torch.nn.LayerNorm(args.hidden_units, eps=1e-8))
            self.attention_layers.append(torch.nn.MultiheadAttention(args.hidden_units, args.num_heads, args.dropout_rate))
            self.forward_layernorms.append(torch.nn.LayerNorm(args.hidden_units, eps=1e-8))
            self.forward_layers.append(PointWiseFeedForward(args.hidden_units, args.dropout_rate))
        self.output_layer = torch.nn.Linear(args.hidden_units, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def seq2feats(self, x, lengths):
        seqs = self.feature_proj(x)
        N, seq_len, _ = x.shape
        poss = torch.arange(1, seq_len + 1, device=self.dev).unsqueeze(0).expand(N, -1).clone()
        poss = poss.clamp(max=self.pos_emb.num_embeddings - 1)
        for i, l in enumerate(lengths):
            poss[i, l:] = 0
        seqs += self.pos_emb(poss)
        seqs = self.emb_dropout(seqs)
        tl = seq_len
        causal_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))
        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            if self.norm_first:
                x_ = self.attention_layernorms[i](seqs)
                mha_out, _ = self.attention_layers[i](x_, x_, x_, attn_mask=causal_mask)
                seqs = seqs + mha_out
                seqs = torch.transpose(seqs, 0, 1)
                seqs = seqs + self.forward_layers[i](self.forward_layernorms[i](seqs))
            else:
                mha_out, _ = self.attention_layers[i](seqs, seqs, seqs, attn_mask=causal_mask)
                seqs = self.attention_layernorms[i](seqs + mha_out)
                seqs = torch.transpose(seqs, 0, 1)
                seqs = self.forward_layernorms[i](seqs + self.forward_layers[i](seqs))
        return self.last_layernorm(seqs)

    def forward(self, x, lengths):
        feats = self.seq2feats(x, lengths)
        skip_probs = self.sigmoid(self.output_layer(feats)).squeeze(-1)
        return skip_probs


def evaluate_sequential(model, loader, device, min_context=5, min_skips_in_context=1, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for sessions, labels, lengths in loader:
            sessions = sessions.to(device)
            preds = model(sessions, lengths)
            for i, length in enumerate(lengths):
                length = length.item()
                if length < min_context + 1:
                    continue
                context_len = torch.randint(min_context, length, (1,)).item()
                if labels[i, :context_len].sum().item() < min_skips_in_context:
                    continue
                all_preds.extend(preds[i, context_len:length].cpu().numpy())
                all_labels.extend(labels[i, context_len:length].numpy())
    return roc_auc_score(all_labels, all_preds)


def permutation_importance(model, loader, device, features, baseline_auc):
    importances = {}
    for feat_idx, feat_name in enumerate(features):
        all_preds, all_labels = [], []
        model.eval()
        with torch.no_grad():
            for sessions, labels, lengths in loader:
                sessions = sessions.clone()
                sessions[:, :, feat_idx] = sessions[:, :, feat_idx][torch.randperm(sessions.size(0))]
                sessions = sessions.to(device)
                preds = model(sessions, lengths)
                for i, length in enumerate(lengths):
                    all_preds.extend(preds[i, :length].cpu().numpy())
                    all_labels.extend(labels[i, :length].numpy())
        shuffled_auc = roc_auc_score(all_labels, all_preds)
        importances[feat_name] = baseline_auc - shuffled_auc
    return dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))


features = ['tempo', 'mode', 'danceability', 'energy', 'loudness', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness', 'valence',
            'hour', 'day_of_week']
target = 'skipped'

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

val_dataset = SessionDataset('../RAW Data/validation_data.parquet', features, target)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

args = argparse.Namespace(
    device=device,
    hidden_units=128,
    maxlen=200,
    dropout_rate=0.3,
    num_blocks=2,
    num_heads=8,
    norm_first=True
)

model = SASRec(feature_no=len(features), args=args).to(device)
model.load_state_dict(torch.load('../Models/sasrec_best.pt', map_location=device))

baseline_auc = evaluate_sequential(model, val_loader, device)
print(f"Baseline AUC: {baseline_auc:.4f}")

print("Calculating importances...")
importances = permutation_importance(model, val_loader, device, features, baseline_auc)
print("\n--- Permutation Feature Importance ---")
for feat, importance in importances.items():
    print(f"{feat}: {importance:.4f}")