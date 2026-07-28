import numpy as np
import torch
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import os
import argparse
import random
import itertools
import signal
from Loss_functions import get_loss_criterion, parse_args, DrRLLoss, PWTSLoss
import sys

args = parse_args()
loss_name = args.loss

def handle_interrupt(sig, frame):
    sys.stdout.write("\n\nCtrl+C detected. What do you want to do?\n")
    sys.stdout.write("  [r] Resume later (save progress and exit)\n")
    sys.stdout.write("  [x] Reset everything and exit\n")
    sys.stdout.flush()
    
    choice = sys.stdin.readline().strip().lower()
    
    if choice == 'r':
        sys.stdout.write("Progress saved. Rerun the script to resume.\n")
        sys.stdout.flush()
        os._exit(0)
    elif choice == 'x':
        csv_path = f'../Models/sasrec_grid_search_{loss_name}_results.csv'
        if os.path.exists(csv_path):
            os.remove(csv_path)
            sys.stdout.write("Progress reset. Rerun the script to start fresh.\n")
            sys.stdout.flush()
        os._exit(0)
    else:
        sys.stdout.write("Invalid choice, resuming run...\n")
        sys.stdout.flush()

##Dataset

class SessionDataset(Dataset):
    def __init__(self, parquet_path, features, target):
        df = pd.read_parquet(parquet_path)
        df = df.dropna(subset=features)
        self.sessions = []
        self.labels = []
        self.song_pos = []
        self.ms_played = []
        self.track_length = []
        self.historical_skip_rate = []
        
        for session_id, session in df.groupby("session_id"):
            session = session.sort_values("ts")
            x = torch.tensor(session[features].values, dtype=torch.float32)
            y = torch.tensor(session[target].values, dtype=torch.float32)
            self.sessions.append(x)
            self.labels.append(y)
            self.song_pos.append(torch.tensor(session['song_pos'].values, dtype=torch.float32))
            self.ms_played.append(torch.tensor(session['ms_played'].values, dtype=torch.float32))
            self.track_length.append(torch.tensor(session['track_length'].values, dtype=torch.float32))
            self.historical_skip_rate.append(torch.tensor(session['historical_skip_rate'].values, dtype=torch.float32))

    def __len__(self):
        return len(self.sessions)

    def __getitem__(self, idx):
        return (self.sessions[idx], self.labels[idx], self.song_pos[idx],
                self.ms_played[idx], self.track_length[idx], 
                self.historical_skip_rate[idx])


def collate_fn(batch):
    sessions, labels, song_pos, ms_played, track_length, historical_skip_rate = zip(*batch)
    sessions_padded = nn.utils.rnn.pad_sequence(sessions, batch_first=True)
    labels_padded = nn.utils.rnn.pad_sequence(labels, batch_first=True)
    song_pos_padded = nn.utils.rnn.pad_sequence(song_pos, batch_first=True)
    ms_played_padded = nn.utils.rnn.pad_sequence(ms_played, batch_first=True)
    track_length_padded = nn.utils.rnn.pad_sequence(track_length, batch_first=True)
    historical_skip_rate_padded = nn.utils.rnn.pad_sequence(historical_skip_rate, batch_first=True)
    lengths = torch.tensor([len(s) for s in sessions])
    return sessions_padded, labels_padded,song_pos_padded, ms_played_padded, track_length_padded,historical_skip_rate_padded, lengths

##Model

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

def log_ablation(best_auc, experiment_name, config_description):
    log_path = '../Models/test_log.csv'
    entry = pd.DataFrame([{
        'experiment': experiment_name,
        'config': config_description,
        'val_auc': f'{best_auc:.4f}',
        'timestamp': pd.Timestamp.now()
    }])
    if os.path.exists(log_path):
        existing = pd.read_csv(log_path)
        combined = pd.concat([existing, entry], ignore_index=True)
    else:
        combined = entry
    combined.to_csv(log_path, index=False)

##Training

def train_epoch(model, loader, optimizer, criterion, device, loss_name='bce'):
    model.train()
    total_loss = 0
    progress = tqdm(loader, desc="Training", leave=False)
    for sessions, labels, song_pos, ms_played, track_length, historical_skip_rate, lengths in progress:
        sessions, labels = sessions.to(device), labels.to(device)
        if loss_name == 'pwts':
            lengths_expanded = torch.zeros_like(labels)
            for i, length in enumerate(lengths):
                lengths_expanded[i, :length] = length
            song_pos = song_pos.to(device)
            ms_played = ms_played.to(device)
            track_length = track_length.to(device)
            lengths_expanded = lengths_expanded.to(device)
            historical_skip_rate = historical_skip_rate.to(device)

        optimizer.zero_grad()
        preds = model(sessions, lengths)
        mask = torch.zeros_like(labels, dtype=torch.bool)
        for i, length in enumerate(lengths):
            mask[i, :length] = True

        if loss_name == "pwts":
            loss = criterion(preds[mask], labels[mask], song_pos[mask], lengths_expanded[mask], ms_played[mask], track_length[mask], historical_skip_rate[mask])
        elif loss_name == "drrl":
            criterion.update_beta(preds[mask], labels[mask])
            loss = criterion(preds[mask], labels[mask])
        else:            
            loss = criterion(preds[mask], labels[mask])

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        progress.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / len(loader)


def evaluate_sequential(model, loader, device, min_context=5, min_skips_in_context=1, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for sessions, labels, song_pos, ms_played, track_length, historical_skip_rate, lengths in loader:
            sessions = sessions.to(device)
            preds = model(sessions, lengths)
            for i, length in enumerate(lengths):
                length = length.item()
                if length < min_context + 1:
                    continue
                context_len = torch.randint(min_context, length, (1,)).item()
                context_skips = labels[i, :context_len].sum().item()
                if context_skips < min_skips_in_context:
                    continue
                all_preds.extend(preds[i, context_len:length].cpu().numpy())
                all_labels.extend(labels[i, context_len:length].numpy())
    return roc_auc_score(all_labels, all_preds)