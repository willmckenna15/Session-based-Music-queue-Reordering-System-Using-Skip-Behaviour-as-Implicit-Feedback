import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np

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

class SkipLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
        super(SkipLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        return self.sigmoid(self.fc(output)).squeeze(-1)

def evaluate_sequential(model, loader, device, min_context=5, min_skips_in_context=1, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for sessions, labels, lengths in loader:
            sessions = sessions.to(device)
            preds = model(sessions, lengths)

            for i, length in enumerate(lengths):
                length = length.item()
                
                # Random split point — must leave at least 1 song to predict on
                if length < min_context + 1:
                    continue
                    
                context_len = torch.randint(min_context, length, (1,)).item()

                # Check minimum skips in context segment
                context_skips = labels[i, :context_len].sum().item()
                if context_skips < min_skips_in_context:
                    continue

                # Only evaluate on prediction segment
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

# ── Main ──────────────────────────────────────────────────────────────────────

features = ['tempo', 'mode', 'danceability', 'energy', 'loudness', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness', 'valence',
            'hour', 'day_of_week']
target = 'skipped'

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

val_dataset = SessionDataset('../RAW Data/validation_data.parquet', features, target)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)

model = SkipLSTM(input_size=len(features), hidden_size=64, num_layers=2, dropout=0.3).to(device)
model.load_state_dict(torch.load('../Models/lstm_best.pt', map_location=device))

baseline_auc = evaluate_sequential(model, val_loader, device)
print(f"Baseline AUC: {baseline_auc:.4f}")

importances = permutation_importance(model, val_loader, device, features, baseline_auc)
print("\n--- Permutation Feature Importance ---")
for feat, importance in importances.items():
    print(f"{feat}: {importance:.4f}")