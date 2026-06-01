import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import os

# ── Dataset ───────────────────────────────────────────────────────────────────

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


# ── Model ─────────────────────────────────────────────────────────────────────

class SkipLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
        super(SkipLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        out = self.sigmoid(self.fc(output)).squeeze(-1)
        return out


# ── Training ──────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    progress = tqdm(loader, desc="Training", leave=False)
    for sessions, labels, lengths in progress:
        sessions, labels = sessions.to(device), labels.to(device)

        optimizer.zero_grad()
        preds = model(sessions, lengths)

        mask = torch.zeros_like(labels, dtype=torch.bool)
        for i, length in enumerate(lengths):
            mask[i, :length] = True
        loss = criterion(preds[mask], labels[mask])

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        progress.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(loader)


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

                if length < min_context + 1:
                    continue

                context_len = torch.randint(min_context, length, (1,)).item()

                context_skips = labels[i, :context_len].sum().item()
                if context_skips < min_skips_in_context:
                    continue

                all_preds.extend(preds[i, context_len:length].cpu().numpy())
                all_labels.extend(labels[i, context_len:length].numpy())

    return roc_auc_score(all_labels, all_preds)


# ── Main ──────────────────────────────────────────────────────────────────────

features = ['tempo', 'mode', 'danceability', 'energy', 'loudness', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness', 'valence',
            'hour', 'day_of_week']
target = 'skipped'

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")


print("Loading datasets...")
train_dataset = SessionDataset('../RAW Data/training_data.parquet', features, target)
val_dataset = SessionDataset('../RAW Data/validation_data.parquet', features, target)
print(f"Train sessions: {len(train_dataset)} | Val sessions: {len(val_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)

model = SkipLSTM(input_size=len(features), hidden_size=64, num_layers=2, dropout=0.3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

os.makedirs('../Models', exist_ok=True)

best_auc = 0
patience = 5
patience_counter = 0

for epoch in tqdm(range(100), desc="Epochs"):
    train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
    val_auc = evaluate_sequential(model, val_loader, device)

    tqdm.write(f"Epoch {epoch+1} | Loss: {train_loss:.4f} | Val AUC: {val_auc:.4f}")

    if val_auc > best_auc:
        best_auc = val_auc
        torch.save(model.state_dict(), '../Models/lstm_best.pt')
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            tqdm.write(f"Early stopping at epoch {epoch+1}")
            break

print(f"\nBest Val AUC: {best_auc:.4f}")