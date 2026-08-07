from torch.utils.data import DataLoader
from SASRec_lib import SessionDataset, collate_fn, evaluate_sequential, SASRec
import torch
import numpy as np
import argparse
import json

features = ['tempo', 'mode', 'danceability', 'energy', 'loudness', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness', 'valence',
            'historical_skip_rate',
            'historical_artist_skip_rate', 'shuffle', 'is_repeat_track', 'same_artist_as_prev']
target = 'skipped'

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

print("Loading datasets...")
train_dataset = SessionDataset('../RAW Data/training_data.parquet', features, target)
val_dataset = SessionDataset('../RAW Data/validation_data.parquet', features, target)
print(f"Train sessions: {len(train_dataset)} | Val sessions: {len(val_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

with open('../Models/sasrec_pwts_best_params.json', 'r') as f:
    best_params = json.load(f)

args_mod = argparse.Namespace(
    device=device,
    hidden_units=int(best_params['hidden_units']),
    maxlen=200,
    dropout_rate=float(best_params['dropout_rate']),
    num_blocks=int(best_params['num_blocks']),
    num_heads=int(best_params['num_heads']),
    norm_first=True
)

model = SASRec(feature_no=len(features), args=args_mod).to(device)
model.load_state_dict(torch.load('../Models/sasrec_bce_best.pt', map_location=device))
model.eval()

train_loader_eval = DataLoader(train_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
train_session_auc, n = evaluate_sequential(model, train_loader_eval, device, min_window_length=10)
print(f"Session-wise AUC on TRAINING set: {train_session_auc:.4f} across {n} sessions")
