from SASRec_lib import SessionDataset, collate_fn, SASRec, train_epoch, evaluate_sequential, log_ablation
import torch
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import argparse
import json
from Loss_functions import get_loss_criterion, parse_args, DrRLLoss, PWTSLoss

features = ['tempo', 'mode', 'danceability', 'energy', 'loudness', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness', 'valence',
            'hour', 'day_of_week', 'actively_selected']
target = 'skipped'

device = torch.device("mps" if torch.mps.is_available() else "cpu")
print(f"Using device: {device}")

args = parse_args()
loss_name = args.loss


# ── Load Best Params ──────────────────────────────────────────────────────────

params_path = f'../Models/sasrec_{loss_name}_best_params.json'
if not os.path.exists(params_path):
    raise FileNotFoundError(f"Best params not found for {loss_name}. Run sasrec_tuning.py --loss {loss_name} first.")

with open(params_path, 'r') as f:
    best_params = json.load(f)

print(f"Loaded best params: {best_params}")

# ── Load Data ─────────────────────────────────────────────────────────────────

print("Loading datasets...")
train_dataset = SessionDataset('../RAW Data/training_data.parquet', features, target)
val_dataset = SessionDataset('../RAW Data/validation_data.parquet', features, target)
print(f"Train sessions: {len(train_dataset)} | Val sessions: {len(val_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

os.makedirs('../Models', exist_ok=True)

# ── Build Model ───────────────────────────────────────────────────────────────

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

if loss_name == "drrl":
    criterion = DrRLLoss(gamma=float(best_params['gamma']))
elif loss_name == "pwts":
    criterion = PWTSLoss()
else:
    criterion = get_loss_criterion(loss_name)

criterion = criterion.to(device)

optimizer = torch.optim.Adam(
    list(model.parameters()) + list(criterion.parameters()),
    lr=float(best_params['lr'])
)

print(f"\nTraining SASRec with best {loss_name.upper()} hyperparameters...\n")

# ── Training Loop ─────────────────────────────────────────────────────────────

best_auc = 0
patience_counter = 0
max_epochs = 100

for epoch in range(max_epochs):
    train_loss = train_epoch(model, train_loader, optimizer, criterion, device, loss_name=loss_name)
    val_auc = evaluate_sequential(model, val_loader, device)

    tqdm.write(f"Epoch {epoch+1} | Loss: {train_loss:.4f} | Val AUC: {val_auc:.4f}")

    if val_auc > best_auc:
        best_auc = val_auc
        torch.save(model.state_dict(), f'../Models/sasrec_{loss_name}_best.pt')
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= 2:
            tqdm.write(f"Early stopping at epoch {epoch+1}")
            break


if args.experiment != 'unnamed':
    log_ablation(
        best_auc,
        experiment_name=loss_name,
        config_description=args.experiment
    )
print(f"\nBest Val AUC: {best_auc:.4f}")
print(f"Model saved to ../Models/sasrec_{loss_name}_best.pt")