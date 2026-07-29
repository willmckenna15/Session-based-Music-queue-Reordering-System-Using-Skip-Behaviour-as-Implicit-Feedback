from SASRec_lib import SessionDataset, collate_fn, SASRec, train_epoch, evaluate_sequential, log_test
import torch
import numpy as np
import random
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

params_path = f'../Models/sasrec_{loss_name}_best_params.json'
if not os.path.exists(params_path):
    raise FileNotFoundError(f"Best params not found for {loss_name}. Run sasrec_tuning.py --loss {loss_name} first.")

with open(params_path, 'r') as f:
    best_params = json.load(f)

print(f"Loaded best params: {best_params}")

print("Loading datasets...")
train_dataset = SessionDataset('../RAW Data/training_data.parquet', features, target)
val_dataset = SessionDataset('../RAW Data/validation_data.parquet', features, target)
print(f"Train sessions: {len(train_dataset)} | Val sessions: {len(val_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

os.makedirs('../Models', exist_ok=True)

args_mod = argparse.Namespace(
    device=device,
    hidden_units=int(best_params['hidden_units']),
    maxlen=200,
    dropout_rate=float(best_params['dropout_rate']),
    num_blocks=int(best_params['num_blocks']),
    num_heads=int(best_params['num_heads']),
    norm_first=True
)


def build_criterion():
    if loss_name == "drrl":
        return DrRLLoss(gamma=float(best_params['gamma'])).to(device)
    elif loss_name == "pwts":
        return PWTSLoss().to(device)
    else:
        return get_loss_criterion(loss_name).to(device)


def run_training(seed=None, verbose=True):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    model = SASRec(feature_no=len(features), args=args_mod).to(device)
    criterion = build_criterion()
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(criterion.parameters()),
        lr=float(best_params['lr'])
    )

    best_auc = 0
    patience_counter = 0

    for epoch in range(100):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, loss_name=loss_name)
        val_auc = evaluate_sequential(model, val_loader, device)

        if verbose:
            tqdm.write(f"Epoch {epoch+1} | Loss: {train_loss:.4f} | Val AUC: {val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            if verbose:
                torch.save(model.state_dict(), f'../Models/sasrec_{loss_name}_best.pt')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 2:
                if verbose:
                    tqdm.write(f"Early stopping at epoch {epoch+1}")
                break

    return best_auc


# ── Main ──────────────────────────────────────────────────────────────────────

if args.experiment != 'unnamed':
    N_RUNS = 3
    run_aucs = []

    for run_idx in range(N_RUNS):
        print(f"\n=== Test run {run_idx+1}/{N_RUNS} ===\n")
        auc = run_training(seed=run_idx, verbose=True)
        run_aucs.append(auc)
        print(f"Run {run_idx+1} best AUC: {auc:.4f}")

    mean_auc = np.mean(run_aucs)
    std_auc = np.std(run_aucs)

    print(f"\n=== summary across {N_RUNS} runs ===")
    print(f"AUCs: {[f'{a:.4f}' for a in run_aucs]}")
    print(f"Mean: {mean_auc:.4f} | Std: {std_auc:.4f}")

    log_test(
        mean_auc,
        std_auc,
        experiment_name=loss_name,
        config_description=args.experiment
    )

else:
    print(f"\nTraining SASRec with best {loss_name.upper()} hyperparameters...\n")
    best_auc = run_training(seed=None, verbose=True)
    print(f"\nBest Val AUC: {best_auc:.4f}")
    print(f"Model saved to ../Models/sasrec_{loss_name}_best.pt")