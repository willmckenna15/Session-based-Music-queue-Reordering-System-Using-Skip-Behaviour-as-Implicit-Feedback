import torch
from torch.utils.data import DataLoader, Subset
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
import os
import sys
import itertools
import signal
from Model_lib import (SessionDataset, train_epoch, evaluate_sequential_tuning, SkipLSTM,
                       collate_fn, LengthBucketSampler)
from Loss_functions import get_loss_criterion, parse_args, DrRLLoss, PWTSLoss

features = ['tempo', 'mode', 'danceability', 'energy', 'loudness', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness', 'valence',
            'historical_skip_rate',
            'historical_artist_skip_rate', 'shuffle', 'is_repeat_track', 'same_artist_as_prev']
target = 'skipped'

args = parse_args()
loss_name = args.loss

SEED = 0

# Set by SLURM inside a job array; -1 means an ordinary serial run
TASK_ID = int(os.environ.get('SLURM_ARRAY_TASK_ID', -1))

completed_csv = f'../Models/LSTM_grid_search_{loss_name}_results.csv'


def handle_interrupt(sig, frame):
    print("\n\nCtrl+C detected. What do you want to do?")
    print("  [r] Resume later (save progress and exit)")
    print("  [x] Reset everything and exit")
    choice = input("Enter choice: ").strip().lower()
    if choice == 'r':
        print("Progress saved. Rerun the script to resume.")
        exit(0)
    elif choice == 'x':
        if os.path.exists(completed_csv):
            os.remove(completed_csv)
            print(f"Progress reset ({completed_csv} deleted). Rerun to start fresh.")
        else:
            print(f"Nothing to reset - {completed_csv} does not exist.")
        exit(0)
    else:
        print("Invalid choice, resuming run...")


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

val_order = np.argsort([len(y) for y in val_dataset.labels]).tolist()
val_eval_dataset = Subset(val_dataset, val_order)

train_lengths = [len(y) for y in train_dataset.labels]
BATCH_SIZE = 64

os.makedirs('../Models', exist_ok=True)

##Grid search

param_grid = {
    'hidden_units': [64, 128, 256],
    'dropout_rate': [0.1, 0.3, 0.5],
    'num_layers':   [1, 2, 3],
    'lr':           [0.001, 0.0001],
}

if loss_name == "drrl":
    param_grid["gamma"] = [2, 3, 4]

lr_to_epochs = {
    0.001:   20,
    0.0001:  40
}

keys = list(param_grid.keys())
combinations = list(itertools.product(*param_grid.values()))
print(f"\nRunning grid search over {len(combinations)} combinations...\n")

if TASK_ID >= 0:
    if TASK_ID >= len(combinations):
        print(f"Task {TASK_ID} exceeds {len(combinations)} combinations - nothing to do.")
        sys.exit(0)
    combinations = [combinations[TASK_ID]]
    completed_csv = f'../Models/LSTM_grid_{loss_name}_task{TASK_ID}.csv'
    print(f"SLURM array task {TASK_ID}: 1 config -> {completed_csv}")

if os.path.exists(completed_csv):
    completed_df = pd.read_csv(completed_csv)
    completed_params = completed_df[keys].to_dict('records')
    results = completed_df.to_dict('records')
else:
    completed_params = []
    results = []

combinations_completed = 0

SMOOTH_WINDOW = 3
PATIENCE = 5

for combo in combinations:
    params = dict(zip(keys, combo))

    if any(all(params[k] == c[k] for k in keys) for c in completed_params):
        print(f"Skipping already completed: {params}")
        continue

    print(f"Trying: {params}")

    max_epochs = lr_to_epochs[params['lr']]

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    train_loader = DataLoader(train_dataset, collate_fn=collate_fn,
                              batch_sampler=LengthBucketSampler(train_lengths, BATCH_SIZE))
    val_loader = DataLoader(val_eval_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=collate_fn)

    model = SkipLSTM(
        input_size=len(features),
        hidden_size=params['hidden_units'],
        num_layers=params['num_layers'],
        dropout=params['dropout_rate'],
    ).to(device)

    if loss_name == "drrl":
        criterion = DrRLLoss(gamma=params["gamma"])
    elif loss_name == "pwts":
        criterion = PWTSLoss()
    else:
        criterion = get_loss_criterion(loss_name)

    criterion = criterion.to(device)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(criterion.parameters()),
        lr=params['lr']
    )

    history = []
    best_smoothed = -np.inf
    best_raw = np.nan
    best_epoch = 0
    patience_counter = 0

    for epoch in range(max_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device,
                                 loss_name=loss_name)
        val_ndcg = evaluate_sequential_tuning(model, val_loader, device)
        history.append(val_ndcg)
        smoothed = float(np.mean(history[-SMOOTH_WINDOW:]))
        tqdm.write(f"Epoch {epoch+1} | Loss: {train_loss:.4f} | "
                   f"Val NDCG: {val_ndcg:.4f} | smoothed: {smoothed:.4f}")
        if smoothed > best_smoothed:
            best_smoothed = smoothed
            best_raw = val_ndcg
            best_epoch = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    epochs_run = len(history)

    print(f"\nBest smoothed NDCG: {best_smoothed:.4f} at epoch {best_epoch} "
          f"of {epochs_run} run")
    results.append({**params, 'val_ndcg': best_smoothed, 'val_ndcg_raw': best_raw,
                    'best_epoch': best_epoch, 'epochs_run': epochs_run,
                    'max_epochs': max_epochs, 'seed': SEED})

    pd.DataFrame(results).sort_values('val_ndcg', ascending=False).to_csv(
        completed_csv, index=False
    )
    combinations_completed += 1
    print(f'\n{combinations_completed+len(completed_params)} combinations completed out of {len(combinations)}\n')

##Summary

results_df = pd.DataFrame(results).sort_values('val_ndcg', ascending=False)
print("\n--- Grid Search Results ---")
print(results_df.to_string(index=False))

if results_df.empty:
    print("\nNo results - every combination was already complete.")
else:
    best_params = results_df.iloc[0].to_dict()
    print(f"\nBest config: {best_params}")
    # If best_epoch sits close to epochs_run, patience is cutting runs off while they
    # are still improving; if epochs_run hits max_epochs, the cap is binding.
    if 'epochs_run' in results_df:
        at_cap = int((results_df.epochs_run >= results_df.max_epochs).sum())
        still_climbing = int((results_df.best_epoch >= results_df.epochs_run - 1).sum())
        print(f'{at_cap} of {len(results_df)} configs reached max_epochs')
        print(f'{still_climbing} of {len(results_df)} were still improving when stopped')

    pd.Series(best_params).to_json(f'../Models/lstm_{loss_name}_best_params.json')
    print(f"Best params saved to ../Models/lstm_{loss_name}_best_params.json")
