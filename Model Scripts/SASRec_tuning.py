from SASRec_lib import SessionDataset, collate_fn, SASRec, train_epoch, evaluate_sequential
import torch
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import argparse
import itertools
import signal
from Loss_functions import get_loss_criterion, parse_args, DrRLLoss, PWTSLoss

def handle_interrupt(sig, frame):
    print("\n\nCtrl+C detected. What do you want to do?")
    print("  [r] Resume later (save progress and exit)")
    print("  [x] Reset everything and exit")
    choice = input("Enter choice: ").strip().lower()
    if choice == 'r':
        print("Progress saved. Rerun the script to resume.")
        exit(0)
    elif choice == 'x':
        if os.path.exists('../Models/sasrec_grid_search_results.csv'):
            os.remove('../Models/sasrec_grid_search_results.csv')
            print("Progress reset. Rerun the script to start fresh.")
        exit(0)
    else:
        print("Invalid choice, resuming run...")

signal.signal(signal.SIGINT, handle_interrupt)

features = ['tempo', 'mode', 'danceability', 'energy', 'loudness', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness', 'valence',
            'hour', 'day_of_week', 'actively_selected']
target = 'skipped'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

loss_name = parse_loss_arg()

print("Loading datasets...")
train_dataset = SessionDataset('../RAW Data/training_data.parquet', features, target)
val_dataset = SessionDataset('../RAW Data/validation_data.parquet', features, target)
print(f"Train sessions: {len(train_dataset)} | Val sessions: {len(val_dataset)}")

os.makedirs('../Models', exist_ok=True)

# ── Grid Search ───────────────────────────────────────────────────────────────

param_grid = {
    'hidden_units': [64, 128, 256],
    'dropout_rate': [0.1, 0.3, 0.5],
    'num_blocks':   [1, 2, 3],
    'num_heads':    [4, 8, 16],
    'lr':           [0.001, 0.0001],
}

if loss_name == "drrl":
    param_grid["gamma"]= [2,3,4]


lr_to_epochs = {
    0.001:   20,
    0.0001:  40
}

keys = list(param_grid.keys())
combinations = [
    combo for combo in itertools.product(*param_grid.values())
    if dict(zip(keys, combo))['hidden_units'] % dict(zip(keys, combo))['num_heads'] == 0
]
print(f"\nRunning grid search over {len(combinations)} combinations...\n")

completed_csv = f'../Models/sasrec_grid_search_{loss_name}_results.csv'
if os.path.exists(completed_csv):
    completed_df = pd.read_csv(completed_csv)
    completed_params = completed_df[keys].to_dict('records')
    results = completed_df.to_dict('records')
else:
    completed_params = []
    results = []

ran_to_end_count = 0
combinations_completed = 0

for combo in combinations:
    ran_to_end = 0
    params = dict(zip(keys, combo))

    if any(all(params[k] == c[k] for k in keys) for c in completed_params):
        print(f"Skipping already completed: {params}")
        continue

    print(f"Trying: {params}")

    max_epochs = lr_to_epochs[params['lr']]

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    args = argparse.Namespace(
        device=device,
        hidden_units=params['hidden_units'],
        maxlen=200,
        dropout_rate=params['dropout_rate'],
        num_blocks=params['num_blocks'],
        num_heads=params['num_heads'],
        norm_first=True
    )

    model = SASRec(feature_no=len(features), args=args).to(device)

    if loss_name == "drrl":
        criterion = DrRLLoss(gamma=params["gamma"])
    elif loss_name == "pwts":
        criterion = PWTSLoss()   
    else: 
        criterion = get_loss_criterion(loss_name)
    
    criterion = criterion.to(device)
        
    optimizer = torch.optim.Adam(
        list(model.parameters())+list(criterion.parameters()),
        lr=params['lr']
    )


    best_auc = 0
    patience_counter = 0

    for epoch in range(max_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, loss_name=loss_name)
        val_auc = evaluate_sequential(model, val_loader, device)
        tqdm.write(f"Epoch {epoch+1} | Loss: {train_loss:.4f} | Val AUC: {val_auc:.4f}")
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 3:
                break
    else:
        ran_to_end = 1
        ran_to_end_count += 1
    

    print(f"\nBest Val AUC: {best_auc:.4f}")
    results.append({**params, 'val_auc': best_auc, 'ran_to_end': ran_to_end})

    pd.DataFrame(results).sort_values('val_auc', ascending=False).to_csv(
        completed_csv, index=False
    )
    combinations_completed += 1
    print(f'\n{combinations_completed+len(completed_params)} combinations completed out of {len(combinations)}\n')

# ── Summary ───────────────────────────────────────────────────────────────────

results_df = pd.DataFrame(results).sort_values('val_auc', ascending=False)
print("\n--- Grid Search Results ---")
print(results_df.to_string(index=False))

best_params = results_df.iloc[0].to_dict()
print(f"\nBest config: {best_params}")
print(f'{ran_to_end_count} combinations ran to maximum epoch before converging')

pd.Series(best_params).to_json(f'../Models/sasrec_{loss_name}_best_params.json')
print(f"Best params saved to ../Models/sasrec_{loss_name}_best_params.json")