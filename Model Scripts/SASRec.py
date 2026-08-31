"""Final training for SASRec using the tuned hyperparameters.

Selection and reporting both use NDCG@5 on a 3-epoch moving average, matching the
tuning stage - selecting on one metric and reporting another is not defensible, and
the raw per-epoch value swings ~0.03, which is larger than the differences being
compared. AUC is recorded at the selected epoch so both describe the same model.
"""

from Model_lib import (SessionDataset, collate_fn, SASRec, train_epoch,
                       evaluate_sequential, log_test, LengthBucketSampler)
import torch
import numpy as np
import random
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import os
import argparse
import json
import copy
from Loss_functions import get_loss_criterion, parse_args, DrRLLoss, PWTSLoss

_fs = argparse.ArgumentParser(add_help=False)
_fs.add_argument('--feature-set', type=str, default=None,
                 choices=['audio-only', 'behavioural-only', 'all'],
                 help='Which features to use (feature ablation)')
_requested = _fs.parse_known_args()[0].feature_set
is_ablation = _requested is not None          # an ordinary run passes nothing
feature_set = _requested or 'all'

audio_features = ['tempo', 'mode', 'danceability', 'energy', 'loudness',
                  'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence']

behavioural_features = ['historical_skip_rate', 'historical_artist_skip_rate',
                        'shuffle', 'is_repeat_track', 'same_artist_as_prev']

if feature_set == 'audio-only':
    features = audio_features
elif feature_set == 'behavioural-only':
    features = behavioural_features
else:
    features = audio_features + behavioural_features
target = 'skipped'

BATCH_SIZE = 64
MAX_EPOCHS = 100
PATIENCE = 5
SMOOTH_WINDOW = 3
N_RUNS = 5

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

args = parse_args()
loss_name = args.loss

params_path = f'../Models/sasrec_{loss_name}_best_params.json'
if not os.path.exists(params_path):
    raise FileNotFoundError(f"Best params not found for {loss_name}. "
                            f"Run SASRec_tuning.py --loss {loss_name} first.")

with open(params_path, 'r') as f:
    best_params = json.load(f)

print(f"Loaded best params: {best_params}")
print(f"Feature set: {feature_set} ({len(features)} features)")

# Any run that passes --feature-set writes tagged checkpoints, including
# --feature-set all. Keying the tag on the value instead would let an "all" run
# write to the untagged paths and overwrite the headline models evaluate.py reads.
TAG = f'_{feature_set.replace("-", "")}' if is_ablation else ''

# NOTE for interpretation: the status channel (status_emb) carries observed skip
# outcomes and is NOT part of `features`, so an "audio-only" model here still
# receives skip momentum. It is not audio-only in the sense the sklearn baselines
# are - those are the clean test of the research question.

print("Loading datasets...")
train_dataset = SessionDataset('../RAW Data/training_data.parquet', features, target)
val_dataset = SessionDataset('../RAW Data/validation_data.parquet', features, target)
print(f"Train sessions: {len(train_dataset)} | Val sessions: {len(val_dataset)}")

train_lengths = [len(y) for y in train_dataset.labels]

# Length-sorted evaluation order: batches otherwise pad to their longest session and
# validation lengths span 7 to 786. Scores are unchanged - evaluate_sequential
# averages within each session before averaging across sessions.
val_order = np.argsort([len(y) for y in val_dataset.labels]).tolist()
val_loader = DataLoader(Subset(val_dataset, val_order), batch_size=BATCH_SIZE,
                        shuffle=False, collate_fn=collate_fn)

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


def run_training(seed, verbose=True):
    """Train one model. Returns the metrics at the best epoch and its weights."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Length-bucketed batches: 6.2x less padded compute, batch order reshuffled
    # each epoch so training stays stochastic
    train_loader = DataLoader(train_dataset, collate_fn=collate_fn,
                              batch_sampler=LengthBucketSampler(train_lengths, BATCH_SIZE))

    model = SASRec(feature_no=len(features), args=args_mod).to(device)
    criterion = build_criterion()
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(criterion.parameters()),
        lr=float(best_params['lr'])
    )

    history = []
    best_smoothed = -np.inf
    best = {'val_ndcg_raw': np.nan, 'val_auc': np.nan, 'epoch': 0}
    best_state = None
    patience_counter = 0

    for epoch in range(MAX_EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device,
                                 loss_name=loss_name)
        val_auc, val_ndcg, n_sessions = evaluate_sequential(model, val_loader, device)
        history.append(val_ndcg)
        smoothed = float(np.mean(history[-SMOOTH_WINDOW:]))

        if verbose:
            tqdm.write(f"Epoch {epoch+1} | Loss: {train_loss:.4f} | "
                       f"NDCG: {val_ndcg:.4f} | smoothed: {smoothed:.4f} | AUC: {val_auc:.4f}")

        if smoothed > best_smoothed:
            best_smoothed = smoothed
            best = {'val_ndcg_raw': val_ndcg, 'val_auc': val_auc, 'epoch': epoch + 1}
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                if verbose:
                    tqdm.write(f"Early stopping at epoch {epoch+1}")
                break

    best.update(val_ndcg=best_smoothed, epochs_run=len(history), n_sessions=n_sessions)
    return best, best_state


# -- Main ---------------------------------------------------------------------

runs = []
best_overall = (-np.inf, None)

for seed in range(N_RUNS):
    print(f"\n=== Run {seed+1}/{N_RUNS} (seed {seed}) ===\n")
    result, state = run_training(seed)
    runs.append(result)
    print(f"Run {seed+1}: NDCG {result['val_ndcg']:.4f} | AUC {result['val_auc']:.4f} "
          f"| best epoch {result['epoch']} of {result['epochs_run']}")
    # every seed is kept: evaluate.py averages the test metrics over them, which is
    # what puts error bars on the final comparison
    torch.save(state, f'../Models/sasrec_{loss_name}{TAG}_seed{seed}.pt')
    if result['val_ndcg'] > best_overall[0]:
        best_overall = (result['val_ndcg'], state)

# Also save the best seed under a stable name
torch.save(best_overall[1], f'../Models/sasrec_{loss_name}{TAG}_best.pt')

ndcgs = np.array([r['val_ndcg'] for r in runs])
aucs = np.array([r['val_auc'] for r in runs])

print(f"\n=== SASRec / {loss_name.upper()} across {N_RUNS} seeds ===")
print(f"NDCG@5 : {ndcgs.mean():.4f} +/- {ndcgs.std():.4f}   {np.round(ndcgs, 4).tolist()}")
print(f"AUC    : {aucs.mean():.4f} +/- {aucs.std():.4f}   {np.round(aucs, 4).tolist()}")
print(f"epochs : {[r['epochs_run'] for r in runs]}  (best at {[r['epoch'] for r in runs]})")
print(f"Model saved to ../Models/sasrec_{loss_name}{TAG}_best.pt")

LOG = ('../Models/feature_ablation_study.csv' if is_ablation
       else '../Models/test_log.csv')

log_test({
    'model': 'SASRec',
    'feature_set': feature_set,
    'n_features': len(features),
    'session_auc': round(float(aucs.mean()), 4),
    'session_ndcg5': round(float(ndcgs.mean()), 4),
    'split': 'validation',
    'loss': loss_name,
    'experiment': args.experiment,
    'n_seeds': N_RUNS,
    'val_ndcg_mean': round(float(ndcgs.mean()), 4),
    'val_ndcg_std': round(float(ndcgs.std()), 4),
    'val_auc_mean': round(float(aucs.mean()), 4),
    'val_auc_std': round(float(aucs.std()), 4),
    'epochs_run': str([r['epochs_run'] for r in runs]),
    'best_epoch': str([r['epoch'] for r in runs]),
    'params': json.dumps({k: best_params[k] for k in
                          ('hidden_units', 'dropout_rate', 'num_blocks', 'num_heads', 'lr')
                          if k in best_params}),
}, log_path=LOG)
