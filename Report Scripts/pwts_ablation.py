import argparse
import copy
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'Model Scripts'))

from Model_lib import (SessionDataset, collate_fn, SkipLSTM, train_epoch,
                       evaluate_sequential, LengthBucketSampler, log_test)

features = ['tempo', 'mode', 'danceability', 'energy', 'loudness', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness', 'valence',
            'historical_skip_rate',
            'historical_artist_skip_rate', 'shuffle', 'is_repeat_track', 'same_artist_as_prev']
target = 'skipped'

BATCH_SIZE = 64
MAX_EPOCHS = 100
PATIENCE = 5
SMOOTH_WINDOW = 3
N_SEEDS = 3            # fewer than the 5 used for headline results - this is a
                       # within-ablation comparison, and there are 15 variants
MIN_EPOCHS = 20        # Early stopping is not allowed to fire before this. In the
                       # first run the variant ranking was perfectly separated by
                       # training length - the top 7 variants were exactly the 7
                       # whose seeds all ran >=15 epochs, the bottom 8 exactly the 8
                       # with a seed stopping at 7 or 9 - so the table ranked when
                       # patience happened to trigger rather than the loss constants.
                       # Well-behaved variants peaked between epochs 10 and 35, so a
                       # floor of 20 clears the artefact without capping convergence.

OUT = '../Models/pwts_experiments.csv'
TASK_ID = int(os.environ.get('SLURM_ARRAY_TASK_ID', -1))


##The loss, with every constant exposed

class PWTSConfigurable(nn.Module):
    """PWTS with each weight switchable and each constant settable. With all three
    weights off it reduces exactly to BCEWithLogitsLoss, since the combined weight is
    normalised to mean 1."""

    def __init__(self, position=True, time=True, historical=True,
                 decay=3.0, start=2.0, thresh=0.1, mid=0.5, steep=6.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.position, self.time, self.historical = position, time, historical
        self.decay, self.start, self.thresh = decay, start, thresh
        self.mid, self.steep = mid, steep

    def forward(self, preds, labels, song_pos, lengths, ms_played, track_length,
                historical_skip_rate):
        w = torch.ones_like(preds)

        if self.position:
            pp = song_pos / torch.clamp(lengths, min=1e-8)
            w = w * (1 + torch.exp(-pp * self.decay))

        if self.time:
            pl = ms_played / torch.clamp(track_length, min=1e-8)
            ramp = self.start - (self.start - 1.0) * (pl / self.thresh)
            w = w * torch.where(pl <= self.thresh,
                                torch.clamp(ramp, min=1.0),
                                torch.tensor(1.0, device=pl.device))

        if self.historical:
            w = w * (1 + torch.sigmoid((historical_skip_rate - self.mid) * self.steep))

        w = w / w.mean()
        return (w * self.bce(preds, labels)).mean()


##Variants

DEFAULTS = dict(position=True, time=True, historical=True,
                decay=3.0, start=2.0, thresh=0.1, mid=0.5, steep=6.0)


def variant(**overrides):
    return {**DEFAULTS, **overrides}


VARIANTS = {
    # which weights contribute at all
    'full':            variant(),
    'no_position':     variant(position=False),
    'no_time':         variant(time=False),
    'no_historical':   variant(historical=False),

    # position: 1 + exp(-percentage_pos * decay). Larger decay concentrates the
    # extra weight on the very start of the session; as decay -> 0 the term becomes
    # a constant 2 and stops discriminating by position at all.
    'decay_1':         variant(decay=1.0),
    'decay_5':        variant(decay=5.0),

    # time: how much extra weight a near-instant skip gets (start), and how far into
    # the track that boost persists (thresh). Default thresh=0.1 covers 46% of rows.
    'start_4':         variant(start=4),
    'start_6':         variant(start=6),
    'thresh_0.05':     variant(thresh=0.05),
    'thresh_0.2':       variant(thresh=0.2),

    # historical: sigmoid midpoint and steepness. Mean historical_skip_rate is 0.281
    # and only 20% of rows exceed the default midpoint of 0.5, so the term currently
    # sits mostly in its lower half - mid_0.28 centres it on the data.
    'steep_3':         variant(steep=3.0),

    'steep_9':        variant(steep=9.0),
    'mid_0.37':        variant(mid=0.37),
    'mid_0.2':         variant(mid=0.2),
    'mid_0.7':         variant(mid=0.7),
}

cli = argparse.ArgumentParser()
cli.add_argument('--variant', help='run a single named variant')
cli.add_argument('--list', action='store_true', help='list variants and exit')
cli.add_argument('--seeds', type=int, default=N_SEEDS)
cli.add_argument('--out', default=OUT)
opts, _ = cli.parse_known_args()

if opts.list:
    for n, c in VARIANTS.items():
        changed = {k: v for k, v in c.items() if DEFAULTS[k] != v} or 'defaults'
        print(f'{n:<16} {changed}')
    sys.exit(0)

names = list(VARIANTS)
if opts.variant:
    if opts.variant not in VARIANTS:
        sys.exit(f"Unknown variant '{opts.variant}'. Options: {', '.join(names)}")
    names = [opts.variant]
elif TASK_ID >= 0:
    if TASK_ID >= len(names):
        print(f"Task {TASK_ID} exceeds {len(names)} variants - nothing to do.")
        sys.exit(0)
    names = [names[TASK_ID]]
    opts.out = f'../Models/pwts_experiments_task{TASK_ID}.csv'
    print(f"SLURM array task {TASK_ID}: variant '{names[0]}' -> {opts.out}")


##Setup

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

params_path = '../Models/lstm_pwts_best_params.json'
if not os.path.exists(params_path):
    raise FileNotFoundError(f"{params_path} not found. Run RNN_tuning.py --loss pwts first.")
best_params = json.load(open(params_path))
print(f"SkipLSTM config: {best_params}")

print("Loading datasets...")
train_dataset = SessionDataset('../RAW Data/training_data.parquet', features, target)
val_dataset = SessionDataset('../RAW Data/validation_data.parquet', features, target)
print(f"Train sessions: {len(train_dataset)} | Val sessions: {len(val_dataset)}")

train_lengths = [len(y) for y in train_dataset.labels]
val_order = np.argsort([len(y) for y in val_dataset.labels]).tolist()
val_loader = DataLoader(Subset(val_dataset, val_order), batch_size=BATCH_SIZE,
                        shuffle=False, collate_fn=collate_fn)

os.makedirs('../Models', exist_ok=True)


def run_one(config, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    train_loader = DataLoader(train_dataset, collate_fn=collate_fn,
                              batch_sampler=LengthBucketSampler(train_lengths, BATCH_SIZE))
    model = SkipLSTM(input_size=len(features),
                     hidden_size=int(best_params['hidden_units']),
                     num_layers=int(best_params['num_layers']),
                     dropout=float(best_params['dropout_rate'])).to(device)
    criterion = PWTSConfigurable(**config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(best_params['lr']))

    history, best_smoothed, best_auc, best_epoch = [], -np.inf, np.nan, 0
    patience = 0
    for epoch in range(MAX_EPOCHS):
        train_epoch(model, train_loader, optimizer, criterion, device, loss_name='pwts')
        val_auc, val_ndcg, _ = evaluate_sequential(model, val_loader, device)
        history.append(val_ndcg)
        smoothed = float(np.mean(history[-SMOOTH_WINDOW:]))
        if smoothed > best_smoothed:
            best_smoothed, best_auc, best_epoch = smoothed, val_auc, epoch + 1
            patience = 0
        else:
            patience += 1
            if patience >= PATIENCE and epoch + 1 >= MIN_EPOCHS:
                break
    return best_smoothed, best_auc, best_epoch, len(history)


##Run

results = []
for name in names:
    config = VARIANTS[name]
    changed = {k: v for k, v in config.items() if DEFAULTS[k] != v} or {}
    print(f"\n=== {name}  {changed if changed else '(defaults)'} ===", flush=True)

    ndcgs, aucs, epochs, bests = [], [], [], []
    for seed in range(opts.seeds):
        nd, au, be, ep = run_one(config, seed)
        ndcgs.append(nd); aucs.append(au); epochs.append(ep); bests.append(be)
        print(f"  seed {seed}: NDCG {nd:.4f} | AUC {au:.4f} | best epoch {be} of {ep}",
              flush=True)

    ndcgs, aucs = np.array(ndcgs), np.array(aucs)
    print(f"  -> NDCG {ndcgs.mean():.4f} +/- {ndcgs.std():.4f} | "
          f"AUC {aucs.mean():.4f} +/- {aucs.std():.4f}")

    results.append({
        'variant': name,
        'position': config['position'], 'time': config['time'],
        'historical': config['historical'],
        'decay': config['decay'], 'start': config['start'], 'thresh': config['thresh'],
        'mid': config['mid'], 'steep': config['steep'],
        'changed_from_default': json.dumps(changed),
        'n_seeds': opts.seeds,
        'val_ndcg_mean': round(float(ndcgs.mean()), 4),
        'val_ndcg_std': round(float(ndcgs.std()), 4),
        'val_auc_mean': round(float(aucs.mean()), 4),
        'val_auc_std': round(float(aucs.std()), 4),
        'epochs_run': str(epochs), 'best_epoch': str(bests),
    })
    for row in results[-1:]:
        log_test(row, log_path=opts.out)


##Summary

if len(results) > 1:
    df = __import__('pandas').DataFrame(results).sort_values('val_ndcg_mean',
                                                             ascending=False)
    print(f"\n{'='*72}\nPWTS ABLATION - SkipLSTM, {opts.seeds} seeds each\n{'='*72}")
    print(df[['variant', 'changed_from_default', 'val_ndcg_mean', 'val_ndcg_std',
              'val_auc_mean']].to_string(index=False))
    full = df[df.variant == 'full']
    if not full.empty:
        print(f"\nfull PWTS = {full.val_ndcg_mean.iloc[0]:.4f}; differences smaller "
              f"than ~2x the seed sd should be read as equivalent.")
print(f"\nWritten to {opts.out}")
