"""Run the best model on one test session and print the reordered queue.

    python3 demo_reorder.py                 # auto-pick a readable session
    python3 demo_reorder.py --session <id>  # a specific one
    python3 demo_reorder.py --context 6     # set the split point
"""

import argparse
import json

import numpy as np
import pandas as pd
import torch

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'Model Scripts'))

from Model_lib import SkipLSTM, ndcg_at_k, fast_auc, valid_splits, pick

FEATURES = ['tempo', 'mode', 'danceability', 'energy', 'loudness', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness', 'valence',
            'historical_skip_rate', 'historical_artist_skip_rate', 'shuffle',
            'is_repeat_track', 'same_artist_as_prev']
CKPT = '../Models/lstm_pwts_best.pt'
PARAMS = '../Models/lstm_pwts_best_params.json'
DATA = '../RAW Data/testing_data.parquet'

cli = argparse.ArgumentParser()
cli.add_argument('--session', default=None)
cli.add_argument('--context', type=int, default=13)
cli.add_argument('--min-len', type=int, default=20)
cli.add_argument('--max-len', type=int, default=40)
cli.add_argument('--pick-seed', type=int, default=None)
opts = cli.parse_args()

df = pd.read_parquet(DATA, columns=FEATURES + ['skipped', 'session_id', 'ts',
                                               'Track Name', 'Artist Name'])
df = df.dropna(subset=FEATURES)

if opts.session:
    sid = opts.session
else:
    size = df.groupby('session_id').size()
    skips = df.groupby('session_id')['skipped'].sum()
    ok = size[(size >= opts.min_len) & (size <= opts.max_len) &
              (skips >= 3) & (skips <= size - 3)]
    if ok.empty:
        raise SystemExit('No session matched; widen --min-len/--max-len.')
    sid = np.random.default_rng(opts.pick_seed).choice(ok.index)


s = df[df.session_id == sid].sort_values('ts').reset_index(drop=True)
y = s['skipped'].to_numpy()
L = len(s)
splits = valid_splits(y, L)
if not splits:
    raise SystemExit(f'{sid}: no valid split under the evaluation protocol.')
if opts.context is not None:
    if opts.context not in splits:
        raise SystemExit(f'context {opts.context} is not a valid split; '
                         f'options are {splits}')
    b = opts.context
else:
    chosen = pick(splits)                   
    b = chosen[len(chosen) // 2]             

b = opts.context if opts.context is not None else max(3, L // 3)
if not 3 <= b <= L - 5:
    raise SystemExit(f'Context {b} invalid for a session of {L} tracks.')

params = json.load(open(PARAMS))
model = SkipLSTM(input_size=len(FEATURES),
                 hidden_size=int(params['hidden_units']),
                 num_layers=int(params['num_layers']),
                 dropout=float(params['dropout_rate']))
model.load_state_dict(torch.load(CKPT, map_location='cpu'))
model.eval()


x = torch.tensor(s[FEATURES].values, dtype=torch.float32).unsqueeze(0)
status = torch.full((1, L), 2, dtype=torch.long)
status[0, :b] = torch.tensor(y[:b], dtype=torch.long)

with torch.no_grad():
    logits = model(x, torch.tensor([L]), status, torch.tensor([b]))[0].numpy()

p = 1 / (1 + np.exp(-logits[b:]))
yq = y[b:]
names = [f"{t[:34]:<34} {a[:20]}" for t, a in
         zip(s['Track Name'][b:], s['Artist Name'][b:])]
order = np.argsort(p, kind='stable')

print(f"\nSession {sid}")
print(f"{L} tracks | context {b} played | {L-b} in the queue | "
      f"model: SkipLSTM/PWTS ({params['hidden_units']:.0f} units)\n")

print("CONTEXT — already played")
for i in range(b):
    print(f"  {i+1:>2}. {s['Track Name'][i][:34]:<34} {s['Artist Name'][i][:20]:<20} "
          f"{'SKIPPED' if y[i] else 'played'}")

print("\nQUEUE — original order" + " " * 42 + "P(skip)   actual")
for i, (n, pr, a) in enumerate(zip(names, p, yq)):
    print(f"  {i+1:>2}. {n:<56} {pr:.3f}    {'SKIPPED' if a else 'played'}")

print("\nQUEUE — reordered by the model" + " " * 33 + "P(skip)   actual")
for rank, j in enumerate(order):
    flag = ' <-' if rank < 5 and yq[j] == 0 else ''
    print(f"  {rank+1:>2}. {names[j]:<56} {p[j]:.3f}    "
          f"{'SKIPPED' if yq[j] else 'played'}{flag}")

rand = np.arange(len(yq), dtype=float)
print(f"\nNDCG@5   model {ndcg_at_k(yq, p):.4f}   |   original order "
      f"{ndcg_at_k(yq, rand):.4f}")
print(f"AUC      model {fast_auc(yq, p):.4f}")
print(f"\nSkips in the first 5:  original {int(yq[:5].sum())}/5   "
      f"reordered {int(yq[order][:5].sum())}/5")
