"""Final evaluation of all eight models on the held-out test set.

Every model is scored under one protocol: for each session, enumerate the valid
context/query split points, take up to five evenly spaced, score the query window at
each, average within the session, then average across sessions. That is the same
procedure used for tuning and training, so the numbers are comparable throughout.

    python evaluate.py                 # test set (default)
    python evaluate.py --split val     # sanity check against training-run numbers
    python evaluate.py --no-baselines  # skip refitting ExtraTrees/LogReg

Per-session scores are written to ../Models/eval_per_session.npz so paired
significance tests can be run afterwards without re-evaluating.
"""

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from Model_lib import (SessionDataset, collate_fn, SASRec, SkipLSTM,
                       valid_splits, pick, fast_auc, ndcg_at_k,
                       MIN_CONTEXT, MIN_WINDOW_LENGTH, POINTS_PER_SESSION)

features = ['tempo', 'mode', 'danceability', 'energy', 'loudness', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness', 'valence',
            'historical_skip_rate',
            'historical_artist_skip_rate', 'shuffle', 'is_repeat_track', 'same_artist_as_prev']
target = 'skipped'

BATCH_SIZE = 64
CHUNK = 32
SEED = 0

cli = argparse.ArgumentParser()
cli.add_argument('--split', default='testing', choices=['testing', 'validation'])
cli.add_argument('--no-baselines', action='store_true')
cli.add_argument('--out', default='../Models/final_results.csv')
opts = cli.parse_args()

DATA = f'../RAW Data/{opts.split}_data.parquet'

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


##Metrics

def mrr(y, p):
    """Reciprocal rank of the first track the listener does NOT skip, under the
    model's ordering (ascending by predicted skip probability).

    Note this saturates on this dataset: ~62% of tracks are not skipped, so the
    first kept track is usually at rank 1 or 2 and MRR sits near 0.9 for every
    model, including a random one. Reported because it is conventional, but
    prec@k discriminates far better here.
    """
    order = np.argsort(p, kind='stable')
    kept = np.flatnonzero(1.0 - y[order])
    return 1.0 / (kept[0] + 1) if kept.size else 0.0


def precision_at_k(y, p, k):
    """Fraction of the top-k reordered queue that the listener does not skip.
    The most directly interpretable metric here: 'of the next k songs this system
    would play, how many are kept?'"""
    order = np.argsort(p, kind='stable')[:k]
    return float((1.0 - y[order]).mean()) if order.size else np.nan


def window_metrics(y, p):
    return {
        'AUC':      fast_auc(y, p),
        'NDCG@5':   ndcg_at_k(y, p, 5),
        'NDCG@10':  ndcg_at_k(y, p, 10),
        'MRR':      mrr(y, p),
        'P@5':      precision_at_k(y, p, 5),
    }


METRICS = ['AUC', 'NDCG@5', 'NDCG@10', 'MRR', 'P@5']


def aggregate(per_session):
    """per_session: {session_id: [dict_of_metrics, ...]} -> mean per metric, plus
    per-session arrays ordered by session_id.

    Keying and sorting by session_id (not position) is what makes the saved arrays
    comparable across models: the neural path iterates a length-sorted loader while
    the sklearn path groups by session_id, so positional alignment would silently
    pair different sessions in a paired test.
    """
    keys = sorted(per_session)
    arrays = {m: np.array([np.mean([w[m] for w in per_session[k]]) for k in keys])
              for m in METRICS}
    return {m: float(a.mean()) for m, a in arrays.items()}, arrays, keys


##Data

print(f"Loading {opts.split} set...")
dataset = SessionDataset(DATA, features, target)
order = np.argsort([len(y) for y in dataset.labels]).tolist()
loader = DataLoader(Subset(dataset, order), batch_size=BATCH_SIZE,
                    shuffle=False, collate_fn=collate_fn)
print(f"{len(dataset)} sessions")


##Neural evaluation

def eval_neural(model):
    model.eval()
    per_session = {}
    with torch.no_grad():
        for batch_i, batch in enumerate(loader):
            sessions, labels, lengths = batch[0], batch[1], batch[-1]
            # map position in this batch back to the dataset index, then to the
            # session_id, so scores can be paired across models
            base = batch_i * BATCH_SIZE
            rows = []
            for i in range(sessions.shape[0]):
                L = int(lengths[i])
                y = labels[i, :L].numpy()
                for c in pick(valid_splits(y, L, MIN_CONTEXT, 1, MIN_WINDOW_LENGTH),
                              POINTS_PER_SESSION):
                    rows.append((i, c))
            if not rows:
                continue

            idx = torch.tensor([i for i, _ in rows])
            bounds = torch.tensor([c for _, c in rows], dtype=torch.long)
            exp_labels = labels[idx]
            status = torch.full(exp_labels.shape, 2, dtype=torch.long)
            for r, (_, c) in enumerate(rows):
                status[r, :c] = exp_labels[r, :c].long()

            preds = []
            for s in range(0, len(rows), CHUNK):
                sl = slice(s, s + CHUNK)
                p = model(sessions[idx[sl]].to(device), lengths[idx[sl]],
                          status[sl].to(device), bounds[sl])
                preds.append(p.cpu())
            preds = torch.cat(preds, dim=0)

            for r, (i, c) in enumerate(rows):
                L = int(lengths[i])
                sid = dataset.session_ids[order[base + i]]
                per_session.setdefault(sid, []).append(
                    window_metrics(labels[i, c:L].numpy(), preds[r, c:L].numpy()))
    return aggregate(per_session)


##Flat (sklearn) evaluation - predictions do not depend on the split point

def eval_flat(probs_by_session):
    """probs_by_session: {session_id: (y, p)} - predictions do not depend on the
    split point, so one forward pass per row is enough."""
    per_session = {}
    for sid, (y, p) in probs_by_session.items():
        for c in pick(valid_splits(y, len(y), MIN_CONTEXT, 1, MIN_WINDOW_LENGTH),
                      POINTS_PER_SESSION):
            per_session.setdefault(sid, []).append(window_metrics(y[c:], p[c:]))
    return aggregate(per_session)


##Model loading

def load_neural(kind, loss):
    """Returns a list of (label, model) - one per saved seed, or the single best."""
    prefix = 'sasrec' if kind == 'SASRec' else 'lstm'
    pj = f'../Models/{prefix}_{loss}_best_params.json'
    if not os.path.exists(pj):
        return []
    bp = json.load(open(pj))

    ckpts = sorted(glob.glob(f'../Models/{prefix}_{loss}_seed*.pt'),
                   key=lambda p: int(re.search(r'seed(\d+)', p).group(1)))
    if not ckpts:
        single = f'../Models/{prefix}_{loss}_best.pt'
        ckpts = [single] if os.path.exists(single) else []
    if not ckpts:
        return []

    out = []
    for path in ckpts:
        if kind == 'SASRec':
            a = argparse.Namespace(device=device, hidden_units=int(bp['hidden_units']),
                                   maxlen=200, dropout_rate=float(bp['dropout_rate']),
                                   num_blocks=int(bp['num_blocks']),
                                   num_heads=int(bp['num_heads']), norm_first=True)
            m = SASRec(feature_no=len(features), args=a)
        else:
            m = SkipLSTM(input_size=len(features),
                         hidden_size=int(bp['hidden_units']),
                         num_layers=int(bp['num_layers']),
                         dropout=float(bp['dropout_rate']))
        m.load_state_dict(torch.load(path, map_location=device))
        out.append((os.path.basename(path), m.to(device)))
    return out


##Baselines - neither script persists its model, so refit here

def baseline_predictions():
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.linear_model import LogisticRegression

    cols = features + [target, 'session_id', 'ts']
    tr = pd.read_parquet('../RAW Data/training_data.parquet', columns=cols).dropna(subset=features)
    te = pd.read_parquet(DATA, columns=cols).dropna(subset=features).reset_index(drop=True)

    grouped = {sid: (g[target].to_numpy(), g.index.to_numpy())
               for sid, g in te.sort_values('ts').groupby('session_id', sort=False)}

    fits = {}

    grid = '../Models/extratrees_grid_search_results.csv'
    if os.path.exists(grid):
        best = pd.read_csv(grid).sort_values('val_ndcg', ascending=False).iloc[0]
        print(f"ExtraTrees: refitting depth={int(best.max_depth)} "
              f"leaf={int(best.min_samples_leaf)} trees={int(best.n_estimators)}")
        fits['ExtraTrees'] = ExtraTreesClassifier(
            random_state=SEED, n_jobs=-1, max_depth=int(best.max_depth),
            min_samples_leaf=int(best.min_samples_leaf),
            n_estimators=int(best.n_estimators))
    fits['LogReg'] = LogisticRegression(solver='lbfgs', max_iter=1000, C=1.0)

    out = {}
    for name, clf in fits.items():
        clf.fit(tr[features], tr[target])
        p = clf.predict_proba(te[features])[:, 1]
        out[name] = {sid: (y, p[pos]) for sid, (y, pos) in grouped.items()}
    # random ordering, on identical windows - the floor every metric must beat
    rng = np.random.default_rng(SEED)
    rp = rng.random(len(te))
    out['Random'] = {sid: (y, rp[pos]) for sid, (y, pos) in grouped.items()}
    return out


##Run

rows, per_session_store = [], {}
session_keys = None


def record(name, means, arrays, keys, n_seeds=1, spread=None):
    global session_keys
    if session_keys is None:
        session_keys = keys
    elif keys != session_keys:
        print(f"  WARNING: {name} scored a different session set "
              f"({len(keys)} vs {len(session_keys)}) - paired tests against it "
              f"would be invalid")
    rows.append({'model': name, 'n_seeds': n_seeds, 'sessions': len(keys),
                 **{m: round(means[m], 4) for m in METRICS},
                 **({f'{m}_sd': round(spread[m], 4) for m in METRICS} if spread else {})})
    for m in METRICS:
        per_session_store[f'{name}|{m}'] = arrays[m]


for kind in ('SASRec', 'SkipLSTM'):
    for loss in ('bce', 'pwts', 'drrl'):
        models = load_neural(kind, loss)
        if not models:
            print(f"skip {kind}/{loss}: no checkpoint")
            continue
        print(f"{kind}/{loss}: {len(models)} checkpoint(s)")
        seed_means, last = [], None
        for label, m in models:
            means, arrays, keys = eval_neural(m)
            seed_means.append(means)
            last = (arrays, keys)
        avg = {k: float(np.mean([s[k] for s in seed_means])) for k in METRICS}
        sd = {k: float(np.std([s[k] for s in seed_means])) for k in METRICS}
        record(f'{kind}/{loss}', avg, last[0], last[1], len(models), sd)

if not opts.no_baselines:
    for name, preds in baseline_predictions().items():
        print(f"{name}: scoring")
        means, arrays, keys = eval_flat(preds)
        record(name, means, arrays, keys)


##Output

df = pd.DataFrame(rows)
cols = ['model', 'n_seeds', 'sessions'] + \
       [c for m in METRICS for c in (m, f'{m}_sd') if c in df.columns]
df = df[[c for c in cols if c in df.columns]]
df.to_csv(opts.out, index=False)

np.savez_compressed('../Models/eval_per_session.npz',
                    session_ids=np.array(session_keys, dtype=object), **per_session_store)

print(f"\n{'='*78}\nTEST RESULTS ({opts.split} set, {len(dataset)} sessions)\n{'='*78}")
print(df.to_string(index=False))
print(f"\nWritten to {opts.out}")
print("Per-session scores -> ../Models/eval_per_session.npz (for paired tests)")
