"""Does the model rank better once it has observed more of the session?
"""

import argparse, glob, json, os, sys
import numpy as np, pandas as pd, torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'Model Scripts'))
from Model_lib import (SkipLSTM, ndcg_at_k, fast_auc, valid_splits, MIN_CONTEXT)

FEATURES = ['tempo', 'mode', 'danceability', 'energy', 'loudness', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness', 'valence',
            'historical_skip_rate', 'historical_artist_skip_rate', 'shuffle',
            'is_repeat_track', 'same_artist_as_prev']
POINTS = 5
RESAMPLES = 10000

cli = argparse.ArgumentParser()
cli.add_argument('--limit', type=int, default=None)
cli.add_argument('--out', default='../Models/context_analysis.csv')
cli.add_argument('--figure', default='../Report/context_length.png')
cli.add_argument('--table', default='../Report/context_length_table.csv')
cli.add_argument('--spread', default='../Report/split_point_spread.csv')
opts = cli.parse_args()

params = json.load(open('../Models/lstm_pwts_best_params.json'))
models = []
for ck in sorted(glob.glob('../Models/lstm_pwts_seed?.pt')):
    m = SkipLSTM(len(FEATURES), int(params['hidden_units']),
                 int(params['num_layers']), float(params['dropout_rate']))
    m.load_state_dict(torch.load(ck, map_location='cpu')); m.eval(); models.append(m)
print(f"{len(models)} seed checkpoints")


def expected_random_ndcg(y, k=5):
    n = len(y); n_rel = int((1 - y).sum())
    if n_rel == 0: return np.nan
    disc = 1 / np.log2(np.arange(2, n + 2))
    idcg = disc[:min(k, n_rel)].sum()
    return (n_rel / n) * disc[:min(k, n)].sum() / idcg if idcg > 0 else np.nan


def evenly(cands, points=POINTS):
    return [cands[i] for i in np.linspace(0, len(cands) - 1, points).astype(int)]


df = pd.read_parquet('../RAW Data/testing_data.parquet',
                     columns=FEATURES + ['skipped', 'session_id', 'ts']).dropna(subset=FEATURES)
df = df.sort_values(['session_id', 'ts'])

rows, kept, seen = [], 0, 0
for sid, g in df.groupby('session_id', sort=False):
    seen += 1
    if opts.limit and kept >= opts.limit: break
    y = g['skipped'].to_numpy(); L = len(y)
    minq = max(5, int(np.ceil(0.10 * L)))
    v = valid_splits(y, L, MIN_CONTEXT, 1, minq)
    if len(v) < POINTS: continue
    splits = evenly(v)
    x = torch.tensor(g[FEATURES].values, dtype=torch.float32).unsqueeze(0).repeat(len(splits), 1, 1)
    status = torch.full((len(splits), L), 2, dtype=torch.long)
    for i, b in enumerate(splits):
        status[i, :b] = torch.tensor(y[:b], dtype=torch.long)
    with torch.no_grad():
        preds = [m(x, torch.tensor([L] * len(splits)), status,
                   torch.tensor(splits)).numpy() for m in models]
    for i, b in enumerate(splits):
        yq = y[b:]
        ps = [1 / (1 + np.exp(-p[i, b:])) for p in preds]
        rows.append({'session_id': sid, 'split_index': i + 1, 'context_len': b,
                     'query_len': L - b, 'session_len': L,
                     'query_skip_rate': round(float(yq.mean()), 4),
                     'auc': round(float(np.mean([fast_auc(yq, p) for p in ps])), 4),
                     'ndcg_model': round(float(np.mean([ndcg_at_k(yq, p) for p in ps])), 4),
                     'ndcg_random': round(float(expected_random_ndcg(yq)), 4)})
    kept += 1
    if kept % 2000 == 0: print(f"  {kept} sessions", flush=True)

r = pd.DataFrame(rows)
r['ndcg_lift'] = (r.ndcg_model - r.ndcg_random).round(4)
r.to_csv(opts.out, index=False)
print(f"\n{kept:,} sessions of {seen:,} scanned  ->  {len(r):,} rows  ->  {opts.out}\n")

wide = r.pivot(index='session_id', columns='split_index', values='auc').dropna()
rng = np.random.default_rng(0)
agg = r.groupby('split_index').agg(
    sessions=('session_id', 'size'),
    median_context=('context_len', 'median'), median_query=('query_len', 'median'),
    auc=('auc', 'mean'), ndcg=('ndcg_model', 'mean'),
    ndcg_random=('ndcg_random', 'mean'), ndcg_lift=('ndcg_lift', 'mean'))

diffs, los, his, ps = [], [], [], []
for s in agg.index:
    d = (wide[s] - wide[1]).to_numpy()
    bs = d[rng.integers(0, len(d), (RESAMPLES, len(d)))].mean(axis=1)
    obs = d.mean()
    diffs.append(obs)
    lo, hi = np.percentile(bs, [2.5, 97.5]); los.append(lo); his.append(hi)
    ps.append(np.nan if s == 1 else
              (1 + (np.abs(bs - obs) >= abs(obs)).sum()) / (RESAMPLES + 1))
agg['auc_vs_split1'] = diffs
agg['ci_low'], agg['ci_high'], agg['p'] = los, his, ps

TABLE = ['median_context', 'median_query', 'auc', 'ndcg', 'ndcg_random', 'ndcg_lift']
agg[TABLE].round(4).to_csv(opts.table)
print(agg[TABLE].round(4).to_string())
print('\npaired against split 1 (not written to the table):')
print(agg[['auc_vs_split1', 'ci_low', 'ci_high', 'p']].round(4).to_string())

w = r.groupby('session_id')['ndcg_model']
pd.DataFrame([('Within-session standard deviation', round(float(w.std(ddof=1).mean()), 4)),
              ('Within-session range (max - min)', round(float((w.max() - w.min()).mean()), 4))],
             columns=['Metric', 'Mean']).to_csv(opts.spread, index=False)

fig, ax = plt.subplots(figsize=(6.4, 4.2))
ax.plot(agg.index, agg.auc, marker='o', color='#1B6E68')
ax.axhline(0.5, ls='--', lw=1, color='#999')
ax.set_xticks(list(agg.index))
ax.set_xlabel('split point')
ax.set_ylabel('session AUC')
ax.spines[['top', 'right']].set_visible(False)
fig.tight_layout(); fig.savefig(opts.figure, dpi=150)
print(f"\n{opts.table}\n{opts.spread}\n{opts.figure}")
