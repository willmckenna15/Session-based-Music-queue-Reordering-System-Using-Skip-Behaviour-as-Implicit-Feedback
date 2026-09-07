import argparse, glob, json
import numpy as np, pandas as pd, torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'Model Scripts'))

from Model_lib import SkipLSTM, ndcg_at_k, valid_splits, MIN_CONTEXT

FEATURES = ['tempo','mode','danceability','energy','loudness','speechiness',
            'acousticness','instrumentalness','liveness','valence',
            'historical_skip_rate','historical_artist_skip_rate','shuffle',
            'is_repeat_track','same_artist_as_prev']
POINTS = 5

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
                     columns=FEATURES + ['skipped','session_id','ts']).dropna(subset=FEATURES)
df = df.sort_values(['session_id','ts'])

rows, kept, seen = [], 0, 0
for sid, g in df.groupby('session_id', sort=False):
    seen += 1
    if opts.limit and kept >= opts.limit: break
    y = g['skipped'].to_numpy(); L = len(y)
    minq = max(5, int(np.ceil(0.10 * L)))
    v = valid_splits(y, L, MIN_CONTEXT, 1, minq)
    if len(v) < POINTS: continue
    splits = evenly(v)
    x = torch.tensor(g[FEATURES].values, dtype=torch.float32).unsqueeze(0).repeat(len(splits),1,1)
    status = torch.full((len(splits), L), 2, dtype=torch.long)
    for i, b in enumerate(splits):
        status[i, :b] = torch.tensor(y[:b], dtype=torch.long)
    bnd = torch.tensor(splits)
    with torch.no_grad():
        preds = [m(x, torch.tensor([L]*len(splits)), status, bnd).numpy() for m in models]
    for i, b in enumerate(splits):
        yq = y[b:]
        nd = float(np.mean([ndcg_at_k(yq, 1/(1+np.exp(-p[i, b:]))) for p in preds]))
        rnd = expected_random_ndcg(yq)
        rows.append({'session_id': sid, 'split_index': i+1, 'context_len': b,
                     'query_len': L-b, 'session_len': L,
                     'query_skip_rate': round(float(yq.mean()),4),
                     'ndcg_model': round(nd,4), 'ndcg_random': round(rnd,4)})
    kept += 1
    if kept % 1000 == 0: print(f"  {kept} sessions", flush=True)

r = pd.DataFrame(rows); r.to_csv(opts.out, index=False)
print(f"\n{kept:,} sessions of {seen:,} scanned  ->  {len(r):,} rows  ->  {opts.out}\n")

agg = r.groupby('split_index').agg(
    n=('lift','size'), context=('context_len','median'), query=('query_len','median'),
    ndcg=('ndcg_model','mean'), random=('ndcg_random','mean'))
print(agg.round(4).to_string())

first = r[r.split_index==1].set_index('session_id')
last  = r[r.split_index==POINTS].set_index('session_id')
pair  = pd.concat([first, last], axis=1, keys=['first','last']).dropna()
d = pair['last'] - pair['first']
rng = np.random.default_rng(0)
bs = d.to_numpy()[rng.integers(0, len(d), (10000, len(d)))].mean(axis=1)
p = 2*min((bs<=0).mean(), (bs>=0).mean())
# Within-session spread across split points: how much the choice of split moves a
# session's score, which is what motivates averaging several points rather than one.
w = r.groupby('session_id')['ndcg_model']
sd, rng_ = w.std(ddof=1), w.max() - w.min()
spread = pd.DataFrame([
    ('Within-session standard deviation', round(float(sd.mean()), 4)),
    ('Within-session range (max - min)',   round(float(rng_.mean()), 4)),
], columns=['Metric', 'Mean'])
spread.to_csv(opts.spread, index=False)

table = agg.reset_index().rename(columns={
    'split_index': 'Split point', 'n': 'Sessions', 'context': 'Median context',
    'query': 'Median query', 'ndcg': 'NDCG@5 (model)', 'random': 'NDCG@5 (random)'})
table.round(4).to_csv(opts.table, index=False)

print(f"\ntable  -> {opts.table}")
print(table.round(4).to_string(index=False))
print(f"\nspread -> {opts.spread}")
print(spread.to_string(index=False))

fig, ax = plt.subplots(figsize=(6.6, 4.3))
x = agg.index
ax.fill_between(x, agg['random'], agg['ndcg'], color='#1B6E68', alpha=.10)
ax.plot(x, agg['ndcg'], 'o-', color='#1B6E68', lw=2, label='model')
ax.plot(x, agg['random'], 's--', color='#8F5512', lw=1.8, label='random floor')
ax.set_xticks(x)
ax.set_xticklabels([str(i) for i in x], fontsize=9.5)
ax.set_xlabel('split point  (1 = earliest valid split, 5 = latest)',
              fontsize=10, labelpad=6)
ax.set_ylabel('NDCG@5', fontsize=10)
ax.legend(frameon=False, fontsize=9.5, loc='upper left')
ax.grid(alpha=.25)
for sp in ('top', 'right'):
    ax.spines[sp].set_visible(False)
plt.tight_layout()
plt.savefig(opts.figure, dpi=220, facecolor='white')
print(f"figure -> {opts.figure}")

print(f"\npaired, split 5 - split 1 over {len(d):,} sessions")
print(f"  mean {d.mean():+.4f}   95% CI [{np.percentile(bs,2.5):+.4f}, {np.percentile(bs,97.5):+.4f}]"
      f"   p {max(p,1e-4):.4f}")
print(f"  higher at split 5 in {(d>0).mean():.1%} of sessions")



