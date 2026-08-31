import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score
import numpy as np
from Model_lib import ndcg_at_k, valid_splits, pick, fast_auc
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--feature-set', type=str, default='all',
                    choices=['audio-only', 'behavioural-only', 'all'],
                    help='Which features to use')
args = parser.parse_args()

audio_features = ['tempo', 'mode', 'danceability', 'energy', 'loudness',
                  'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence']

behavioural_features = ['historical_skip_rate', 'historical_artist_skip_rate',
                        'shuffle', 'is_repeat_track', 'same_artist_as_prev']

all_features = audio_features + behavioural_features

if args.feature_set == 'audio-only':
    features = audio_features
elif args.feature_set == 'behavioural-only':
    features = behavioural_features
else:
    features = all_features

print(f"Using {args.feature_set}: {len(features)} features")

# Tuned on the full 15-feature set by Random_Forest_tuning.py, and the same config
# evaluate.py refits for the test-set results. Pinned rather than read from the grid
# CSV: that file currently holds only the max_depth=8 configs, and every leading
# config in the fuller search sat at the maximum depth tried, so the grid was
# bounded from above.
BEST_PARAMS = dict(max_depth=15, min_samples_leaf=20, n_estimators=200)

training_file = '../RAW Data/training_data.parquet'
validation_file = '../RAW Data/validation_data.parquet'

extra_cols = ['user_id', 'spotify_track_uri', 'session_id', 'ts']

target = 'skipped'
print("Reading Datasets...")
training_df = pd.read_parquet(training_file, columns=features + [target] + extra_cols)
training_df = training_df.dropna(subset=features)

validation_df = pd.read_parquet(validation_file, columns=features + [target] + extra_cols)
# reset_index so row positions line up with predict_proba output
validation_df = validation_df.dropna(subset=features).reset_index(drop=True)

X = training_df[features]
Y = training_df[target]

X_val = validation_df[features]
Y_val = validation_df[target]

print("Indexing validation sessions...")
sessions = []
for _, g in validation_df.sort_values('ts').groupby('session_id', sort=False):
    y = g[target].to_numpy()
    splits = valid_splits(y, len(y))
    if splits:
        sessions.append((g.index.to_numpy(), y, pick(splits)))
print(f"{len(sessions)} scorable sessions of {validation_df['session_id'].nunique()}")


def score(probs):
    """Mean per-session AUC and NDCG@5, averaged within session before across -
    the same protocol as Model_lib.evaluate_sequential."""
    aucs, ndcgs = [], []
    for pos, y, splits in sessions:
        p = probs[pos]
        aucs.append(np.mean([fast_auc(y[c:], p[c:]) for c in splits]))
        ndcgs.append(np.mean([ndcg_at_k(y[c:], p[c:]) for c in splits]))
    return float(np.mean(aucs)), float(np.mean(ndcgs))


print(f"Training Model... {BEST_PARAMS}")
model = ExtraTreesClassifier(random_state=42, n_jobs=-1, **BEST_PARAMS)
model.fit(X, Y)
print("Model trained")

print("Validating model...")
probs = model.predict_proba(X_val)[:, 1]

pooled_auc = roc_auc_score(Y_val, probs)
session_auc, session_ndcg = score(probs)

print(f'Pooled AUC:         {pooled_auc:.4f}')
print(f'Per-session AUC:    {session_auc:.4f}')
print(f'Per-session NDCG@5: {session_ndcg:.4f} across {len(sessions)} sessions')

print("\n--- Feature Importances ---")
for name, imp in sorted(zip(features, model.feature_importances_),
                        key=lambda kv: -kv[1]):
    print(f"{name:<30} {imp:.4f}")

# Append this run to the ablation table. One row per feature set, so running the
# script three times builds the comparison. Same columns as Log_regression.py, so
# the two models' rows sit in one table.
OUT = '../Models/feature_ablation_study.csv'
row = pd.DataFrame([{
    'model': 'ExtraTrees',
    'feature_set': args.feature_set,
    'n_features': len(features),
    'pooled_auc': round(pooled_auc, 4),
    'session_auc': round(session_auc, 4),
    'session_ndcg5': round(session_ndcg, 4),
    'n_sessions': len(sessions),
    'split': 'validation',
    'timestamp': pd.Timestamp.now(),
}])
if os.path.exists(OUT):
    row = pd.concat([pd.read_csv(OUT), row], ignore_index=True)
row.to_csv(OUT, index=False)
print(f'\nAppended to {OUT}')
