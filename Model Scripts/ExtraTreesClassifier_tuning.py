import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_curve, roc_auc_score, ndcg_score
import numpy as np
from matplotlib import pyplot
from sklearn.model_selection import ParameterGrid
import numpy as np
from Model_lib import ndcg_at_k, valid_splits, pick, fast_auc



training_file = '../RAW Data/training_data.parquet'
validation_file = '../RAW Data/validation_data.parquet'

extra_cols = ['user_id', 'spotify_track_uri', 'session_id', 'ts']



features = ['tempo', 'mode', 'danceability', 'energy', 'loudness', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness', 'valence',
            'historical_skip_rate',
            'historical_artist_skip_rate', 'shuffle', 'is_repeat_track', 'same_artist_as_prev']

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


# Hyper-Paramater Selection
param_grid = {
    'max_depth': [8,10,12,15],
    'min_samples_leaf': [10,20,30],
    'n_estimators': [100,200,300]
}

RESULTS_CSV = '../Models/extratrees_grid_search_results.csv'
combinations = list(ParameterGrid(param_grid))
print(f"\nRunning grid search over {len(combinations)} combinations...\n")

results = []
for i, params in enumerate(combinations, 1):
    print(f"[{i}/{len(combinations)}] {params}")

    model = ExtraTreesClassifier(random_state=42, n_jobs=-1, **params)
    model.fit(X, Y)
    probs = model.predict_proba(X_val)[:, 1]

    _, val_ndcg = score(probs)
    print(f"NDCG@5: {val_ndcg:.4f} "
          )

    results.append({**params, 'val_ndcg': val_ndcg, 'n_sessions': len(sessions)})
    pd.DataFrame(results).sort_values('val_ndcg', ascending=False).to_csv(
        RESULTS_CSV, index=False)

results_df = pd.DataFrame(results).sort_values('val_ndcg', ascending=False)
print("\n--- Grid Search Results ---")
print(results_df.to_string(index=False))

best_params = results_df.iloc[0].to_dict()
print(f"\nBest config: {best_params}")
print(f"Results saved to {RESULTS_CSV}")
