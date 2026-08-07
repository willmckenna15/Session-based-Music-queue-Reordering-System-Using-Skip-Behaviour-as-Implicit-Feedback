import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
from matplotlib import pyplot
from sklearn.linear_model import LogisticRegression
# same metric definitions the sequence models are scored on, so the numbers compare
from Model_lib import ndcg_at_k, valid_splits, pick, fast_auc

training_file = '../RAW Data/training_data.parquet'
validation_file = '../RAW Data/validation_data.parquet'

extra_cols = ['user_id', 'spotify_track_uri', 'session_id', 'ts']



# actively_selected removed: it is not knowable for an unplayed queue track, so it
# cannot be deployed. Same 17 features as SASRec/SkipLSTM/ExtraTrees.
features = ['tempo', 'mode', 'danceability', 'energy', 'loudness', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness', 'valence',
            'historical_skip_rate',
            'historical_artist_skip_rate', 'shuffle', 'is_repeat_track', 'same_artist_as_prev']
target = 'skipped'

training_df = pd.read_parquet(training_file, columns=features + [target] + extra_cols)
training_df = training_df.dropna(subset=features)

validation_df = pd.read_parquet(validation_file, columns=features + [target] + extra_cols)
# reset_index so row positions line up with predict_proba output
validation_df = validation_df.dropna(subset=features).reset_index(drop=True)

X = training_df[features]
Y = training_df[target]

'''
train_skip_rate = (
    training_df.groupby(["user_id", "spotify_track_uri"])["skipped"]
    .mean()
    .rename("historical_skip_rate")
    .reset_index()
)

validation_df = validation_df.drop(columns=["historical_skip_rate"])
validation_df = validation_df.merge(train_skip_rate, on=["user_id", "spotify_track_uri"], how="left")
validation_df["historical_skip_rate"] = validation_df["historical_skip_rate"].fillna(0)
'''
X_val = validation_df[features]
Y_val = validation_df[target]

model = LogisticRegression(solver = 'lbfgs', max_iter = 1000, C=1.0)

model.fit(X,Y)

probs = model.predict_proba(X_val)
probs = probs[:, 1]

auc = roc_auc_score(Y_val, probs)
print('Pooled AUC: %.4f' % auc)

# Per-session scoring - the protocol Model_lib.evaluate_sequential uses, so these
# numbers sit in the same table as SASRec/SkipLSTM. Pooled AUC is a different
# quantity and the two are not interchangeable.
session_aucs, session_ndcgs = [], []
for _, g in validation_df.sort_values('ts').groupby('session_id', sort=False):
    y, p = g[target].to_numpy(), probs[g.index.to_numpy()]
    splits = valid_splits(y, len(y))
    if not splits:
        continue
    chosen = pick(splits)
    session_aucs.append(np.mean([fast_auc(y[c:], p[c:]) for c in chosen]))
    session_ndcgs.append(np.mean([ndcg_at_k(y[c:], p[c:]) for c in chosen]))

print(f'Per-session AUC:    {np.mean(session_aucs):.4f}')
print(f'Per-session NDCG@5: {np.mean(session_ndcgs):.4f} across {len(session_aucs)} sessions')

'''
fpr, tpr, thresholds = roc_curve(Y_val, probs)
pyplot.plot([0, 1], [0, 1], linestyle='--')
pyplot.plot(fpr, tpr, marker='.')
pyplot.show()
'''

