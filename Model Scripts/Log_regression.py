import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score
from matplotlib import pyplot
from sklearn.linear_model import LogisticRegression

training_file = '../RAW Data/training_data.parquet'
validation_file = '../RAW Data/validation_data.parquet'

extra_cols = ['user_id', 'spotify_track_uri']



features = ['tempo','mode', 'danceability', 'energy', 'loudness','speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence','hour','day_of_week']

target = 'skipped'

training_df = pd.read_parquet(training_file, columns=features + [target] + extra_cols)
training_df = training_df.dropna(subset=features)

validation_df = pd.read_parquet(validation_file, columns=features + [target] + extra_cols)
validation_df = validation_df.dropna(subset=features)

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
print('AUC: %.3f' % auc)

'''
fpr, tpr, thresholds = roc_curve(Y_val, probs)
pyplot.plot([0, 1], [0, 1], linestyle='--')
pyplot.plot(fpr, tpr, marker='.')
pyplot.show()
'''

