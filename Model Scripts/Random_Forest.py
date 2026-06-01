import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_curve, roc_auc_score, ndcg_score
import numpy as np
from matplotlib import pyplot
from sklearn.model_selection import GridSearchCV

training_file = '../RAW Data/training_data.parquet'
validation_file = '../RAW Data/validation_data.parquet'

extra_cols = ['user_id', 'spotify_track_uri']



features = ['tempo','mode', 'danceability', 'energy', 'loudness','speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence','hour','day_of_week']

target = 'skipped'
print("Reading Datasets...")
training_df = pd.read_parquet(training_file, columns=features + [target] + extra_cols)
training_df = training_df.dropna(subset=features)

validation_df = pd.read_parquet(validation_file, columns=features + [target] + extra_cols)
validation_df = validation_df.dropna(subset=features)

X = training_df[features]
Y = training_df[target]

print("Chronologising historical skip rate...")

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

# Hyper-Paramater Selection
'''
param_grid = {
    'max_depth': [8,10,12,15],
    'min_samples_leaf': [10,20,30],
    'n_estimators': [100,200,300]
}
model = ExtraTreesClassifier(random_state=42)
grid_search = GridSearchCV(model, param_grid, scoring='roc_auc', cv=3, n_jobs=-1, verbose=1)
grid_search.fit(X, Y)

print(f"Best params: {grid_search.best_params_}")
print(f"Best CV AUC: {grid_search.best_score_:.3f}")

best_model = grid_search.best_estimator_
probs = best_model.predict_proba(X_val)[:, 1]
auc = roc_auc_score(Y_val, probs)
print(f"Validation AUC: {auc:.3f}")
'''
print("Training Model...")
model = ExtraTreesClassifier(random_state = 42, min_samples_leaf=10, max_depth = 15, n_estimators=300)

model.fit(X, Y)
print("Model trained")

print("Validating model...")
probs = model.predict_proba(X_val)
probs = probs[:, 1]

print("Calculating importances...")
importances = model.feature_importances_
print("--- Feature Importances ---")
for i in range(len(features)):
    print(f"{features[i]}: {importances[i]}")

print("Calculating AUC-ROC score...")
auc = roc_auc_score(Y_val, probs)
print('AUC: %.3f' % auc)


'''
fpr, tpr, thresholds = roc_curve(Y_val, probs)
pyplot.plot([0, 1], [0, 1], linestyle='--')
pyplot.plot(fpr, tpr, marker='.')
pyplot.show()
'''