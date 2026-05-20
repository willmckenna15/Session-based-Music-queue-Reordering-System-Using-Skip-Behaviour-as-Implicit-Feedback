import pandas as pd

audio_features = pd.read_parquet('../RAW Data/audio_features.parquet')
print(audio_features.columns.to_list())
print(max(audio_features["track_popularity"]),min(audio_features["track_popularity"]))