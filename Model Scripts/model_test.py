import pandas as pd
df = pd.read_parquet("../RAW Data/audio_features/spotify_audio_features_0.parquet")
print(df.columns.tolist())