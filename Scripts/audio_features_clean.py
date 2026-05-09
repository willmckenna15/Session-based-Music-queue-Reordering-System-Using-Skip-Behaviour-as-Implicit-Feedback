import pandas as pd

audio_features = pd.read_parquet('../Raw Data/audio_features.parquet')
Streaming_history = pd.read_csv('../Raw Data/Combined_Streaming_History.csv')

cols = ['track_id', 'tempo', 'key', 'mode', 'danceability', 'energy', 'loudness', 
        'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence']
existing_cols = ['tempo', 'key', 'mode', 'danceability', 'energy', 'loudness', 
                 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence']
Streaming_history = Streaming_history.drop(columns=[c for c in existing_cols if c in Streaming_history.columns])
Streaming_history['spotify_track_uri'] = Streaming_history['spotify_track_uri'].str.replace('spotify:track:', '')

Streaming_history=Streaming_history.drop_duplicates()

Streaming_history = Streaming_history.merge(
    audio_features[cols],
    left_on='spotify_track_uri',
    right_on='track_id',
    how='inner'
).drop(columns='track_id')

Streaming_history.to_csv('../Raw Data/Combined_Streaming_History.csv', index=False)