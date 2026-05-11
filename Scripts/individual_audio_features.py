import pandas as pd
from sklearn.preprocessing import StandardScaler

def main(user_id):
    while True:
        try:
            Streaming_history = pd.read_csv(f"../Volunteer Data/RAW/Streaming_history_{user_id}.csv").copy()
        except FileNotFoundError:
            print(f"No file found for UUID: {user_id}")
            continue
        break
    audio_features = pd.read_parquet('../Raw Data/audio_features.parquet')

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

    Streaming_history.to_csv(f'../Volunteer Data/RAW/Audio_Streaming_History_{user_id}.csv', index=False)
    print(f"Audio features for user {user_id} have been added")
    print(" ")

if __name__ == "__main__":
    main()