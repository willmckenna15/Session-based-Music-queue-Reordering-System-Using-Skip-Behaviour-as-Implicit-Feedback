import pandas as pd
from sklearn.preprocessing import StandardScaler

def main():
    print("Reading Files..")
    audio_features = pd.read_parquet('../RAW Data/audio_features.parquet')
    csv_path = '../RAW Data/Combined_Streaming_History.csv'
    try:
        Streaming_history = pd.read_csv(csv_path)
        print("Files Read")
    except FileNotFoundError:
        print("No existing Combined_Streaming_History.csv found, creating new one...")
        Streaming_history = pd.DataFrame()

    cols = ['track_id', 'tempo', 'key', 'mode', 'danceability', 'energy', 'loudness', 
            'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence']
    existing_cols = ['tempo', 'key', 'mode', 'danceability', 'energy', 'loudness', 
                    'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence']

    Streaming_history = Streaming_history.drop(columns=[c for c in existing_cols if c in Streaming_history.columns])
    Streaming_history['spotify_track_uri'] = Streaming_history['spotify_track_uri'].str.replace('spotify:track:', '')

    print("Dropping Duplicates...")
    Streaming_history = Streaming_history.drop_duplicates()
    print("Duplicates Dropped")

    print("Merging Datasets...")
    Streaming_history = Streaming_history.merge(
        audio_features[cols],
        left_on='spotify_track_uri',
        right_on='track_id',
        how='inner'
    ).drop(columns='track_id')
    print("Datasets Merged")

    print("Normalising Dataset")
    scaler = StandardScaler()
    cols_to_normalise = ['danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence']
    Streaming_history[cols_to_normalise] = scaler.fit_transform(Streaming_history[cols_to_normalise])
    print("Data Normalised")

    print("Writing to csv...")
    Streaming_history.to_csv(csv_path, index=False)
    print("Combined Streaming History saved")

if __name__ == "__main__":
    main()