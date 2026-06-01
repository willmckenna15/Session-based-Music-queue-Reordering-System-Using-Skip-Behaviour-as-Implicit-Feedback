import pandas as pd
from sklearn.preprocessing import StandardScaler
import glob

def main():
    print("Reading parquet...")
    parquet_path = '../RAW Data/Filtered_sessions.parquet'
    Streaming_history = pd.read_parquet(parquet_path)
    print("Parquet Read")

    existing_cols = ['tempo', 'key', 'mode', 'danceability', 'energy', 'loudness',
                    'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence']

    Streaming_history = Streaming_history.drop(columns=[c for c in existing_cols if c in Streaming_history.columns])
    Streaming_history['spotify_track_uri'] = Streaming_history['spotify_track_uri'].str.replace('spotify:track:', '')
    Streaming_history = Streaming_history.drop_duplicates()
    Streaming_history = Streaming_history.rename(columns={
        "master_metadata_track_name": "Track Name",
        "master_metadata_album_artist_name": "Artist Name"
    })

    track_ids = set(Streaming_history['spotify_track_uri'].dropna().unique())
    print(f"Unique tracks to match: {len(track_ids)}")

    audio_cols = ['id', 'tempo', 'mode', 'danceability', 'energy', 'loudness',
                'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence']

    print("Reading and filtering parquets...")
    parquet_files = glob.glob("../RAW Data/audio_features/*.parquet")
    chunks = []

    for i, f in enumerate(parquet_files):
        print(f"Processing file {i+1}/{len(parquet_files)}...")
        df = pd.read_parquet(f, columns=audio_cols)
        matched = df[df['id'].isin(track_ids)]
        chunks.append(matched)
        print(f"  Matched {len(matched)} tracks")
        del df

    audio_features = pd.concat(chunks, ignore_index=True).drop_duplicates(subset='id')
    print(f"Total matched audio features: {len(audio_features)}")

    print("Merging Datasets...")
    Streaming_history_merged = Streaming_history.merge(
        audio_features,
        left_on='spotify_track_uri',
        right_on='id',
        how='inner'
    ).drop(columns='id')

    song_count_a = len(Streaming_history_merged[["Track Name", "Artist Name"]].drop_duplicates())
    song_count_f = len(Streaming_history[["Track Name", "Artist Name"]].drop_duplicates())
    lost_songs = song_count_f - song_count_a

    print(f"Songs lost: {lost_songs} out of {song_count_f} ({round((lost_songs / song_count_f) * 100, 2)}%)")
    print("Datasets Merged")

    print(" ")

    print("Normalising Dataset")
    scaler = StandardScaler()
    cols_to_normalise = ['danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'loudness','tempo']
    Streaming_history_merged[cols_to_normalise] = scaler.fit_transform(Streaming_history_merged[cols_to_normalise])
    print("Data Normalised")

    print("Writing to csv...")
    Streaming_history_merged.to_parquet('../RAW Data/Combined_Streaming_History.parquet', index=False)
    print("Combined Streaming History saved")

if __name__ == "__main__":
    main()