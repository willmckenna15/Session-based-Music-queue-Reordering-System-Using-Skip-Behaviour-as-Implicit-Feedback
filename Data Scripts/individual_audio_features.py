import pandas as pd
from sklearn.preprocessing import StandardScaler
import glob

def main(user_id):

    print("Reading csv...")
    csv_path = f'../Volunteer Data/RAW/Streaming_history_{user_id}.csv'
    Streaming_history = pd.read_csv(csv_path)
    print("CSV Read")

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

    audio_cols = ['id', 'tempo', 'key', 'mode', 'danceability', 'energy', 'loudness',
                'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'popularity']

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

    Streaming_history_merged.to_csv(f'../Volunteer Data/RAW/Audio_Streaming_History_{user_id}.csv', index=False)
    print(f"Audio features for user {user_id} have been added")
    print(" ")

if __name__ == "__main__":
    main()