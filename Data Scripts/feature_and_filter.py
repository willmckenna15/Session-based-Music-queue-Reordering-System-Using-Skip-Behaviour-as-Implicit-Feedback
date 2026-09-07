import pandas as pd
from datetime import timedelta

MIN_SONGS = 7


def filter_valid_sessions(df, min_songs=MIN_SONGS):

    g = df.groupby("session_id")
    keep = (
        (g["session_id"].transform("size") >= min_songs) &
        (g["Artist Name"].transform("nunique") > 1) &
        (g["skipped"].transform("max") > 0)
    )
    return df[keep]

def amend_reason_start(df):
    reason_start_index = {
        "fwdbtn" : 0,
        "trackdone" : 1,
        "clickrow" : 2,
        "backbtn" : 3,
        "playbtn" : 4
    }
 
    df["reason_start"] = df["reason_start"].map(reason_start_index).fillna(5)
    return df

        


def historical_skip_rate(df):

    df = df.sort_values(['user_id', 'spotify_track_uri', 'ts'])


    grouped = df.groupby(['user_id', 'spotify_track_uri'])['skipped']
    
    cumulative_skips = grouped.transform(lambda x: x.shift(1).expanding().sum())
    cumulative_plays = grouped.transform(lambda x: x.shift(1).expanding().count())
    
    df['historical_skip_rate'] = (cumulative_skips / cumulative_plays).fillna(0.0)
    
    df = df.sort_values(['user_id', 'session_id', 'ts'])

    return df

def historical_artist_skip_rate(df):

    df = df.sort_values(['user_id', 'Artist Name', 'ts'])

    grouped = df.groupby(['user_id', 'Artist Name'])['skipped']

    cumulative_skips = grouped.transform(lambda x: x.shift(1).expanding().sum())
    cumulative_plays = grouped.transform(lambda x: x.shift(1).expanding().count())

    df['historical_artist_skip_rate'] = (cumulative_skips / cumulative_plays).fillna(0.0)

    df = df.sort_values(['user_id', 'session_id', 'ts'])

    return df

def add_behavioural_features(df):
    df = df.sort_values(['session_id', 'ts'])
    df['shuffle'] = df['shuffle'].astype(int)
    df['is_repeat_track'] = (df.groupby(['session_id', 'spotify_track_uri']).cumcount() > 0).astype(int)
    df['same_artist_as_prev'] = (
        df.groupby('session_id')['Artist Name'].shift(1) == df['Artist Name']
    ).fillna(False).astype(int)
    return df

def song_position(df):
    df = df.sort_values(['session_id', 'ts'])
    df['song_pos'] = df.groupby('session_id').cumcount()
    return df

def add_track_length(df):
    completed = df[df['reason_end'] == 'trackdone'].groupby('spotify_track_uri')['ms_played'].max()
    
    all_max = df.groupby('spotify_track_uri')['ms_played'].max()
    
    track_lengths = all_max.copy()
    track_lengths = track_lengths.clip(lower=240000) #4 minutes
    track_lengths.update(completed)
    df['track_length'] = df['spotify_track_uri'].map(track_lengths)
    df['ms_played'] = df[['ms_played', 'track_length']].min(axis=1)
    df = df[df['track_length'] > 0]
    return df

def feature_vectors(sessions):
    sessions = sessions.sort_values(["session_id", "ts"]).reset_index(drop=True)
    sessions["ts"] = pd.to_datetime(sessions["ts"])
    sessions["hour"] = sessions["ts"].dt.hour
    sessions["day_of_week"] = sessions["ts"].dt.dayofweek
    sessions["skipped"] = sessions["reason_end"].isin(["fwdbtn", "clickrow"]).astype(int)
    sessions["actively_selected"] = (sessions["reason_start"] == "clickrow").astype(int)

    return sessions

def main():
    print("Reading merged sessions...")
    df = pd.read_parquet("../RAW Data/Combined_Streaming_History.parquet")
    before_rows, before_sessions = len(df), df["session_id"].nunique()

    print("Creating feature vectors...")
    Filtered_sessions = feature_vectors(df)
    Filtered_sessions = add_track_length(Filtered_sessions)      

    print("Applying Secondary Filters...")
    Filtered_sessions = filter_valid_sessions(Filtered_sessions)
    print(f"  {before_rows:,} rows / {before_sessions:,} sessions -> "
          f"{len(Filtered_sessions):,} rows / {Filtered_sessions['session_id'].nunique():,} sessions")

    print("Deriving sequential features...")
    Filtered_sessions = historical_skip_rate(Filtered_sessions)
    Filtered_sessions = historical_artist_skip_rate(Filtered_sessions)
    Filtered_sessions = add_behavioural_features(Filtered_sessions)
    Filtered_sessions = song_position(Filtered_sessions)
    Filtered_sessions = amend_reason_start(Filtered_sessions)

    lengths = Filtered_sessions.groupby("session_id").size()
    print(f"  session length: min {lengths.min()}, median {lengths.median():.0f}, max {lengths.max()}")
    if lengths.min() < MIN_SONGS:
        print(f"  WARNING: {(lengths < MIN_SONGS).sum()} session(s) below the {MIN_SONGS}-song floor")

    print("Writing to parquet...")
    Filtered_sessions.to_parquet("../RAW Data/Filtered_Sessions.parquet", index=False)
    print("Filtered sessions saved")

    valid_sessions = Filtered_sessions["session_id"].nunique()
    song_count = len(Filtered_sessions)
    agg_skip_count = int(Filtered_sessions["skipped"].sum())

    print("Formatting Statistics...")
    if not Filtered_sessions.empty:
        user_stats = Filtered_sessions.groupby("user_id").agg(
            valid_sessions=("session_id", "nunique"),
            avg_songs_per_session=("session_id", lambda x: len(x) / x.nunique()),
            skip_rate=("reason_end", lambda x: x.isin(["fwdbtn", "clickrow"]).mean() * 100)
        ).reset_index().rename(columns={
            "user_id": "UUID",
            "valid_sessions": "Number of Sessions",
            "avg_songs_per_session": "Average Songs per Session",
            "skip_rate": "Skip Rate"
        })

        session_file = "../Session Data.csv"
        
        try:
            sessions_df = pd.read_csv(session_file)
        except (FileNotFoundError, pd.errors.EmptyDataError):
            sessions_df = pd.DataFrame(columns=["UUID", "Number of Sessions", "Average Songs per Session", "Skip Rate"])

        print("writing to csv..")
        sessions_df = pd.concat([sessions_df, user_stats], ignore_index=True)
        sessions_df.to_csv(session_file, index=False)
        print("csv saved")

        print(user_stats)
        print(f"\nTotal valid sessions: {valid_sessions}")
        print(f"\nNumber of songs: {song_count}")
        print(f"\nBaseline Skip Rate: {agg_skip_count / song_count * 100:.2f}%")