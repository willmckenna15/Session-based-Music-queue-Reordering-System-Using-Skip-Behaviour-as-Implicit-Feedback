import pandas as pd
from datetime import timedelta


def build_sessions(df):
    print("Constructing sessions...")
    
    df = df.copy()
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.sort_values(["user_id", "ts"]).reset_index(drop=True)

    df["time_gap"] = df.groupby("user_id")["ts"].diff()

    new_session = (
        (df["time_gap"] > pd.Timedelta(minutes=30)) |
        (df["user_id"] != df["user_id"].shift(1))
    )

    df["session_id"] = new_session.cumsum()
    df = df.drop(columns=["time_gap"])

    streaming_sessions = {
        session_id: group.to_dict('records')
        for session_id, group in df.groupby("session_id")
    }

    print("All Sessions Constructed")
    return streaming_sessions

def is_valid_session(songs):
    if len(songs) < 10:
        return False
    if len(set(song["Artist Name"] for song in songs)) == 1:
        return False
    if not any(song["reason_end"] in ("fwdbtn", "clickrow") for song in songs):
        return False
    return True

def session_compiler():
    df = pd.read_csv("../RAW Data/Combined_Streaming_History.csv")

    
    streaming_sessions = build_sessions(df)

    print("Applying Secondary Filters...")
    valid = {k: v for k, v in streaming_sessions.items() if is_valid_session(v)}

    Filtered_sessions = pd.DataFrame([
        {**song, "session_id": f"{song['user_id']}_{session_no}"}
        for session_no, songs in valid.items()
        for song in songs
    ])

    if Filtered_sessions.empty:
        print("No valid sessions found.")
        return Filtered_sessions, 0, 0, 0

    valid_sessions = Filtered_sessions["session_id"].nunique()
    song_count = len(Filtered_sessions)
    agg_skip_count = Filtered_sessions["reason_end"].isin(["fwdbtn", "clickrow"]).sum()
    print("Sessions Filtered")
    return Filtered_sessions, valid_sessions, song_count, agg_skip_count

def feature_vectors(sessions):
    sessions = sessions.sort_values(["session_id", "ts"]).reset_index(drop=True)
    sessions["ts"] = pd.to_datetime(sessions["ts"])
    sessions["hour"] = sessions["ts"].dt.hour
    sessions["day_of_week"] = sessions["ts"].dt.dayofweek
    sessions["skipped"] = sessions["reason_end"].isin(["fwdbtn", "clickrow"]).astype(int)
    sessions["actively_selected"] = (sessions["reason_start"] == "clickrow").astype(int)

    return sessions

def historical_skip_rate(df):

    df = df.sort_values(['user_id', 'spotify_track_uri', 'session_id'])
    

    grouped = df.groupby(['user_id', 'spotify_track_uri'])['skipped']
    
    cumulative_skips = grouped.transform(lambda x: x.shift(1).expanding().sum())
    cumulative_plays = grouped.transform(lambda x: x.shift(1).expanding().count())
    
    df['historical_skip_rate'] = (cumulative_skips / cumulative_plays).fillna(0.0)
    
    df = df.sort_values(['user_id', 'session_id', 'ts'])

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
        

        



def main():
    Filtered_sessions, valid_sessions, song_count, agg_skip_count = session_compiler()

    print("Creating features vectors...")
    Filtered_sessions = feature_vectors(Filtered_sessions)
    Filtered_sessions = historical_skip_rate(Filtered_sessions)
    Filtered_sessions = song_position(Filtered_sessions)
    Filtered_sessions = add_track_length(Filtered_sessions)
    print("Writing to parquet...")
    print(Filtered_sessions.head(5))

    Filtered_sessions.to_parquet("../RAW Data/Filtered_Sessions.parquet", index=False)
    print("Filtered sessions saved")

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


