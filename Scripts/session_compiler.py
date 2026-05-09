import pandas as pd
from datetime import timedelta

def build_sessions(df):
    streaming_sessions = {}
    session_no = 1
    shuffle_sessions = df[df["shuffle"] == True].copy()
    shuffle_sessions["ts"] = pd.to_datetime(shuffle_sessions["ts"])
    shuffle_sessions = shuffle_sessions.sort_values("ts").reset_index(drop=True)

    for i in range(len(shuffle_sessions) - 1):
        streaming_sessions.setdefault(session_no, []).append(shuffle_sessions.iloc[i].to_dict())
        if shuffle_sessions.iloc[i + 1]["ts"] - shuffle_sessions.iloc[i]["ts"] >= timedelta(minutes=30):
            session_no += 1

    streaming_sessions.setdefault(session_no, []).append(shuffle_sessions.iloc[-1].to_dict())
    return streaming_sessions

def is_valid_session(songs):
    if len(songs) < 10:
        return False
    if len(set(song["master_metadata_album_artist_name"] for song in songs)) == 1:
        return False
    if not any(song["reason_end"] in ("fwdbtn", "clickrow") for song in songs):
        return False
    return True

def session_compiler():
    df = pd.read_csv("../RAW Data/Combined_Streaming_History.csv")
    streaming_sessions = build_sessions(df)

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

    return Filtered_sessions, valid_sessions, song_count, agg_skip_count

if __name__ == "__main__":
    Filtered_sessions, valid_sessions, song_count, agg_skip_count = session_compiler()

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

        sessions_df = pd.concat([sessions_df, user_stats], ignore_index=True)
        sessions_df.to_csv(session_file, index=False)

        print(user_stats)
        print(f"\nTotal valid sessions: {valid_sessions}")
        print(f"\nNumber of songs: {song_count}")
        print(f"\nBaseline Skip Rate: {agg_skip_count / song_count * 100:.2f}%")