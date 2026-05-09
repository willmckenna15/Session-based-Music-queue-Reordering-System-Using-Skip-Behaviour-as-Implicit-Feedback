import pandas as pd
from datetime import timedelta
def session_compiler():
    while True:
        user_id = input("First four letters of UUID: ")
        try:
            df = pd.read_csv(f"Streaming_History_{user_id}.csv")
        except FileNotFoundError:
            print(f"No file found for UUID: {user_id}")
            continue
        break

    shuffle_sessions = df[df["shuffle"]==True].copy()

    shuffle_sessions["ts"] = pd.to_datetime(shuffle_sessions["ts"])
    shuffle_sessions["date"] = shuffle_sessions["ts"].dt.date
    shuffle_sessions["time"] = shuffle_sessions["ts"].dt.time


    shuffle_sessions = shuffle_sessions.sort_values("ts").reset_index(drop=True)

    streaming_sessions = {}
    session_no = 1

    for i in range(0, shuffle_sessions.shape[0] - 1):
        song = shuffle_sessions.iloc[i]

        if session_no not in streaming_sessions:
            streaming_sessions[session_no] = []
        
        streaming_sessions[session_no].append(song)

        if shuffle_sessions.iloc[i + 1]["ts"] - song["ts"] >= timedelta(minutes=30):
            session_no += 1

    if session_no not in streaming_sessions:
        streaming_sessions[session_no] = []
    streaming_sessions[session_no].append(shuffle_sessions.iloc[-1])

    Filtered_sessions = pd.DataFrame()
    valid_sessions = 0
    song_count=0
    agg_skip_count = 0
    for session_no, songs in streaming_sessions.items():
        if len(songs) < 10:
            continue
        if len(set(song["master_metadata_album_artist_name"] for song in songs)) == 1:
            continue

        skip_count = 0
        for song in songs:
            if song["reason_end"] == "fwdbtn" or song["reason_end"] == "clickrow":
                skip_count += 1
                agg_skip_count += 1
        if skip_count == 0:
            continue
        Filtered_sessions = pd.concat([Filtered_sessions,pd.DataFrame(songs)]).reset_index(drop=True)
        song_count += len(songs)
        valid_sessions += 1
        #sesh.to_csv(f"session_{session_no}.csv", index=False))
    return Filtered_sessions, valid_sessions, song_count, agg_skip_count

if __name__ == "__main__":
    Filtered_sessions, valid_sessions, song_count, agg_skip_count = session_compiler()

    sessions_df = pd.read_csv('Sessions_data.csv')
    new_row = pd.DataFrame([{
        'UUID': Filtered_sessions['user_id'].iloc[0],
        'Number of Sessions': valid_sessions,
        'Number of Songs': song_count,
        'Skip Rate': agg_skip_count/song_count * 100
    }])

    sessions_df = pd.concat([sessions_df,new_row]).reset_index(drop=True)
    print(sessions_df.head())
    sessions_df.to_csv('Sessions_data.csv', index=False)

    print(" ")
    print(f"Total valid sessions: {valid_sessions}")
    print(" ")
    print(f"Number of songs: {song_count}")
    print(" ")
    print(f"Baseline Skip Rate: {agg_skip_count/song_count * 100}")
    print(" ")