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

    # session_id stays "<user_id>_<n>" so it remains unique per user and readable downstream
    df["session_id"] = df["user_id"].astype(str) + "_" + new_session.cumsum().astype(str)
    df = df.drop(columns=["time_gap"])

    print("All Sessions Constructed")
    return df


def session_compiler():
    df = pd.read_csv("../RAW Data/Combined_Streaming_History.csv")
    return build_sessions(df)


def main():
    streaming_sessions = session_compiler()

    streaming_sessions.to_parquet("../RAW Data/Streaming_Sessions.parquet", index=False)
    print(f"Streaming sessions saved: {len(streaming_sessions):,} rows, "
          f"{streaming_sessions['session_id'].nunique():,} sessions")
