import pandas as pd


def main():
    INPUT = '../RAW Data/Combined_Streaming_History.parquet'

    sessions = pd.read_parquet(INPUT)

    users = sessions["user_id"].unique()

    training_parts = []
    validation_parts = []
    testing_parts = []



    for user in users:
        user_sessions = sessions[sessions["user_id"] == user].sort_values("ts", ascending =True)
        
        session_ids = user_sessions["session_id"].unique()
        n = len(session_ids)
        
        train_end = int(n * 0.7)
        val_end = int(n * 0.85)
        
        train_ids = session_ids[:train_end]
        val_ids = session_ids[train_end:val_end]
        test_ids = session_ids[val_end:]

        training_parts.append(user_sessions[user_sessions["session_id"].isin(train_ids)])
        validation_parts.append(user_sessions[user_sessions["session_id"].isin(val_ids)])
        testing_parts.append(user_sessions[user_sessions["session_id"].isin(test_ids)])

    training_sessions = pd.concat(training_parts, ignore_index=True)
    validation_sessions = pd.concat(validation_parts, ignore_index=True)
    testing_sessions = pd.concat(testing_parts, ignore_index=True)

    training_sessions.to_parquet("../RAW Data/training_data.parquet")
    validation_sessions.to_parquet("../RAW Data/validation_data.parquet")
    testing_sessions.to_parquet("../RAW Data/testing_data.parquet")
    print(f"Training: {len(training_sessions)} rows, {training_sessions['session_id'].nunique()} sessions")
    print(f"Validation: {len(validation_sessions)} rows, {validation_sessions['session_id'].nunique()} sessions")
    print(f"Testing: {len(testing_sessions)} rows, {testing_sessions['session_id'].nunique()} sessions")    

if __name__ == "__main__":
    main()

