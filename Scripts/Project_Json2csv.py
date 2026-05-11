import uuid
import glob
import pandas as pd
import os

def main():
    base_path = "../Spotify Extended Streaming Histories"
    folders = sorted(glob.glob(f"{base_path}/Spotify Extended Streaming History*/"))

    for folder in folders:
        user_id = str(uuid.uuid4())
        json_files = glob.glob(os.path.join(folder, "Streaming_History_Audio*.json"))

        if not json_files:
            print(f"No JSON files found in {folder}, skipping...")
            continue

        df = pd.concat([pd.read_json(f) for f in json_files], ignore_index=True)
        df["user_id"] = user_id
        df.to_csv(f'../RAW Data/Streaming_history_{user_id[:4]}.csv', index=False)
        print(f"Streaming history for user {user_id[:4]} has been saved ({folder})")
    
    print("Combining Streaming Histories...")
    Streaming_histories = glob.glob("../RAW Data/Streaming_history_*")
    file_no = len(Streaming_histories)
    Comb_hist = pd.concat([pd.read_csv(f) for f in Streaming_histories], ignore_index = True)
    Comb_hist.to_csv("../RAW Data/Combined_Streaming_History.csv")
    print(f"Streaming Histories Combined for {file_no} volunteers")
if __name__ == "__main__":
    main()