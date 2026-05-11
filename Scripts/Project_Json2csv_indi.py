import pandas as pd
import glob
import os
import uuid

def main(name):
    Folder_no = input("Number of folder: ")
    user_id = str(uuid.uuid4())

    json_files = glob.glob(f"../Spotify Extended Streaming Histories/Spotify Extended Streaming History {Folder_no}/Streaming_History_Audio*.json")
    df = pd.concat([pd.read_json(f) for f in json_files], ignore_index=True)
    df["user_id"] = user_id
    df.to_csv(f'../Volunteer Data/RAW/Streaming_history_{name}.csv', index=False)
    print(f"Streaming history for user {name} has been saved")
    print(" ")
if __name__ == "__main__":
    main()
