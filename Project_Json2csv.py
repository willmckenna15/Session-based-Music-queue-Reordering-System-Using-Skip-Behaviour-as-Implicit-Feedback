import pandas as pd
import glob
import os
import uuid

user_id = str(uuid.uuid4())

json_files = glob.glob("Spotify Extended Streaming History 2/Streaming_History_Audio*.json")
df = pd.concat([pd.read_json(f) for f in json_files], ignore_index=True)
df["user_id"] = user_id
df.to_csv(f'Streaming_history_{user_id[:4]}.csv', index=False)