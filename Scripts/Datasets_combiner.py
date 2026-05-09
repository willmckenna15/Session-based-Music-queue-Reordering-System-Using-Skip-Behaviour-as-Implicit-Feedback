import pandas as pd
import glob
import os
import uuid

user_id = str(uuid.uuid4())

csv_files = glob.glob("../RAW Data/Streaming_history_*.csv")
df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
df.to_csv(f'../RAW Data/Combined_Streaming_History.csv', index=False)
