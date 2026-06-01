import Project_Json2csv
import audio_features_clean
import session_compiler
import dataset_splitter
import os
import glob

print("removing pre-existing files...")
try:
    files = glob.glob("../RAW Data/Streaming_history_*.csv")
    for f in files:
        os.remove(f)

    os.remove("../RAW Data/Combined_Streaming_History.csv")
    os.remove("../Session Data.csv")
    p_files=glob.glob("../RAW Data/*.parquet")
    for p in p_files:
        os.remove(p)
except FileNotFoundError:
    pass
print("Pre-existing files removed")

print("\033[1;4mCombining RAW Datasets\033[0m")
Project_Json2csv.main()
print("\033[1;4mRAW Datasets Combined\033[0m")

print(" ")

print("\033[1;4mConstructing Streaming Sessions\033[0m")
session_compiler.main()
print("\033[1;4mStreaming Sessions Finalised\033[0m")

print(" ")

print("\033[1;4mMerging Streaming Sessions with Audio Features \033[0m")
audio_features_clean.main()
print("\033[1;4mDatasets Merged\033[0m")

print("\033[1;4mSplitting Dataset into training, validation & testing...\033[0m")
dataset_splitter.main()
print("\033[1;4mDatasets split.\033[0m")