import Json2csv
import audio_features_clean
import session_compiler
import dataset_splitter
import os
import glob
import feature_and_filter

print("removing pre-existing files...")
# Each deletion is guarded separately. Previously one try/except wrapped them all, so a single
# already-missing file aborted the rest and left stale parquets behind for the next stage to
# read.
stale = (glob.glob("../RAW Data/Streaming_history_*.csv")
         + glob.glob("../RAW Data/*.parquet")
         + ["../RAW Data/Combined_Streaming_History.csv", "../Session Data.csv"])
for f in stale:
    try:
        os.remove(f)
    except FileNotFoundError:
        pass
print("Pre-existing files removed")

print("\033[1;4m\nCombining RAW Datasets\033[0m")
Json2csv.main()
print("\033[1;4mRAW Datasets Combined\033[0m")

print(" ")

print("\033[1;4m\nConstructing Streaming Sessions\033[0m")
session_compiler.main()
print("\033[1;4mStreaming Sessions Finalised\033[0m")

print(" ")

print("\033[1;4mMerging Streaming Sessions with Audio Features \033[0m")
audio_features_clean.main()
print("\033[1;4mDatasets Merged\033[0m")

print("\033[1;4m\nCreating sequential features and applying filters... \033[0m")
feature_and_filter.main()
print("\033[1;4mFilters applied\033[0m")

print("\033[1;4m\nSplitting Dataset into training, validation & testing...\033[0m")
dataset_splitter.main()
print("\033[1;4mDatasets split.\033[0m")