import Project_Json2csv_indi
import individual_audio_features
import Streaming_History_stats

name = input("Name of Volunteer: ")
print("Converting to csv...")
Project_Json2csv_indi.main(name)
print("Combining with audio features...")
individual_audio_features.main(name)
print("Running Analytics...")
Streaming_History_stats.main(name)
