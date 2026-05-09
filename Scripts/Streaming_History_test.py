from session_compiler import session_compiler
import pandas as pd

while True:
    user_id = input("First four letters of UUID: ")
    try:
        df = pd.read_csv(f"Streaming_History_{user_id}.csv").copy()
    except FileNotFoundError:
        print(f"No file found for UUID: {user_id}")
        continue
    break
df = df.copy()

df = df.rename(columns={
    'master_metadata_track_name': 'Track Name',
    'master_metadata_album_artist_name': 'Artist Name'
})

most_listened_songs = df.groupby(['Track Name', 'Artist Name']).size().rename('Play Count')

most_listened_artists = df.groupby(['Artist Name']).size().rename('Play Count')

df_skipped = df[df["skipped"] == True].copy()
skip_counts_songs = df_skipped.groupby(['Track Name', 'Artist Name']).size().rename('Skip Count')

skip_counts_artists = df_skipped.groupby(['Artist Name']).size().rename('Skip Count')

# Merge play count and skip count, then calculate skip rate
songs_df = pd.concat([most_listened_songs, skip_counts_songs], axis=1).fillna(0)
songs_df['Skip Rate'] = songs_df['Skip Count'] / songs_df['Play Count'] * 100
skip_songs_df = songs_df[songs_df['Play Count'] >= 50].copy()
most_skipped_songs_df = skip_songs_df.sort_values('Skip Rate', ascending=False).head(50)
most_listened_songs_df = songs_df.sort_values('Play Count',ascending = False)


artists_df = pd.concat([most_listened_artists, skip_counts_artists], axis=1).fillna(0)
artists_df['Skip Rate'] = artists_df['Skip Count'] / artists_df['Play Count'] * 100
skip_artists_df = artists_df[artists_df['Play Count'] >= 50].copy()
most_skipped_artists_df = skip_artists_df.sort_values('Skip Rate', ascending=False)
most_listened_artists_df = artists_df.sort_values('Play Count',ascending = False)

'''with pd.ExcelWriter(f"{user_id} Song Stats.xlsx") as writer:
    most_skipped_songs_df.to_excel(writer,sheet_name="Most Skipped Songs")
    most_listened_songs_df.to_excel(writer,sheet_name="Most Played Songs")
    most_skipped_artists_df.to_excel(writer,sheet_name="Most Skipped Artists")
    most_listened_artists_df.to_excel(writer,sheet_name="Most Played Artists")
    for sheet_name, df_out in {
    "Most Skipped Songs": most_skipped_songs_df,
    "Most Played Songs": most_listened_songs_df,
    "Most Skipped Artists": most_skipped_artists_df,
    "Most Played Artists": most_listened_artists_df
    }.items():
        worksheet = writer.sheets[sheet_name]
        skip_rate_col = df_out.columns.get_loc('Skip Rate') + 2  # +2 for index and 1-based
        for row in range(2, len(df_out) + 2):
            worksheet.cell(row=row, column=skip_rate_col).number_format = '0.00"%"'
    for sheet in writer.sheets.values():
      for column in sheet.columns:
          max_length = max(len(str(cell.value)) for cell in column if cell.value)
          sheet.column_dimensions[column[0].column_letter].width = max_length + 2'''

df["ts"] = pd.to_datetime(df["ts"])
df["date"] = df["ts"].dt.date
df["time"] = df["ts"].dt.time



df_date = df.groupby('date')['ms_played'].sum().reset_index()
df_date = df_date.sort_values('ms_played',ascending = False)
df_date.to_csv("first_songs.csv", index=False)



