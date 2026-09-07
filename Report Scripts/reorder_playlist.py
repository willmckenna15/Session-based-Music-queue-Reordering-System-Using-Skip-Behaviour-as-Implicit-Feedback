"""Reorder a real playlist with the trained model, using one person's own history.

    python3 reorder_playlist.py --playlist my_playlist.csv
    python3 reorder_playlist.py --playlist my_playlist.csv --shuffle 0 --top 20

The playlist file needs a `spotify_track_uri` column (bare id or spotify:track:...).
Track name and artist are used for display if present, otherwise taken from the
audio-feature corpus. Historical skip rates come from the streaming history in
--history, so the features match those the model was trained on.

There is no ground truth here: the output is the ordering the deployed system
would produce, not a measurement of whether it is correct.
"""

import argparse, glob, json, os, sys
import numpy as np, pandas as pd, torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'Model Scripts'))
from Model_lib import SkipLSTM

AUDIO = ['tempo', 'mode', 'danceability', 'energy', 'loudness', 'speechiness',
         'acousticness', 'instrumentalness', 'liveness', 'valence']
BEHAV = ['historical_skip_rate', 'historical_artist_skip_rate', 'shuffle',
         'is_repeat_track', 'same_artist_as_prev']
FEATURES = AUDIO + BEHAV
NORMALISE = ['danceability', 'energy', 'speechiness', 'acousticness',
             'instrumentalness', 'liveness', 'valence', 'loudness', 'tempo']

cli = argparse.ArgumentParser()
cli.add_argument('--playlist', required=True)
cli.add_argument('--history', default='../Spotify Extended Streaming Histories/Spotify Extended Streaming History 1')
cli.add_argument('--corpus', default='../RAW Data/audio_features')
cli.add_argument('--checkpoint', default='../Models/lstm_pwts_best.pt')
cli.add_argument('--params', default='../Models/lstm_pwts_best_params.json')
cli.add_argument('--shuffle', type=int, default=1, choices=[0, 1])
cli.add_argument('--top', type=int, default=None, help='show only the first N of each ordering')
cli.add_argument('--out', default=None)
opts = cli.parse_args()


def bare(uri):
    """Accept spotify:track:ID, an open.spotify.com/track/ID link, or a bare id."""
    u = str(uri).strip()
    if 'open.spotify.com/track/' in u:
        u = u.split('open.spotify.com/track/')[1]
    u = u.replace('spotify:track:', '')
    return u.split('?')[0].split('/')[0].strip()


def load_playlist(path):
    """A CSV with a track id column, or a plain text file of one link per line -
    which is what selecting a playlist in the desktop app and copying produces."""
    if path.lower().endswith('.csv'):
        df = pd.read_csv(path)
        col = next((c for c in df.columns
                    if 'uri' in c.lower() or c.lower() in ('id', 'track_id', 'link', 'url')), None)
        if col is None:
            sys.exit(f'No track id column in {path}. Columns: {list(df.columns)}')
        df['track_id'] = df[col].map(bare)
        return df
    ids = [bare(l) for l in open(path) if 'track' in l or len(l.strip()) == 22]
    ids = [i for i in ids if len(i) == 22]
    if not ids:
        sys.exit(f'No track ids found in {path}')
    return pd.DataFrame({'track_id': ids})


##  1. the listener's own history -> historical skip rates

files = sorted(glob.glob(os.path.join(opts.history, 'Streaming_History_Audio*.json')))
if not files:
    sys.exit(f'No history files in {opts.history}')
hist = pd.concat([pd.read_json(f) for f in files], ignore_index=True)
hist = hist.dropna(subset=['spotify_track_uri'])
hist['track_id'] = hist['spotify_track_uri'].map(bare)
hist['artist'] = hist['master_metadata_album_artist_name']
hist['skipped'] = hist['reason_end'].isin(['fwdbtn', 'clickrow']).astype(int)

track_rate = hist.groupby('track_id')['skipped'].mean()
artist_rate = hist.groupby('artist')['skipped'].mean()
print(f'history: {len(hist):,} plays | {hist.track_id.nunique():,} tracks | '
      f'{hist.artist.nunique():,} artists | skip rate {hist.skipped.mean():.3f}')


##  2. the playlist

pl = load_playlist(opts.playlist).drop_duplicates('track_id').reset_index(drop=True)
print(f'playlist: {len(pl)} tracks from {opts.playlist}')


##  3. audio features, scanned from the corpus one shard at a time

want = set(pl.track_id)
cols = ['id', 'name'] + AUDIO
chunks = []
for f in sorted(glob.glob(os.path.join(opts.corpus, '*.parquet'))):
    df = pd.read_parquet(f, columns=cols)
    chunks.append(df[df['id'].isin(want)])
    del df
feats = pd.concat(chunks, ignore_index=True).drop_duplicates('id')
print(f'audio features matched for {len(feats)} of {len(pl)} tracks')

q = pl.merge(feats, left_on='track_id', right_on='id', how='inner')
if len(q) < 6:
    sys.exit(f'Only {len(q)} tracks matched the corpus - too few to rank.')

name_col = next((c for c in ('Track Name', 'track_name', 'name') if c in q.columns), 'name')
art_col = next((c for c in ('Artist Name', 'artist_name', 'artist') if c in q.columns), None)
q['display'] = q[name_col].astype(str)
q['artist'] = q[art_col].astype(str) if art_col else ''


##  4. behavioural features, matching the training pipeline

q['historical_skip_rate'] = q.track_id.map(track_rate).fillna(0.0)
q['historical_artist_skip_rate'] = q.artist.map(artist_rate).fillna(0.0)
q['shuffle'] = opts.shuffle
q['is_repeat_track'] = (q.groupby('track_id').cumcount() > 0).astype(int)
q['same_artist_as_prev'] = (q.artist.shift(1) == q.artist).fillna(False).astype(int)

seen = (q.historical_skip_rate > 0).sum()
print(f'{seen} of {len(q)} tracks have prior skip history for this listener')

# per-session z-scoring, exactly as dataset_splitter.py does it
mu, sd = q[NORMALISE].mean(), q[NORMALISE].std().replace(0, np.nan)
q[NORMALISE] = ((q[NORMALISE] - mu) / sd).fillna(0.0)


##  5. score

params = json.load(open(opts.params))
model = SkipLSTM(len(FEATURES), int(params['hidden_units']),
                 int(params['num_layers']), float(params['dropout_rate']))
model.load_state_dict(torch.load(opts.checkpoint, map_location='cpu'))
model.eval()

L = len(q)
x = torch.tensor(q[FEATURES].values, dtype=torch.float32).unsqueeze(0)
status = torch.full((1, L), 2, dtype=torch.long)      # nothing played yet: all unknown
with torch.no_grad():
    logits = model(x, torch.tensor([L]), status, torch.tensor([0]))[0].numpy()
q['p_skip'] = 1 / (1 + np.exp(-logits))

order = q.sort_values('p_skip', kind='stable').reset_index(drop=True)
if opts.out:
    order[['track_id', 'display', 'artist', 'p_skip']].to_csv(opts.out, index=False)


##  6. show

def show(frame, title):
    print(f'\n{title}' + ' ' * max(1, 54 - len(title)) + 'P(skip)')
    rows = frame if opts.top is None else frame.head(opts.top)
    for i, r in rows.iterrows():
        print(f'  {i+1:>3}. {r.display[:34]:<34} {r.artist[:18]:<18} {r.p_skip:.3f}')
    if opts.top is not None and len(frame) > opts.top:
        print(f'       ... {len(frame) - opts.top} more')

show(q, 'PLAYLIST - as it stands')
show(order, "PLAYLIST - reordered by the model")
print(f'\nP(skip) ranges {q.p_skip.min():.3f} to {q.p_skip.max():.3f}, '
      f'mean {q.p_skip.mean():.3f}')
print('No ground truth here - this is the ordering the system would produce, '
      'not a measurement of whether it is correct.')
