"""Interactive listening session over my own playlist.

    python3 simulate_session.py          # 100-track queue
    python3 simulate_session.py 30       # or pick the size

Shows the track playing and the next 20 in the queue, each with the model's
P(skip).  Press s to skip, f to play through, q to quit.  A skip reorders
everything still to come, cheapest-to-skip first.
"""

import glob, json, os, sys, termios, tty
import numpy as np, pandas as pd, torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'Model Scripts'))
from Model_lib import SkipLSTM

PLAYLIST   = os.path.join(HERE, 'my_playlist.txt')
HISTORY    = os.path.join(HERE, '..', 'Spotify Extended Streaming Histories',
                          'Spotify Extended Streaming History 1')
CORPUS     = os.path.join(HERE, '..', 'RAW Data', 'audio_features')
CHECKPOINT = os.path.join(HERE, '..', 'Models', 'lstm_pwts_best.pt')
PARAMS     = os.path.join(HERE, '..', 'Models', 'lstm_pwts_best_params.json')
CACHE      = os.path.join(HERE, '.playlist_cache.parquet')
QUEUE      = int(sys.argv[1]) if len(sys.argv) > 1 else 100   # queue size
SHOW       = 20

AUDIO = ['tempo', 'mode', 'danceability', 'energy', 'loudness', 'speechiness',
         'acousticness', 'instrumentalness', 'liveness', 'valence']
BEHAV = ['historical_skip_rate', 'historical_artist_skip_rate', 'shuffle',
         'is_repeat_track', 'same_artist_as_prev']
FEATURES  = AUDIO + BEHAV
NORMALISE = ['tempo', 'danceability', 'energy', 'loudness', 'speechiness',
             'acousticness', 'instrumentalness', 'liveness', 'valence']
DIM, OK, BOLD, OFF = '\033[90m', '\033[32m', '\033[1m', '\033[0m'


def bare(url):
    return url.strip().split('open.spotify.com/track/')[-1].split('?')[0]


def build():
    """Match the playlist against the audio corpus and my own listening history."""
    files = sorted(glob.glob(os.path.join(HISTORY, 'Streaming_History_Audio*.json')))
    hist = pd.concat([pd.read_json(f) for f in files], ignore_index=True)
    hist = hist.dropna(subset=['spotify_track_uri'])
    hist['track_id'] = hist['spotify_track_uri'].str.replace('spotify:track:', '')
    hist['skipped'] = hist['reason_end'].isin(['fwdbtn', 'clickrow']).astype(int)
    track_rate = hist.groupby('track_id')['skipped'].mean()
    artist_rate = hist.groupby('master_metadata_album_artist_name')['skipped'].mean()
    artists = hist.drop_duplicates('track_id').set_index('track_id')[
        'master_metadata_album_artist_name']
    print(f'history: {len(hist):,} plays, skip rate {hist.skipped.mean():.3f}')

    ids = [bare(l) for l in open(PLAYLIST) if 'open.spotify.com/track/' in l]
    ids = [i for i in dict.fromkeys(ids) if len(i) == 22]
    print(f'playlist: {len(ids)} tracks — scanning corpus, takes a few minutes')

    chunks = []
    for f in sorted(glob.glob(os.path.join(CORPUS, '*.parquet'))):
        d = pd.read_parquet(f, columns=['id', 'name'] + AUDIO)
        chunks.append(d[d['id'].isin(set(ids))])
    feats = pd.concat(chunks, ignore_index=True).drop_duplicates('id')

    q = pd.DataFrame({'track_id': ids}).merge(feats, left_on='track_id', right_on='id')
    q['title'] = q['name'].astype(str)
    q['artist'] = q['track_id'].map(artists).fillna('')
    q['historical_skip_rate'] = q['track_id'].map(track_rate).fillna(0.0)
    q['historical_artist_skip_rate'] = q['artist'].map(artist_rate).fillna(0.0)
    q['shuffle'] = 1
    print(f'matched {len(q)} of {len(ids)} tracks')
    return q[['title', 'artist'] + AUDIO + BEHAV[:3]]


if os.path.exists(CACHE):
    q = pd.read_parquet(CACHE)
else:
    q = build()
    q.to_parquet(CACHE, index=False)

q = q.sample(n=min(QUEUE, len(q))).reset_index(drop=True)   # a random, shuffled queue
q[NORMALISE] = ((q[NORMALISE] - q[NORMALISE].mean()) / q[NORMALISE].std()).fillna(0.0)
q['is_repeat_track'] = 0                             # the playlist has no duplicates
q['same_artist_as_prev'] = 0

cfg = json.load(open(PARAMS))
model = SkipLSTM(len(FEATURES), int(cfg['hidden_units']),
                 int(cfg['num_layers']), float(cfg['dropout_rate']))
model.load_state_dict(torch.load(CHECKPOINT, map_location='cpu'))
model.eval()

played, outcomes, queue = [], [], list(range(len(q)))


def score(seq):
    """P(skip) for every track in seq; outcomes are known for the played prefix."""
    f = q.iloc[seq][FEATURES].reset_index(drop=True)
    artist = q.iloc[seq]['artist'].reset_index(drop=True)
    f['same_artist_as_prev'] = (artist.shift(1) == artist).fillna(False).astype(int)
    x = torch.tensor(f.values, dtype=torch.float32).unsqueeze(0)
    status = torch.full((1, len(seq)), 2, dtype=torch.long)
    if outcomes:
        status[0, :len(outcomes)] = torch.tensor(outcomes)
    with torch.no_grad():
        logits = model(x, torch.tensor([len(seq)]), status, torch.tensor([len(played)]))
    return 1 / (1 + np.exp(-logits[0].numpy()))


def getkey():
    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1).lower()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)


p = score(queue)          # scored once, so the shuffled queue has numbers

while queue:
    now, n = queue[0], len(played)
    print('\033[2J\033[H', end='')
    print(f'{BOLD}{n} played   {sum(outcomes)} skipped   {len(queue)} left{OFF}\n')
    print(f'{BOLD}{OK}▶ PLAYING{OFF}  {q.title[now][:42]:<42} {DIM}{q.artist[now][:22]}{OFF}')
    print(f'           {DIM}P(skip) {p[n]:.3f}{OFF}\n')
    print(f'{BOLD}NEXT{OFF}{"":<57}{DIM}P(skip){OFF}')
    for i, t in enumerate(queue[1:SHOW + 1], start=1):
        print(f'  {i:>2}. {q.title[t][:38]:<38} {DIM}{q.artist[t][:20]:<20}{OFF}  {p[n + i]:.3f}')
    print(f'\n{DIM}[s] skip   [f] full play   [q] quit{OFF}')

    key = getkey()
    if key == 'q':
        break
    if key not in ('s', 'f'):
        continue

    outcomes.append(int(key == 's'))
    played.append(queue.pop(0))
    if key == 's':
        # A skip is the only event that runs the model.  The scores that
        # produced the ranking are carried through rather than recomputed: a
        # causal model scores each track in the context it now sits in, so
        # rescoring after the move would print different numbers.
        p = score(played + queue)
        order = np.argsort(p[len(played):], kind='stable')
        queue = [queue[i] for i in order]
        p = np.concatenate([p[:len(played)], p[len(played):][order]])

print(f'\n{BOLD}SESSION ENDED{OFF}  {len(played)} played, {sum(outcomes)} skipped')
