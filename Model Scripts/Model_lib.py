import numpy as np
import torch
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
from tqdm import tqdm
import os
import argparse
import random
import itertools
import signal
from Loss_functions import get_loss_criterion, parse_args, DrRLLoss, PWTSLoss
import sys

def _current_loss_name():
    try:
        return parse_args().loss
    except SystemExit:
        return 'unknown'


def handle_interrupt(sig, frame):
    sys.stdout.write("\n\nCtrl+C detected. What do you want to do?\n")
    sys.stdout.write("  [r] Resume later (save progress and exit)\n")
    sys.stdout.write("  [x] Reset everything and exit\n")
    sys.stdout.flush()
    
    choice = sys.stdin.readline().strip().lower()
    
    if choice == 'r':
        sys.stdout.write("Progress saved. Rerun the script to resume.\n")
        sys.stdout.flush()
        os._exit(0)
    elif choice == 'x':
        csv_path = f'../Models/sasrec_grid_search_{_current_loss_name()}_results.csv'
        if os.path.exists(csv_path):
            os.remove(csv_path)
            print(f"Progress reset ({csv_path} deleted). Rerun the script to start fresh.")
        else:
            print(f"Nothing to reset - {csv_path} does not exist.")
        exit(0)
    else:
        sys.stdout.write("Invalid choice, resuming run...\n")
        sys.stdout.flush()

##Dataset

class SessionDataset(Dataset):
    def __init__(self, parquet_path, features, target):
        df = pd.read_parquet(parquet_path)
        df = df.dropna(subset=features)
        self.sessions = []
        self.labels = []
        self.song_pos = []
        self.ms_played = []
        self.track_length = []
        self.historical_skip_rate = []
        # kept so evaluation can key per-session scores by identity rather than
        # position - different scripts iterate sessions in different orders, and
        # paired significance tests need the rows to line up
        self.session_ids = []

        for session_id, session in df.groupby("session_id"):
            session = session.sort_values("ts")
            self.session_ids.append(session_id)
            x = torch.tensor(session[features].values, dtype=torch.float32)
            y = torch.tensor(session[target].values, dtype=torch.float32)
            self.sessions.append(x)
            self.labels.append(y)
            self.song_pos.append(torch.tensor(session['song_pos'].values, dtype=torch.float32))
            self.ms_played.append(torch.tensor(session['ms_played'].values, dtype=torch.float32))
            self.track_length.append(torch.tensor(session['track_length'].values, dtype=torch.float32))
            self.historical_skip_rate.append(torch.tensor(session['historical_skip_rate'].values, dtype=torch.float32))

    def __len__(self):
        return len(self.sessions)

    def __getitem__(self, idx):
        return (self.sessions[idx], self.labels[idx], self.song_pos[idx],
                self.ms_played[idx], self.track_length[idx],
                self.historical_skip_rate[idx])


def collate_fn(batch):
    sessions, labels, song_pos, ms_played, track_length, historical_skip_rate = zip(*batch)
    sessions_padded = nn.utils.rnn.pad_sequence(sessions, batch_first=True)
    labels_padded = nn.utils.rnn.pad_sequence(labels, batch_first=True)
    song_pos_padded = nn.utils.rnn.pad_sequence(song_pos, batch_first=True)
    ms_played_padded = nn.utils.rnn.pad_sequence(ms_played, batch_first=True)
    track_length_padded = nn.utils.rnn.pad_sequence(track_length, batch_first=True)
    historical_skip_rate_padded = nn.utils.rnn.pad_sequence(historical_skip_rate, batch_first=True)
    lengths = torch.tensor([len(s) for s in sessions])
    return sessions_padded, labels_padded,song_pos_padded, ms_played_padded, track_length_padded,historical_skip_rate_padded, lengths

class LengthBucketSampler(Sampler):
    def __init__(self, lengths, batch_size, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        order = np.argsort(np.asarray(lengths), kind='stable')
        self.batches = [order[i:i + batch_size].tolist()
                        for i in range(0, len(order), batch_size)]

    def __iter__(self):
        batches = list(self.batches)
        if self.shuffle:
            random.shuffle(batches)
        return iter(batches)

    def __len__(self):
        return len(self.batches)


##Model

class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()
        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)
        return outputs

MIN_CONTEXT = 3
MIN_WINDOW_LENGTH = 5

def build_attn_mask(seq_len, boundaries, num_heads, device):
    """True = blocked. attn_mask[i, j] blocks query position i from attending key j."""
    idx    = torch.arange(seq_len, device=device)
    causal = idx.unsqueeze(1) < idx.unsqueeze(0)    
    is_q   = idx.unsqueeze(0) >= boundaries.unsqueeze(1).to(device)      
    qq     = is_q.unsqueeze(2) & is_q.unsqueeze(1)            
    eye    = torch.eye(seq_len, dtype=torch.bool, device=device)
    mask   = causal.unsqueeze(0) | (qq & ~eye)               
    return mask.repeat_interleave(num_heads, dim=0)           

class SkipLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.feature_proj = nn.Linear(input_size, hidden_size)
        self.status_emb   = nn.Embedding(3, hidden_size)
        nn.init.zeros_(self.status_emb.weight[2])
        self.emb_dropout = nn.Dropout(p=dropout)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.fc       = nn.Linear(hidden_size, 1)

    def forward(self, x, lengths, status, boundaries): 
        h = self.emb_dropout(self.feature_proj(x) + self.status_emb(status))
        packed = nn.utils.rnn.pack_padded_sequence(h, lengths.cpu(),
                                                   batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True,
                                                  total_length=x.shape[1])
        return self.fc(out).squeeze(-1)

class SASRec(torch.nn.Module):

    def __init__(self, feature_no, args):
        super(SASRec, self).__init__()
        self.feature_no = feature_no
        self.dev = args.device
        self.norm_first = args.norm_first
        self.num_heads = args.num_heads
        self.feature_proj = torch.nn.Linear(self.feature_no, args.hidden_units)
        self.pos_emb = torch.nn.Embedding(args.maxlen + 1, args.hidden_units, padding_idx=0)
        self.status_emb = nn.Embedding(3, args.hidden_units)
        nn.init.zeros_(self.status_emb.weight[2])  
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.attention_layernorms = torch.nn.ModuleList()
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()
        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
        for _ in range(args.num_blocks):
            self.attention_layernorms.append(torch.nn.LayerNorm(args.hidden_units, eps=1e-8))
            self.attention_layers.append(torch.nn.MultiheadAttention(args.hidden_units, args.num_heads, args.dropout_rate))
            self.forward_layernorms.append(torch.nn.LayerNorm(args.hidden_units, eps=1e-8))
            self.forward_layers.append(PointWiseFeedForward(args.hidden_units, args.dropout_rate))
        self.output_layer = torch.nn.Linear(args.hidden_units, 1)

    def seq2feats(self, x, lengths, status, boundaries):
        N, seq_len, _ = x.shape
        poss = torch.arange(1, seq_len + 1, device=self.dev).unsqueeze(0).expand(N, -1).clone()
        poss = poss.clamp(max=self.pos_emb.num_embeddings - 1)
        for i, l in enumerate(lengths):
            poss[i, l:] = 0
        seqs = self.feature_proj(x) + self.status_emb(status) + self.pos_emb(poss)
        seqs = self.emb_dropout(seqs)
        tl = seq_len
        attn_mask = build_attn_mask(tl, boundaries, self.num_heads, self.dev)
        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            if self.norm_first:
                x_ = self.attention_layernorms[i](seqs)
                mha_out, _ = self.attention_layers[i](x_, x_, x_, attn_mask=attn_mask)
                seqs = seqs + mha_out
                seqs = torch.transpose(seqs, 0, 1)
                seqs = seqs + self.forward_layers[i](self.forward_layernorms[i](seqs))
            else:
                mha_out, _ = self.attention_layers[i](seqs, seqs, seqs, attn_mask=attn_mask)
                seqs = self.attention_layernorms[i](seqs + mha_out)
                seqs = torch.transpose(seqs, 0, 1)
                seqs = self.forward_layernorms[i](seqs + self.forward_layers[i](seqs))
        return self.last_layernorm(seqs)

    def forward(self, x, lengths, status, boundaries):
        feats = self.seq2feats(x, lengths, status, boundaries)
        return self.output_layer(feats).squeeze(-1)

def log_test(row, log_path='../Models/test_log.csv'):
    """Append one result row (a dict) to the test log, creating it if absent."""
    entry = pd.DataFrame([{**row, 'timestamp': pd.Timestamp.now()}])
    if os.path.exists(log_path):
        entry = pd.concat([pd.read_csv(log_path), entry], ignore_index=True)
    entry.to_csv(log_path, index=False)

##Training

def train_epoch(model, loader, optimizer, criterion, device, loss_name='bce'):
    model.train()
    total_loss = 0
    # disable when not a terminal: tqdm's carriage returns turn a SLURM .out file
    # into a single unreadable line
    progress = tqdm(loader, desc="Training", leave=False, disable=not sys.stderr.isatty())
    for sessions, labels, song_pos, ms_played, track_length, historical_skip_rate, lengths in progress:
        sessions, labels = sessions.to(device), labels.to(device)
        if loss_name == 'pwts':
            lengths_expanded = torch.zeros_like(labels)
            for i, length in enumerate(lengths):
                lengths_expanded[i, :length] = length
            song_pos = song_pos.to(device)
            ms_played = ms_played.to(device)
            track_length = track_length.to(device)
            lengths_expanded = lengths_expanded.to(device)
            historical_skip_rate = historical_skip_rate.to(device)
        status = torch.full_like(labels, 2, dtype=torch.long)
        boundaries = torch.empty(len(lengths), dtype=torch.long)
        for i, L in enumerate(lengths):
            L = int(L)
            b = torch.randint(MIN_CONTEXT, max(int(L), MIN_CONTEXT + 1), (1,)).item()
            b=min(b,L)
            boundaries[i] = b
            status[i, :b] = labels[i, :b].long()  

        optimizer.zero_grad()
        preds = model(sessions, lengths, status, boundaries)
        mask = torch.zeros_like(labels, dtype=torch.bool)
        for i, length in enumerate(lengths):
            mask[i, boundaries[i]:int(length)] = True  

        if loss_name == "pwts":
            loss = criterion(preds[mask], labels[mask], song_pos[mask], lengths_expanded[mask], ms_played[mask], track_length[mask], historical_skip_rate[mask])
        elif loss_name == "drrl":
            criterion.update_beta(preds[mask], labels[mask])
            loss = criterion(preds[mask], labels[mask])
        else:
            loss = criterion(preds[mask], labels[mask])

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        progress.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / len(loader)


def valid_splits(y, length, min_context=MIN_CONTEXT, min_skips=1, min_window=MIN_WINDOW_LENGTH):
    """Every context length at which this session can legitimately be scored."""
    return [c for c in range(min_context, length - min_window + 1)
            if y[:c].sum() >= min_skips and len(np.unique(y[c:length])) >= 2]

POINTS_PER_SESSION = 5
NDCG_K =5

def fast_auc(y, p):
    """AUC via the Mann-Whitney statistic. Identical to roc_auc_score to machine
    precision, ~14x faster - it skips per-call input validation and never builds
    the ROC curve. rankdata handles tied predictions by averaging ranks, which is
    what roc_auc_score does too."""
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.nan
    r = rankdata(p)
    return (r[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def ndcg_at_k(y, p, k=NDCG_K):
    """Queue ordered ascending by predicted skip probability; a track NOT skipped is relevant."""
    order = np.argsort(p, kind='stable')
    rel   = 1.0 - y[order]
    disc  = 1.0 / np.log2(np.arange(2, len(rel) + 2))
    dcg   = float((rel[:k] * disc[:len(rel[:k])]).sum())
    ideal = np.sort(1.0 - y)[::-1][:k]
    idcg  = float((ideal * disc[:len(ideal)]).sum())
    return dcg / idcg if idcg > 0 else 0.0

def pick(cands, points=POINTS_PER_SESSION):
    """Evenly spaced subset - deterministic, so there is no seed and no run-to-run variation."""
    if len(cands) <= points:
        return cands
    return [cands[k] for k in np.linspace(0, len(cands) - 1, points).astype(int)]

def rank_first_skip(y, p):
    """How many tracks play before the listener hits a skip, under the model's
    ordering. Rank of the first skipped track, so higher is better."""
    order = np.argsort(p, kind='stable')
    skipped = np.flatnonzero(y[order])
    # valid_splits guarantees both classes in the window, so this should not fire
    return float(skipped[0] + 1) if skipped.size else float(len(y) + 1)


def precision_at_k(y, p, k):
    """Fraction of the top-k reordered queue that the listener does not skip.
    The most directly interpretable metric here: 'of the next k songs this system
    would play, how many are kept?'"""
    order = np.argsort(p, kind='stable')[:k]
    return float((1.0 - y[order]).mean()) if order.size else np.nan


def evaluate_sequential(model, loader, device, min_context=MIN_CONTEXT, min_skips_in_context=1,
                        min_window_length=MIN_WINDOW_LENGTH, points=POINTS_PER_SESSION,
                        chunk=32):
    model.eval()
    session_aucs = []
    session_ndcgs = []
    with torch.no_grad():
        for sessions, labels, song_pos, ms_played, track_length, hsr, lengths in loader:
            B = sessions.shape[0]

            # 1. enumerate split points; a session is dropped only if NONE are valid
            rows = []
            for i in range(B):
                length = int(lengths[i])
                y = labels[i, :length].numpy()
                for c in pick(valid_splits(y, length, min_context,
                                           min_skips_in_context, min_window_length), points):
                    rows.append((i, c))
            if not rows:
                continue

            # 2. expand: one row per (session, split point), each with its own status + boundary
            idx    = torch.tensor([i for i, _ in rows])
            bounds = torch.tensor([c for _, c in rows], dtype=torch.long)
            exp_labels = labels[idx]
            status = torch.full(exp_labels.shape, 2, dtype=torch.long)
            for r, (_, c) in enumerate(rows):
                status[r, :c] = exp_labels[r, :c].long()

            # 3. forward in chunks - the attention mask is (rows x heads x L x L), so an
            #    un-chunked expanded batch can exhaust memory on long sessions
            preds = []
            for s in range(0, len(rows), chunk):
                sl = slice(s, s + chunk)
                p = model(sessions[idx[sl]].to(device), lengths[idx[sl]],
                          status[sl].to(device), bounds[sl])
                preds.append(p.cpu())
            preds = torch.cat(preds, dim=0)

            # 4. score each window, then average WITHIN session before averaging across sessions
            g_auc, g_ndcg = {}, {}
            for r, (i, c) in enumerate(rows):
                length = int(lengths[i])
                y = labels[i, c:length].numpy()
                p = preds[r, c:length].numpy()
                g_auc.setdefault(i, []).append(fast_auc(y, p))
                g_ndcg.setdefault(i, []).append(ndcg_at_k(y, p))
            session_aucs.extend(np.mean(v) for v in g_auc.values())
            session_ndcgs.extend(np.mean(v) for v in g_ndcg.values())

    return np.mean(session_aucs), np.mean(session_ndcgs), len(session_aucs)

def evaluate_sequential_tuning(model, loader, device, min_context=MIN_CONTEXT, min_skips_in_context=1,
                        min_window_length=MIN_WINDOW_LENGTH, points=POINTS_PER_SESSION,
                        chunk=32):
    model.eval()
    session_ndcgs = []
    with torch.no_grad():
        for sessions, labels, song_pos, ms_played, track_length, hsr, lengths in loader:
            B = sessions.shape[0]

            # 1. enumerate split points; a session is dropped only if NONE are valid
            rows = []
            for i in range(B):
                length = int(lengths[i])
                y = labels[i, :length].numpy()
                for c in pick(valid_splits(y, length, min_context,
                                           min_skips_in_context, min_window_length), points):
                    rows.append((i, c))
            if not rows:
                continue

            # 2. expand: one row per (session, split point), each with its own status + boundary
            idx    = torch.tensor([i for i, _ in rows])
            bounds = torch.tensor([c for _, c in rows], dtype=torch.long)
            exp_labels = labels[idx]
            status = torch.full(exp_labels.shape, 2, dtype=torch.long)
            for r, (_, c) in enumerate(rows):
                status[r, :c] = exp_labels[r, :c].long()

            # 3. forward in chunks - the attention mask is (rows x heads x L x L), so an
            #    un-chunked expanded batch can exhaust memory on long sessions
            preds = []
            for s in range(0, len(rows), chunk):
                sl = slice(s, s + chunk)
                p = model(sessions[idx[sl]].to(device), lengths[idx[sl]],
                          status[sl].to(device), bounds[sl])
                preds.append(p.cpu())
            preds = torch.cat(preds, dim=0)

            # 4. score each window, then average WITHIN session before averaging across sessions
            g_ndcg = {}
            for r, (i, c) in enumerate(rows):
                length = int(lengths[i])
                y = labels[i, c:length].numpy()
                p = preds[r, c:length].numpy()
                g_ndcg.setdefault(i, []).append(ndcg_at_k(y, p))
            session_ndcgs.extend(np.mean(v) for v in g_ndcg.values())

    return np.mean(session_ndcgs)

