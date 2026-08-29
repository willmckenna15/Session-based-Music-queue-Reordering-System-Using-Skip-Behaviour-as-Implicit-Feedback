"""Merge the per-task CSVs from the PWTS ablation into ../Models/pwts_experiments.csv.

Each SLURM array task writes its own file so concurrent tasks cannot clobber each
other. This concatenates them, sorts by validation NDCG, and reports any tasks that
did not produce output.

    python merge_pwts.py
"""

import ast
import glob
import re
import sys

import pandas as pd

OUT = '../Models/pwts_experiments.csv'
PATTERN = '../Models/pwts_experiments_task*.csv'
N_VARIANTS = 15
MIN_EPOCHS = 20

paths = sorted(glob.glob(PATTERN),
               key=lambda p: int(re.search(r'task(\d+)\.csv$', p).group(1)))
if not paths:
    sys.exit(f"No files matching {PATTERN}")

# log_test appends, so a task file re-run without being deleted first holds one row
# per run. Only the last is current.
frames, stale = [], []
for f in paths:
    part = pd.read_csv(f)
    if len(part) > 1:
        stale.append((f, len(part) - 1))
    frames.append(part.tail(1))
df = pd.concat(frames, ignore_index=True)
if stale:
    print(f"Dropped superseded rows from {len(stale)} task file(s): "
          + ", ".join(f"{re.search(r'task(\d+)', f).group(0)} (-{n})" for f, n in stale) + "\n")
df = df.sort_values('val_ndcg_mean', ascending=False).reset_index(drop=True)
df.to_csv(OUT, index=False)

print(f"Merged {len(paths)} task files -> {len(df)} variants")
print(f"Written to {OUT}\n")

# Shortest run across the seeds. Early stopping firing at epoch 7 on one seed drags
# a variant's mean down by more than any of the loss constants move it, so this
# column is needed to read the table honestly.
df['min_epochs'] = df.epochs_run.apply(lambda s: min(ast.literal_eval(s)))

full = df[df.variant == 'full']
threshold = 2 * df.val_ndcg_std.mean()

cols = ['variant', 'changed_from_default', 'val_ndcg_mean', 'val_ndcg_std',
        'val_auc_mean', 'min_epochs', 'epochs_run']
print(df[cols].to_string(index=False))

print(f"\nmean seed sd {df.val_ndcg_std.mean():.4f}  ->  differences below "
      f"~{threshold:.4f} are not resolvable")
if not full.empty:
    print(f"full PWTS = {full.val_ndcg_mean.iloc[0]:.4f}")
    n_out = (abs(df.val_ndcg_mean - full.val_ndcg_mean.iloc[0]) > threshold).sum()
    print(f"{n_out} of {len(df)} variants differ from it by more than that")

if df.min_epochs.min() < MIN_EPOCHS:
    corr = df.min_epochs.corr(df.val_ndcg_mean)
    print(f"\nWARNING: some variants had a seed stop before epoch {MIN_EPOCHS}, and "
          f"min_epochs correlates {corr:+.2f} with NDCG.\n"
          f"The MIN_EPOCHS floor in pwts_ablation.py should make this impossible - if "
          f"it fires, the cluster ran an older copy of the script and the ranking is "
          f"confounded by training length, not the loss constants.")
else:
    print(f"\nAll seeds ran at least {df.min_epochs.min()} epochs - the early-stopping "
          f"artefact that invalidated the first run is not present.")

found = {int(re.search(r'task(\d+)', p).group(1)) for p in paths}
missing = sorted(set(range(N_VARIANTS)) - found)
if missing:
    print(f"\nWARNING: no output from tasks {missing}")
