"""Merge the per-task CSVs a SLURM job array produces into one results file.

Each array task writes ../Models/{sasrec,LSTM}_grid_{loss}_task{N}.csv so that
concurrent tasks never write to the same file. This concatenates them, sorts by
val_ndcg, and writes the combined table plus the best-parameters JSON.

    python merge_results.py sasrec bce
    python merge_results.py LSTM bce
"""

import glob
import os
import re
import sys

import pandas as pd

if len(sys.argv) != 3:
    sys.exit("usage: python merge_results.py <sasrec|LSTM> <bce|drrl|pwts>")

model, loss = sys.argv[1], sys.argv[2]

pattern = f'../Models/{model}_grid_{loss}_task*.csv'
paths = sorted(glob.glob(pattern))
if not paths:
    sys.exit(f"No per-task files matching {pattern}")

df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
df = df.sort_values('val_ndcg', ascending=False).reset_index(drop=True)

out_csv = f'../Models/{model}_grid_search_{loss}_results.csv'
df.to_csv(out_csv, index=False)

print(f"Merged {len(paths)} task files -> {len(df)} configurations")
print(f"Written to {out_csv}\n")
print(df.head(10).to_string(index=False))

best = df.iloc[0].to_dict()
best_json = f'../Models/{model.lower()}_{loss}_best_params.json'
pd.Series(best).to_json(best_json)
print(f"\nBest config: {best}")
print(f"Best params saved to {best_json}")

# Gaps mean array tasks that failed or timed out - worth knowing before you treat
# the merged table as a complete grid
present = {int(re.search(r'task(\d+)\.csv$', p).group(1)) for p in paths}
gaps = sorted(set(range(max(present) + 1)) - present)
if gaps:
    print(f"\nWARNING: {len(gaps)} task files missing between 0 and {max(present)} - "
          f"those array tasks failed or timed out. First few: {gaps[:10]}")
