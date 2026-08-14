#!/bin/bash
#SBATCH --job-name=sasrec
#SBATCH --partition=gpu
#SBATCH --output=logs/sasrec_%A_%a.out
#SBATCH --error=logs/sasrec_%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-161

# Array size by loss - drrl adds gamma(3) as a third axis:
#   bce / pwts : 3x3x3x3x2 = 162   sbatch --array=0-161 run_sasrec.sh bce
#   drrl       : 162x3     = 486   sbatch --array=0-485 run_sasrec.sh drrl
#
# Partition 'gpu' spans all 28 GPUs. Worst-case attention memory is
# batch(64) x heads(16) x 807^2 x ~9 bytes = 6GB, which fits a 12GB GTX 1080.
# Only 0.8% of sessions exceed 200 tracks, so typical batches are under 0.5GB.

set -euo pipefail

LOSS="${1:-bce}"

source ~/venv/bin/activate
cd "$SLURM_SUBMIT_DIR/Model Scripts"

echo "task ${SLURM_ARRAY_TASK_ID} | loss ${LOSS} | node $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python3 -u SASRec_tuning.py --loss "$LOSS"
