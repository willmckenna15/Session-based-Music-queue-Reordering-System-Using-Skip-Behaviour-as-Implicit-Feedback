#!/bin/bash
#SBATCH --job-name=lstm
#SBATCH --partition=gpu
#SBATCH --output=logs/lstm_%A_%a.out
#SBATCH --error=logs/lstm_%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-53

# Array size by loss - drrl adds gamma(3) as a third axis:
#   bce / pwts : 3x3x3x2   =  54   sbatch --array=0-53  run_lstm.sh bce
#   drrl       : 54x3      = 162   sbatch --array=0-161 run_lstm.sh drrl
#
# Partition 'gpu' spans all 28 GPUs. Peak attention memory is ~6GB worst case,
# so a 12GB GTX 1080 is sufficient - no reason to queue for the a40 partition.

set -euo pipefail

LOSS="${1:-bce}"

source ~/venv/bin/activate
cd "$SLURM_SUBMIT_DIR/Model Scripts"

echo "task ${SLURM_ARRAY_TASK_ID} | loss ${LOSS} | node $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python RNN_tuning.py --loss "$LOSS"
