

set -euo pipefail

LOSS="${1:-bce}"

source ~/venv/bin/activate
cd "$SLURM_SUBMIT_DIR/Model Scripts"

echo "task ${SLURM_ARRAY_TASK_ID} | loss ${LOSS} | node $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python3 -u SASRec_tuning.py --loss "$LOSS"
