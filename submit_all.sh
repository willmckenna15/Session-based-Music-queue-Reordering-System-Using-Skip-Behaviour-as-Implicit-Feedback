
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

DRY=0
FILTER=""
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY=1 ;;
        bce|drrl|pwts) FILTER="$arg" ;;
        *) echo "unknown argument: $arg" >&2; exit 1 ;;
    esac
done

submit () {
    local script=$1 loss=$2 last=$3 model=$4
    [ -n "$FILTER" ] && [ "$FILTER" != "$loss" ] && return 0

    if [ "$DRY" = 1 ]; then
        printf '  would submit: %-16s %-5s --array=0-%s  (%s tasks)\n' \
               "$script" "$loss" "$last" "$((last + 1))"
        return 0
    fi

    local jid
    jid=$(sbatch --parsable --array=0-"$last" "$script" "$loss")
    printf '  %-16s %-5s array 0-%-3s  job %s\n' "$script" "$loss" "$last" "$jid"


    sbatch --parsable \
           --dependency=afterany:"$jid" \
           --job-name="merge_${model}_${loss}" \
           --output="logs/merge_${model}_${loss}.out" \
           --time=00:15:00 --mem=8G --cpus-per-task=1 \
           --wrap="source ~/venv/bin/activate && cd '$PWD/Model Scripts' && python merge_results.py ${model} ${loss}" \
           > /dev/null
}

echo "Submitting tuning arms:"
submit run_lstm.sh   bce   53  LSTM
submit run_lstm.sh   pwts  53  LSTM
submit run_lstm.sh   drrl 161  LSTM
submit run_sasrec.sh bce  161  sasrec
submit run_sasrec.sh pwts 161  sasrec
submit run_sasrec.sh drrl 485  sasrec

if [ "$DRY" = 1 ]; then
    echo
    echo "Dry run - nothing submitted."
else
    echo
    echo "Watch:   squeue -u \$USER"
    echo "Results: Models/{LSTM,sasrec}_grid_search_<loss>_results.csv"
    echo "Merges run automatically as each array finishes."
fi
