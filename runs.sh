#!/usr/bin/env bash
#
# Run a queue of training configs back-to-back, unattended.
#
#   ./runs.sh              # run the queue below
#   ./runs.sh --wait       # first wait for any in-flight training to finish, then run
#
# Detach it so it survives logout / a closed terminal:
#
#   tmux new -d -s runs './runs.sh --wait'
#   tmux attach -t runs                       # look in on it later
#
# Each line of QUEUE is the env for one run. Anything not set falls back to the
# default in train_update.py, so a line only names what it changes. TAG is derived
# from the config automatically; set it explicitly only if you want a specific name.

set -u
cd "$(dirname "$0")"

PYTHON=.venv/bin/python
LOGDIR=logs
mkdir -p "$LOGDIR"

QUEUE=(
  # 1. Baseline: f32 under cosine annealing. Identical to the 9.2421 row in results_log.csv
  #    except for the schedule, so it isolates cosine vs MultiStepLR. Runs 2-6 are read
  #    against this, not against the older step-decay rows.
  "LR_T_MAX=100"
  # 2. Capacity control. The f16 row used BatchNorm at batch 5, so the 13.486 -> 9.2421 gap
  #    confounds capacity with the BN->GN fix; this run is f16 under today's setup.
  "FEATURES=16 LR_T_MAX=100"
  # 3. Capacity up. ~6.5GB predicted against 8192MiB -- if it OOMs it fails fast.
  "FEATURES=48 LR_T_MAX=100"
  # 4. Label smoothing (SIGMA feeds the label noise at line ~215). Zero in every run so far.
  "SIGMA=2 LR_T_MAX=100"
  # 5. Base LR. Under step decay 1e-4 only applied for 20 epochs, so it was never really tested.
  "LR=3e-4 LR_T_MAX=100"
  # 6. Width vs temporal context: f64 only fits at 8GB with the clip halved.
  "FEATURES=64 FRAME_LEN=32 LR_T_MAX=100"
)

# --wait: hold off until the GPU has no training process on it, so this can be queued
# behind a run that is already going without fighting it for the 8GB.
if [[ "${1:-}" == "--wait" ]]; then
  while pgrep -f "train_update.py" >/dev/null; do
    echo "$(date '+%F %T') waiting for in-flight training to finish..."
    sleep 60
  done
  echo "$(date '+%F %T') GPU clear, starting queue"
fi

echo "=== queue of ${#QUEUE[@]} run(s) started $(date '+%F %T') ==="

for i in "${!QUEUE[@]}"; do
  cfg="${QUEUE[$i]}"
  stamp=$(date '+%Y%m%d-%H%M%S')
  log="$LOGDIR/run$((i + 1))-$stamp.log"

  echo ""
  echo "--- [$((i + 1))/${#QUEUE[@]}] $(date '+%F %T')  $cfg"
  echo "--- log: $log"

  # env applies the config for this run only; the queue keeps going if one run dies,
  # so a single OOM at the top of the sweep doesn't waste the rest of the night.
  if env $cfg "$PYTHON" train_update.py >"$log" 2>&1; then
    echo "--- ok: $(grep -E 'best val MAE|Run result appended' "$log" | tail -2)"
  else
    echo "--- FAILED (exit $?), see $log:"
    tail -5 "$log" | sed 's/^/      /'
  fi

  # let the GPU drain before the next run claims memory
  sleep 30
done

echo ""
echo "=== queue finished $(date '+%F %T') ==="
echo "results: results_log.csv"