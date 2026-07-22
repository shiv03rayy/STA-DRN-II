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
  # All f32/GroupNorm (capacity is settled) and all on multistep step decay (the schedule that reached
  # 9.24 and beat cosine's 9.97). Tonight's question: can the levers that helped under cosine break 9.24
  # once they run on the better schedule? Explicit TAGs keep the runs self-documenting and collision-free.

  # 1. Reproduce the 9.24 champion: step decay, sigma 0, lr 1e-4. Confirms 9.24 is real (single seed so far)
  #    AND is the control that runs 2-4 are measured against.
  "TAG=f32_multistep_baseline LR_SCHEDULE=multistep"
  # 2. Best lever (label smoothing) on the best schedule. Top candidate to beat 9.24.
  "TAG=f32_multistep_sigma2 LR_SCHEDULE=multistep SIGMA=2"
  # 3. Second lever (higher base lr) on the best schedule.
  "TAG=f32_multistep_lr3e-4 LR_SCHEDULE=multistep LR=3e-4"
  # 4. Both levers stacked. The swing for the fences.
  "TAG=f32_multistep_sigma2_lr3e-4 LR_SCHEDULE=multistep SIGMA=2 LR=3e-4"
  # 5. Regularization probe: AdamW decoupled weight decay on the step baseline.
  "TAG=f32_multistep_adamw_wd1e-4 LR_SCHEDULE=multistep OPTIMIZER=AdamW WEIGHT_DECAY=1e-4"
  # 6. Aggressive schedule: drop lr at 15/30 instead of 20/40. The 9.24 landed exactly when lr hit 1e-6 at
  #    epoch 40, so reaching the floor sooner may reach the minimum sooner/lower.
  "TAG=f32_multistep_ms15-30 LR_SCHEDULE=multistep MILESTONES=15,30"
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