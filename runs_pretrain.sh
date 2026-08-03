#!/usr/bin/env bash
#
# Pretraining experiment: does initialising from the authors' published checkpoint help?
#
#   tmux new -d -s pre './runs_pretrain.sh --wait'
#   tmux attach -t pre
#
# Design. A 2x2: {pretrained, scratch} x {seed 1, seed 2}. The question is whether
# weights/best.pth as an initialiser beats random init, so the ONLY thing allowed to differ
# between the arms is the initialisation. Everything else is pinned identical: f16 (forced --
# best.pth is an f16 model and shares no tensor shapes with f32), GroupNorm(8), LR 1e-4,
# multistep 15/30.
#
# Two seeds per arm, not one. Finding 1 measured 1.06 MAE of run-to-run variance on a fixed
# config, which is larger than most effects this project has chased. A single run per arm
# produces a number that cannot be interpreted, and reporting one would repeat the exact
# error the dissertation criticises. Two seeds is still thin -- read the arms as overlapping
# or not overlapping, not as a ranking.

set -u
cd "$(dirname "$0")"

PYTHON=.venv/bin/python
LOGDIR=logs
mkdir -p "$LOGDIR"

# LR 1e-4, not the 1e-5 used by the f16_pretrained_seed1 probe. At 1e-5 the milestone-15 decay
# drops the rate to 1e-6 before a randomly-initialised model has learned anything, so the scratch
# arm would lose for want of training rather than for want of a good initialisation -- biasing the
# experiment toward "pretraining helps". 1e-4 on multistep 15/30 is the setting that produced this
# project's best results and is a fair starting point for both arms.
COMMON="FEATURES=16 NORM=groupnorm LR=1e-4 LR_SCHEDULE=multistep MILESTONES=15,30"

QUEUE=(
  # Arm A -- initialised from the authors' published checkpoint, two seeds.
  "$COMMON PRETRAIN_CKPT=weights/best.pth TAG=f16_pre_lr1e-4_seed1"
  "$COMMON PRETRAIN_CKPT=weights/best.pth TAG=f16_pre_lr1e-4_seed2"

  # Arm B -- identical in every respect except that PRETRAIN_CKPT is unset, i.e. random init.
  # This is the control the conclusion actually rests on. Note the existing 10.3258 f16 row in
  # results_log.csv cannot serve as it: that run used LR 1e-4 on *cosine*, so it differs from
  # arm A in schedule as well as initialisation.
  "$COMMON TAG=f16_scratch_lr1e-4_seed1"
  "$COMMON TAG=f16_scratch_lr1e-4_seed2"
)

# The in-flight f16_pretrained_seed1 run (LR 1e-5) is NOT part of this 2x2. It stands on its own
# as a probe of whether a gentler fine-tuning rate does better than 1e-4, and it has no matched
# scratch control, so it cannot support a claim about pretraining by itself.

# --wait: hold off until the GPU has no training process on it, so this can be queued behind
# the seed 1 run that is already going without fighting it for the 8GB.
if [[ "${1:-}" == "--wait" ]]; then
  while pgrep -f "train_update.py" >/dev/null; do
    echo "$(date '+%F %T') waiting for in-flight training to finish..."
    sleep 60
  done
  echo "$(date '+%F %T') GPU clear, starting queue"
fi

echo "=== pretraining experiment, ${#QUEUE[@]} run(s), started $(date '+%F %T') ==="

for i in "${!QUEUE[@]}"; do
  cfg="${QUEUE[$i]}"
  stamp=$(date '+%Y%m%d-%H%M%S')
  log="$LOGDIR/pretrain$((i + 1))-$stamp.log"

  echo ""
  echo "--- [$((i + 1))/${#QUEUE[@]}] $(date '+%F %T')  $cfg"
  echo "--- log: $log"

  # env applies the config for this run only; the queue keeps going if one run dies, so a
  # single failure does not waste the rest of the queue.
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
