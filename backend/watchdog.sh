#!/bin/bash
# Lightweight training watchdog. Every CHECK_INTERVAL it verifies the EfficientNet
# training is alive, progressing, and not thrashing swap. Self-heals:
#   - process dead  -> relaunch (memory-safe)
#   - log hung      -> kill + relaunch
#   - 3 failures    -> escalate to a lighter model (efficientnet_b0)
#   - swap thrash   -> kill + relaunch at smaller batch
# Exits cleanly once training prints its final HEADLINE / exports the model.
# Cost per check: a handful of ps/sysctl/stat calls — negligible vs the GPU work.

REPO="/Users/mitulpatel/Documents/Claude/body language reader/.claude/worktrees/eloquent-shaw-bfbe0a"
LOG="$REPO/logs/train_b4.log"
HLOG="$REPO/logs/health.log"
STATE="$REPO/logs/watchdog.state"   # holds restart count

CHECK_INTERVAL=600     # 10 minutes
LOG_STALE_SECS=360     # alive but log untouched this long => hung
SWAP_THRASH_MB=7000    # swap used above this => renewed thrash
DATA="resources/datasets/affectnet_faces"

ts() { date "+%Y-%m-%d %H:%M:%S"; }
hlog() { echo "$(ts) | $1" >> "$HLOG"; }

get_restarts() { [ -f "$STATE" ] && cat "$STATE" || echo 0; }
set_restarts() { echo "$1" > "$STATE"; }

swap_used_mb() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1; }

train_pid() { pgrep -f "train_cnn2.py" | head -1; }

log_age() {
  [ -f "$LOG" ] || { echo 999999; return; }
  local m; m=$(stat -f %m "$LOG" 2>/dev/null || echo 0)
  echo $(( $(date +%s) - m ))
}

launch() {
  local arch="$1" batch="$2"
  cd "$REPO" || return 1
  # shellcheck disable=SC1091
  source .venv-train/bin/activate
  PYTORCH_ENABLE_MPS_FALLBACK=1 nohup python -u backend/train_cnn2.py \
    --data "$DATA" --arch "$arch" --epochs 22 --batch "$batch" --workers 2 \
    >> "$LOG" 2>&1 &
  hlog "ACTION relaunch arch=$arch batch=$batch pid=$!"
}

heal() {
  # decide arch/batch based on how many times we've already restarted
  local n; n=$(get_restarts); n=$((n + 1)); set_restarts "$n"
  local arch="efficientnet_b4" batch=16
  if [ "$n" -ge 2 ]; then batch=8; fi                            # shrink batch if it keeps failing
  if [ "$n" -ge 4 ]; then arch="efficientnet_b2"; batch=24; fi   # fall back to the proven net
  pkill -9 -f "train_cnn2.py" 2>/dev/null
  pkill -9 -f "torch_shm_manager" 2>/dev/null
  sleep 3
  launch "$arch" "$batch"
}

hlog "watchdog started (interval=${CHECK_INTERVAL}s, pid=$$)"

while true; do
  # 1) training finished successfully?
  if grep -q "HEADLINE: test_acc" "$LOG" 2>/dev/null; then
    hlog "OK training COMPLETE (HEADLINE found) — watchdog exiting"
    exit 0
  fi

  PID=$(train_pid)
  SWAP=$(swap_used_mb)
  AGE=$(log_age)
  BEST=$(grep -Eo "new best [0-9.]+" "$LOG" 2>/dev/null | tail -1 | awk '{print $3}')
  EP=$(grep -E "^epoch " "$LOG" 2>/dev/null | tail -1 | awk '{print $2}')

  if [ -z "$PID" ]; then
    hlog "FAIL process DOWN (best=${BEST:-?} ep=${EP:-?}) — healing"
    heal
  elif [ "$AGE" -gt "$LOG_STALE_SECS" ]; then
    hlog "FAIL log HUNG ${AGE}s (pid=$PID best=${BEST:-?} ep=${EP:-?}) — healing"
    heal
  elif [ "${SWAP:-0}" -gt "$SWAP_THRASH_MB" ]; then
    hlog "FAIL swap THRASH ${SWAP}MB (pid=$PID) — healing at smaller batch"
    pkill -9 -f "train_cnn2.py"; pkill -9 -f "torch_shm_manager"; sleep 3
    launch "efficientnet_b4" 8
  else
    hlog "OK pid=$PID ep=${EP:-1} best=${BEST:-?} swap=${SWAP}MB log_age=${AGE}s"
  fi

  sleep "$CHECK_INTERVAL"
done
