#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-$ROOT/logs/mean_feature_recompute_$RUN_ID}"
mkdir -p "$LOG_ROOT"

FEATURES="${FEATURES:-length index relation_type arity}"
MODELS="${MODELS:-bert gpt2}"
METRICS="${METRICS:-iso lpca99 gride corrint fishers mom rand spect tle twonn}"

MAX_PER_CLASS="${MAX_PER_CLASS:-30000}"
FAST_BOOTSTRAP="${FAST_BOOTSTRAP:-20}"
HEAVY_BOOTSTRAP="${HEAVY_BOOTSTRAP:-8}"
FAST_SAMPLE_CAP="${FAST_SAMPLE_CAP:-30000}"
HEAVY_SAMPLE_CAP="${HEAVY_SAMPLE_CAP:-3000}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MEM_LIMIT_GB="${MEM_LIMIT_GB:-24}"
CHECK_INTERVAL_SEC="${CHECK_INTERVAL_SEC:-10}"
STOP_ON_FAIL="${STOP_ON_FAIL:-1}"
OUT_ROOT="${OUT_ROOT:-$ROOT/plots_extra/metrics/features}"
SKIP_DONE="${SKIP_DONE:-1}"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-2}"

if [[ "${ALLOW_DOWNLOAD:-0}" != "1" ]]; then
  export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
fi

MEM_LIMIT_KB=$((MEM_LIMIT_GB * 1024 * 1024))

read -r -a METRIC_ARGS <<< "$METRICS"
DOWNLOAD_ARGS=()
if [[ "${ALLOW_DOWNLOAD:-0}" == "1" ]]; then
  DOWNLOAD_ARGS=(--allow-download)
fi

log_main="$LOG_ROOT/queue.log"
mem_log="$LOG_ROOT/memory.tsv"
printf "timestamp\tfeature\tmodel\tpid\trss_kb\trss_gb\n" > "$mem_log"

echo "run_id=$RUN_ID" | tee -a "$log_main"
echo "log_root=$LOG_ROOT" | tee -a "$log_main"
echo "features=$FEATURES" | tee -a "$log_main"
echo "models=$MODELS" | tee -a "$log_main"
echo "metrics=$METRICS" | tee -a "$log_main"
echo "out_root=$OUT_ROOT" | tee -a "$log_main"
echo "skip_done=$SKIP_DONE" | tee -a "$log_main"
echo "mem_limit_gb=$MEM_LIMIT_GB" | tee -a "$log_main"
echo "caps: max_per_class=$MAX_PER_CLASS fast_bootstrap=$FAST_BOOTSTRAP heavy_bootstrap=$HEAVY_BOOTSTRAP fast_sample_cap=$FAST_SAMPLE_CAP heavy_sample_cap=$HEAVY_SAMPLE_CAP" | tee -a "$log_main"

run_guarded() {
  local feature="$1"
  local model="$2"
  local job_log="$LOG_ROOT/${feature}_${model}.log"
  local started
  started="$(date --iso-8601=seconds)"
  echo "[$started] start feature=$feature model=$model log=$job_log" | tee -a "$log_main"

  python3 scripts/recompute_mean_feature_plots.py \
    --feature "$feature" \
    --model "$model" \
    --word-rep-mode mean \
    --batch-size "$BATCH_SIZE" \
    --out-root "$OUT_ROOT" \
    --max-per-class "$MAX_PER_CLASS" \
    --fast-bootstrap "$FAST_BOOTSTRAP" \
    --heavy-bootstrap "$HEAVY_BOOTSTRAP" \
    --fast-sample-cap "$FAST_SAMPLE_CAP" \
    --heavy-sample-cap "$HEAVY_SAMPLE_CAP" \
    --metrics "${METRIC_ARGS[@]}" \
    "${DOWNLOAD_ARGS[@]}" \
    > "$job_log" 2>&1 &

  local pid=$!
  local rss_kb=0
  while kill -0 "$pid" 2>/dev/null; do
    if [[ -r "/proc/$pid/status" ]]; then
      rss_kb="$(awk '/VmRSS:/ {print $2}' "/proc/$pid/status" 2>/dev/null || echo 0)"
      if [[ -z "$rss_kb" ]]; then
        rss_kb=0
      fi
      awk -v ts="$(date --iso-8601=seconds)" \
          -v feature="$feature" \
          -v model="$model" \
          -v pid="$pid" \
          -v rss="$rss_kb" \
          'BEGIN { printf "%s\t%s\t%s\t%s\t%d\t%.2f\n", ts, feature, model, pid, rss, rss/1024/1024 }' \
          >> "$mem_log"
      if (( rss_kb > MEM_LIMIT_KB )); then
        echo "[$(date --iso-8601=seconds)] memory guard stopping pid=$pid feature=$feature model=$model rss_kb=$rss_kb" | tee -a "$log_main" "$job_log"
        kill -TERM "$pid" 2>/dev/null || true
        sleep 20
        kill -KILL "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null
        return 137
      fi
    fi
    sleep "$CHECK_INTERVAL_SEC"
  done

  wait "$pid"
  local status=$?
  echo "[$(date --iso-8601=seconds)] end feature=$feature model=$model status=$status" | tee -a "$log_main"
  return "$status"
}

feature_dir() {
  case "$1" in
    relation_type) printf "relations" ;;
    *) printf "%s" "$1" ;;
  esac
}

feature_suffix() {
  case "$1" in
    relation_type) printf "relation" ;;
    *) printf "%s" "$1" ;;
  esac
}

is_done() {
  local feature="$1"
  local model="$2"
  local dir suffix metric file
  dir="$(feature_dir "$feature")"
  suffix="$(feature_suffix "$feature")"
  for metric in "${METRIC_ARGS[@]}"; do
    file="$OUT_ROOT/$dir/mean/$model/${metric}_${model}_${suffix}.png"
    if [[ ! -s "$file" ]]; then
      return 1
    fi
  done
  return 0
}

overall=0
for feature in $FEATURES; do
  for model in $MODELS; do
    if [[ "$SKIP_DONE" == "1" ]] && is_done "$feature" "$model"; then
      echo "[$(date --iso-8601=seconds)] skip done feature=$feature model=$model" | tee -a "$log_main"
      continue
    fi
    run_guarded "$feature" "$model"
    status=$?
    if (( status != 0 )); then
      overall=$status
      echo "job failed feature=$feature model=$model status=$status" | tee -a "$log_main"
      if [[ "$STOP_ON_FAIL" == "1" ]]; then
        exit "$overall"
      fi
    fi
  done
done

echo "[$(date --iso-8601=seconds)] queue complete status=$overall" | tee -a "$log_main"
exit "$overall"
