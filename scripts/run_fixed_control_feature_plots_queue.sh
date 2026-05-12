#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-$ROOT/logs/fixed_control_feature_plots_$RUN_ID}"
mkdir -p "$LOG_ROOT"

CONTROLS="${CONTROLS:-fixed_freq fixed_type}"
FEATURES="${FEATURES:-pos length index arity head_dist relation_type}"
MODELS="${MODELS:-bert gpt2}"
METRICS="${METRICS:-iso lpca lpca99 gride corrint fishers mom rand spect tle twonn}"

MIN_CLASS_COUNT="${MIN_CLASS_COUNT:-1000}"
MIN_TYPE_COUNT="${MIN_TYPE_COUNT:-1000}"
FAST_BOOTSTRAP="${FAST_BOOTSTRAP:-20}"
HEAVY_BOOTSTRAP="${HEAVY_BOOTSTRAP:-8}"
FAST_SAMPLE_CAP="${FAST_SAMPLE_CAP:-30000}"
HEAVY_SAMPLE_CAP="${HEAVY_SAMPLE_CAP:-3000}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MEM_LIMIT_GB="${MEM_LIMIT_GB:-24}"
CHECK_INTERVAL_SEC="${CHECK_INTERVAL_SEC:-10}"
STOP_ON_FAIL="${STOP_ON_FAIL:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
NO_TITLE="${NO_TITLE:-0}"
UNTRAINED="${UNTRAINED:-0}"
OUT_ROOT="${OUT_ROOT:-}"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-2}"

if [[ "${ALLOW_DOWNLOAD:-0}" != "1" ]]; then
  export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
  export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
fi

MEM_LIMIT_KB=$((MEM_LIMIT_GB * 1024 * 1024))

read -r -a METRIC_ARGS <<< "$METRICS"
DOWNLOAD_ARGS=()
if [[ "${ALLOW_DOWNLOAD:-0}" == "1" ]]; then
  DOWNLOAD_ARGS=(--allow-download)
fi
SKIP_ARGS=()
if [[ "$SKIP_EXISTING" == "1" ]]; then
  SKIP_ARGS=(--skip-existing)
fi
TITLE_ARGS=()
if [[ "$NO_TITLE" == "1" ]]; then
  TITLE_ARGS=(--no-title)
fi
UNTRAINED_ARGS=()
if [[ "$UNTRAINED" == "1" ]]; then
  UNTRAINED_ARGS=(--untrained)
fi
OUT_ROOT_ARGS=()
if [[ -n "$OUT_ROOT" ]]; then
  OUT_ROOT_ARGS=(--out-root "$OUT_ROOT")
fi

log_main="$LOG_ROOT/queue.log"
mem_log="$LOG_ROOT/memory.tsv"
printf "timestamp\tcontrol\tfeature\tmodel\tpid\trss_kb\trss_gb\n" > "$mem_log"

echo "run_id=$RUN_ID" | tee -a "$log_main"
echo "log_root=$LOG_ROOT" | tee -a "$log_main"
echo "controls=$CONTROLS" | tee -a "$log_main"
echo "features=$FEATURES" | tee -a "$log_main"
echo "models=$MODELS" | tee -a "$log_main"
echo "metrics=$METRICS" | tee -a "$log_main"
echo "min_class_count=$MIN_CLASS_COUNT min_type_count=$MIN_TYPE_COUNT" | tee -a "$log_main"
echo "skip_existing=$SKIP_EXISTING no_title=$NO_TITLE untrained=$UNTRAINED out_root=${OUT_ROOT:-default} mem_limit_gb=$MEM_LIMIT_GB" | tee -a "$log_main"

word_rep_mode_for_model() {
  case "$1" in
    bert) printf "first" ;;
    gpt2) printf "last" ;;
    *) printf "mean" ;;
  esac
}

run_guarded() {
  local control="$1"
  local feature="$2"
  local model="$3"
  local word_rep
  local job_log
  word_rep="$(word_rep_mode_for_model "$model")"
  job_log="$LOG_ROOT/${control}_${feature}_${model}.log"

  echo "[$(date --iso-8601=seconds)] start control=$control feature=$feature model=$model word_rep=$word_rep log=$job_log" | tee -a "$log_main"

  python3 scripts/recompute_fixed_control_feature_plots.py \
    --control "$control" \
    --feature "$feature" \
    --model "$model" \
    --word-rep-mode "$word_rep" \
    --output-rep-dir first_last \
    --batch-size "$BATCH_SIZE" \
    --min-class-count "$MIN_CLASS_COUNT" \
    --min-type-count "$MIN_TYPE_COUNT" \
    --fast-bootstrap "$FAST_BOOTSTRAP" \
    --heavy-bootstrap "$HEAVY_BOOTSTRAP" \
    --fast-sample-cap "$FAST_SAMPLE_CAP" \
    --heavy-sample-cap "$HEAVY_SAMPLE_CAP" \
    --metrics "${METRIC_ARGS[@]}" \
    "${SKIP_ARGS[@]}" \
    "${TITLE_ARGS[@]}" \
    "${UNTRAINED_ARGS[@]}" \
    "${OUT_ROOT_ARGS[@]}" \
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
          -v control="$control" \
          -v feature="$feature" \
          -v model="$model" \
          -v pid="$pid" \
          -v rss="$rss_kb" \
          'BEGIN { printf "%s\t%s\t%s\t%s\t%s\t%d\t%.2f\n", ts, control, feature, model, pid, rss, rss/1024/1024 }' \
          >> "$mem_log"
      if (( rss_kb > MEM_LIMIT_KB )); then
        echo "[$(date --iso-8601=seconds)] memory guard stopping pid=$pid control=$control feature=$feature model=$model rss_kb=$rss_kb" | tee -a "$log_main" "$job_log"
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
  echo "[$(date --iso-8601=seconds)] end control=$control feature=$feature model=$model status=$status" | tee -a "$log_main"
  return "$status"
}

overall=0
for control in $CONTROLS; do
  for feature in $FEATURES; do
    for model in $MODELS; do
      run_guarded "$control" "$feature" "$model"
      status=$?
      if (( status != 0 )); then
        overall=$status
        echo "job failed control=$control feature=$feature model=$model status=$status" | tee -a "$log_main"
        if [[ "$STOP_ON_FAIL" == "1" ]]; then
          exit "$overall"
        fi
      fi
    done
  done
done

echo "[$(date --iso-8601=seconds)] queue complete status=$overall" | tee -a "$log_main"
exit "$overall"
