#!/usr/bin/env bash
# Correctness and performance profiling on a discrete NVIDIA GPU.
#
# Run this first on rented hardware. It gates on correctness before it reports
# a single timing, because a fast wrong kernel is worth nothing, and it locks
# clocks where it can so two runs are comparable.
#
#   scripts/profile_gpu.sh [output-dir]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT/build}"
OUT="${1:-$ROOT/profiles/$(hostname -s)-$(date +%Y%m%d-%H%M%S)}"
BENCH="$BUILD_DIR/bench/bench_sam"
DETECT="$BUILD_DIR/src/hsi_detect"

[[ -x "$BENCH" ]] || { echo "build first: scripts/build.sh" >&2; exit 1; }
mkdir -p "$OUT"
echo "writing to $OUT"

{
  echo "=== host ==="; uname -a; date -u
  echo; echo "=== gpu ==="; nvidia-smi || true
  echo; echo "=== nvcc ==="; nvcc --version || true
} > "$OUT/environment.txt" 2>&1

# Deliberately NOT locking the SM clock to its maximum.
#
# On a card whose power limit sits below its board maximum - most rental
# hardware, and every Jetson - the maximum SM clock is not sustainable. Pinning
# it just means every measurement drifts downwards as the card throttles into
# its real operating point, and the first variant timed looks faster than the
# last for no reason to do with the code.
#
# Instead bench_sam warms up by wall-clock time until the clock has settled,
# reports the median of several passes with the spread, and prints the SM clock
# beside each result so a throttled number says so.
nvidia-smi -pm 1 >/dev/null 2>&1 && echo "persistence mode on" \
  || echo "note: could not enable persistence mode (needs root)"

POWER=$(nvidia-smi --query-gpu=power.limit,power.max_limit --format=csv,noheader 2>/dev/null)
echo "power limit: $POWER"
echo "$POWER" | awk -F'[ ,]+' '$1 < $3 * 0.9 {
  print "  this card is capped below its board maximum, so expect the SM clock"
  print "  to fall under sustained load - watch the SM MHz column" }' 

# ---- 1. correctness, before anything is timed ------------------------------
echo
echo "== correctness against the CPU reference =="
"$BENCH" --width=128 --height=96 --bands=24 --targets=4 --check-only \
  | tee "$OUT/correctness_small.txt"
"$BENCH" --width=696 --height=520 --bands=113 --targets=8 --check-only \
  | tee "$OUT/correctness_hyperblood_shape.txt"

# ---- 2. kernel throughput --------------------------------------------------
echo
echo "== kernel throughput =="
"$BENCH" --width=512 --height=512 --bands=128 --targets=4 \
  | tee "$OUT/kernels_512x512x128_t4.txt"

# Band count is the inner-loop trip count and the whole byte budget, so it is
# the axis that decides whether this is real-time at a given sensor.
echo
echo "== sweep: bands =="
for BANDS in 16 32 64 113 128 224; do
  "$BENCH" --width=512 --height=512 --bands="$BANDS" --targets=4 --iterations=100
done | tee "$OUT/sweep_bands.txt"

# Target count is what separates the variants: the baseline re-reads the cube
# per target, the vectorised kernels amortise up to eight over one read.
echo
echo "== sweep: targets =="
for TARGETS in 1 2 4 8 16; do
  "$BENCH" --width=512 --height=512 --bands=128 --targets="$TARGETS" --iterations=100
done | tee "$OUT/sweep_targets.txt"

# ---- 3. end-to-end pipeline ------------------------------------------------
echo
echo "== end-to-end pipeline =="
for VARIANT in baseline optimized half; do
  echo "--- variant=$VARIANT ---"
  "$DETECT" --variant="$VARIANT" --memory=copy --width=512 --height=512 \
            --bands=128 --targets=4 --frames=200 --prefill \
            --csv="$OUT/pipeline_$VARIANT.csv"
done | tee "$OUT/pipeline.txt"

echo
echo "== stream count =="
for STREAMS in 1 2 3 4 6; do
  echo "--- streams=$STREAMS ---"
  "$DETECT" --streams="$STREAMS" --frames=200 --prefill --quiet
done | tee "$OUT/sweep_streams.txt"

# ---- 4. nsight ------------------------------------------------------------
if command -v nsys >/dev/null 2>&1; then
  echo
  echo "== nsys timeline =="
  nsys profile --force-overwrite=true -o "$OUT/timeline" \
       --trace=cuda,nvtx,osrt --cuda-memory-usage=true \
       "$DETECT" --frames=60 --warmup=10 --prefill --quiet \
       > "$OUT/nsys.log" 2>&1 || echo "nsys failed; see $OUT/nsys.log"
  nsys stats "$OUT/timeline.nsys-rep" > "$OUT/nsys_stats.txt" 2>&1 || true
fi

if command -v ncu >/dev/null 2>&1; then
  echo
  echo "== ncu kernel counters =="
  # A targeted metric list rather than --set full: these five answer the only
  # question that matters for a bandwidth-bound kernel - how close to peak is
  # it, and if it is not close, is that coalescing or occupancy.
  ncu --force-overwrite --target-processes all \
      --kernel-name-base function --launch-count 3 \
      --metrics \
dram__bytes.sum.per_second,\
gpu__time_duration.sum,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum \
      -o "$OUT/kernels" \
      "$BENCH" --width=512 --height=512 --bands=128 --targets=4 \
               --iterations=3 --warmup-ms=0 --repeats=1 \
      > "$OUT/ncu.log" 2>&1 || echo "ncu failed (often needs --privileged or CAP_SYS_ADMIN); see $OUT/ncu.log"
fi

nvidia-smi -rgc >/dev/null 2>&1 || true
echo
echo "done. results in $OUT"
