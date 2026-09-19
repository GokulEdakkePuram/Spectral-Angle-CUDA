#!/usr/bin/env bash
# Correctness and performance profiling on a Jetson AGX Orin.
#
# Differs from the discrete-GPU script in three ways that matter:
#
#   * Power mode and clocks dominate everything. An Orin left on its default
#     mode can sit at a third of its peak, so a number taken without pinning
#     nvpmodel and jetson_clocks says nothing about what the platform can do.
#   * Copy versus zero-copy is the headline experiment, not a footnote. The
#     CPU and GPU share one LPDDR5 pool, so the H2D copy is pure overhead here
#     and removing it should show up directly in the stage breakdown.
#   * Thermals are real. tegrastats runs alongside so a throughput drop can be
#     attributed to throttling rather than to the code.
#
#   scripts/profile_jetson.sh [output-dir]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT/build}"
OUT="${1:-$ROOT/profiles/orin-$(date +%Y%m%d-%H%M%S)}"
BENCH="$BUILD_DIR/bench/bench_sam"
DETECT="$BUILD_DIR/src/hsi_detect"

[[ -x "$BENCH" ]] || { echo "build first: scripts/build.sh 87" >&2; exit 1; }
mkdir -p "$OUT"
echo "writing to $OUT"

{
  echo "=== host ==="; uname -a; date -u
  echo; echo "=== l4t ==="; cat /etc/nv_tegra_release 2>/dev/null
  head -1 /etc/nv_tegra_release 2>/dev/null
  echo; echo "=== power mode ==="; sudo -n nvpmodel -q 2>/dev/null || nvpmodel -q 2>/dev/null || echo "unavailable"
  echo; echo "=== memory ==="; free -h
  echo; echo "=== nvcc ==="; nvcc --version 2>/dev/null || echo "not on PATH"
} > "$OUT/environment.txt" 2>&1

# MAXN plus pinned clocks. Without this the Orin's DVFS will happily report
# whatever frequency it felt like, and the run is not reproducible.
if sudo -n true 2>/dev/null; then
  echo "setting MAXN and pinning clocks"
  sudo nvpmodel -m 0 || true
  sudo jetson_clocks || true
  sudo jetson_clocks --show > "$OUT/clocks.txt" 2>&1 || true
else
  echo "WARNING: no passwordless sudo - clocks are NOT pinned."
  echo "         Run 'sudo nvpmodel -m 0 && sudo jetson_clocks' first, or these"
  echo "         numbers are a lower bound of unknown tightness."
fi

# Thermals and per-rail power alongside the run, so a slow stretch can be
# blamed on throttling rather than on the kernel.
TEGRASTATS_PID=""
if command -v tegrastats >/dev/null 2>&1; then
  tegrastats --interval 500 --logfile "$OUT/tegrastats.log" &
  TEGRASTATS_PID=$!
  trap '[[ -n "$TEGRASTATS_PID" ]] && kill "$TEGRASTATS_PID" 2>/dev/null || true' EXIT
fi

# ---- 1. correctness --------------------------------------------------------
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

echo
echo "== sweep: bands =="
for BANDS in 16 32 64 113 128 224; do
  "$BENCH" --width=512 --height=512 --bands="$BANDS" --targets=4 --iterations=100
done | tee "$OUT/sweep_bands.txt"

# ---- 3. the experiment this platform exists for ----------------------------
echo
echo "== copy vs zero-copy =="
echo "On unified memory an H2D copy moves bytes from LPDDR5 to LPDDR5."
echo "Watch the upload column in the stage breakdown."
for MEMORY in copy zerocopy; do
  for VARIANT in baseline optimized half; do
    echo "--- memory=$MEMORY variant=$VARIANT ---"
    "$DETECT" --memory="$MEMORY" --variant="$VARIANT" \
              --width=512 --height=512 --bands=128 --targets=4 \
              --frames=200 --prefill \
              --csv="$OUT/pipeline_${MEMORY}_${VARIANT}.csv"
  done
done | tee "$OUT/memory_mode.txt"

# ---- 4. is it real-time, and at what sensor size ---------------------------
echo
echo "== frame budget =="
echo "A 25 fps sensor allows 40 ms per frame; 30 fps allows 33.3 ms."
for CONFIG in "256 512 16" "256 512 25" "512 512 113" "512 512 128" "1024 1024 128"; do
  read -r H W B <<< "$CONFIG"
  echo "--- ${H}x${W}x${B} ---"
  "$DETECT" --memory=zerocopy --height="$H" --width="$W" --bands="$B" \
            --targets=4 --frames=150 --prefill --quiet
done | tee "$OUT/frame_budget.txt"

# The honest end-to-end number: no prefill, so the frame generator's own cost
# is inside the measurement, the way a real sensor's arrival rate would be.
echo
echo "== end-to-end, source in the loop =="
"$DETECT" --memory=zerocopy --width=512 --height=512 --bands=128 \
          --targets=4 --frames=150 | tee "$OUT/end_to_end.txt"

# ---- 5. nsight -------------------------------------------------------------
if command -v nsys >/dev/null 2>&1; then
  echo
  echo "== nsys timeline =="
  nsys profile --force-overwrite=true -o "$OUT/timeline" \
       --trace=cuda,nvtx,osrt --cuda-memory-usage=true \
       "$DETECT" --memory=zerocopy --frames=60 --warmup=10 --prefill --quiet \
       > "$OUT/nsys.log" 2>&1 || echo "nsys failed; see $OUT/nsys.log"
  nsys stats "$OUT/timeline.nsys-rep" > "$OUT/nsys_stats.txt" 2>&1 || true
fi

if command -v ncu >/dev/null 2>&1; then
  echo
  echo "== ncu kernel counters =="
  sudo -n ncu --force-overwrite --target-processes all \
      --kernel-name-base function --launch-count 3 \
      --metrics \
dram__bytes.sum.per_second,\
gpu__time_duration.sum,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct \
      -o "$OUT/kernels" \
      "$BENCH" --width=512 --height=512 --bands=128 --targets=4 \
               --iterations=3 --warmup-ms=0 --repeats=1 \
      > "$OUT/ncu.log" 2>&1 || echo "ncu needs root on Jetson; see $OUT/ncu.log"
fi

if [[ -n "$TEGRASTATS_PID" ]]; then
  kill "$TEGRASTATS_PID" 2>/dev/null || true
  TEGRASTATS_PID=""
  # A thermal ceiling reached mid-run invalidates the tail of every number above.
  echo
  echo "peak temperatures seen during the run:"
  grep -o 'gpu@[0-9.]*C' "$OUT/tegrastats.log" 2>/dev/null | sort -t@ -k2 -rn | head -1 || true
  grep -o 'cpu@[0-9.]*C' "$OUT/tegrastats.log" 2>/dev/null | sort -t@ -k2 -rn | head -1 || true
fi

echo
echo "done. results in $OUT"
