# Results

**Status: no measurements yet.** Nothing in this file has been run on real
hardware. What follows is the ceiling the performance model predicts, the exact
commands that produce the numbers, and what each one should be checked against.
Measured tables go in below, replacing the placeholders.

## What to run

```sh
scripts/build.sh                 # or: scripts/build.sh 87   on the Orin
scripts/profile_gpu.sh           # discrete GPU
scripts/profile_jetson.sh        # AGX Orin
```

Both write a timestamped directory under `profiles/`. Correctness runs first
and gates everything after it.

## Predicted ceilings

SAM is bandwidth bound at 0.5 FLOP/byte (see [design.md](design.md)), so the
kernel cannot beat `cube_bytes / peak_bandwidth`. These are therefore hard
upper bounds for the SAM kernel alone — not for the pipeline, which also pays
for detection and readback, and in `--memory=copy` for the upload as well.

| cube                     |     MB |
|--------------------------|-------:|
| synthetic 512×512×128     |  134.2 |
| HyperBlood 520×696×113    |  163.6 |
| HOT VIS 256×512×16        |    8.4 |

| device                   | peak GB/s | 512×512×128 | 520×696×113 | 256×512×16 |
|--------------------------|----------:|------------:|------------:|-----------:|
| AGX Orin (LPDDR5)        |     204.8 |   1526 fps |   1252 fps |  24414 fps |
| L4                       |     300.0 |   2235 fps |   1834 fps |  35763 fps |
| A10                      |     600.0 |   4470 fps |   3668 fps |  71526 fps |
| RTX 4090                 |    1008.0 |   7510 fps |   6162 fps | 120163 fps |
| A100 80GB                |    1935.0 |  14417 fps |  11829 fps | 230670 fps |
| H100 SXM                 |    3350.0 |  24959 fps |  20478 fps | 399351 fps |

A well-behaved streaming kernel usually lands at 70–85% of theoretical peak, so
treat anything above ~85% as a measurement error and anything below ~50% as a
problem worth finding with `ncu`.

The headline for the Orin: even the heaviest configuration here has roughly
**1250 fps** of headroom against a 25 fps sensor, so the frame budget is not
close to tight. The interesting question on that platform is not whether it
keeps up but how much of the 40 ms budget is left for everything downstream.

## The predictions worth falsifying

These are the specific claims the design rests on. Each one is a number the
profiling scripts produce directly.

**1. Zero-copy is roughly a 3× reduction in memory traffic on Orin.**
In `--memory=copy` the H2D reads 134 MB from LPDDR5 and writes 134 MB back to
the same LPDDR5, then the kernel reads 134 MB again: 402 MB of traffic for a
134 MB cube. Zero-copy moves 134 MB. So the end-to-end frame time should drop
substantially, and the `upload` column of the stage breakdown should go to
zero by construction. Produced by `scripts/profile_jetson.sh`, section
"copy vs zero-copy".

**2. The optimized kernel pulls away from the baseline as targets increase.**
At one target both read the cube exactly once, so the gap is only
vectorisation and constant memory — expect something modest. At eight targets
the baseline reads the cube eight times and the optimized kernel once, so the
gap should approach 8×. If it does not, the amortisation is not working.
Produced by the `sweep: targets` section of either script.

**3. BIP costs about 8× on load efficiency.**
A warp reading at a stride of `bands` fetches 32 sectors to use 128 bytes of
them. `smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct` in the
`ncu` output should be near 100% for `optimized` and near 12.5% for `bip`.
The wall-clock gap will be smaller than that because the L2 absorbs some of it.

**4. fp16 is close to 2× at the kernel and roughly nothing end to end.**
`bench_sam` measures kernels in isolation, where halving the bytes should
nearly halve the time. The pipeline narrows an fp32 cube on the device first,
so end to end it reads 4 bytes, writes 2 and reads 2 — worse than reading 4.
If the pipeline shows `half` winning end to end, something is wrong with the
measurement, not with the world.

## Kernel throughput

_To be filled from `kernels_512x512x128_t4.txt`._

| device | variant | ms | GB/s | % of peak | vs baseline |
|--------|---------|---:|-----:|----------:|------------:|
| | `baseline`  | | | | 1.00× |
| | `optimized` | | | | |
| | `half`      | | | | |
| | `bip`       | | | | |

## Correctness

_To be filled from `correctness_*.txt`._

Every variant is compared against the double-precision CPU reference. The
tolerance is 2e-3 rad for the fp32 paths and 5e-3 rad for fp16; both are
reported in the output rather than assumed.

| device | variant | max angle error (rad) | significant label mismatches | verdict |
|--------|---------|----------------------:|-----------------------------:|---------|
| | `baseline`  | | | |
| | `optimized` | | | |
| | `half`      | | | |
| | `bip`       | | | |

A label mismatch is only counted when the two candidate targets were
meaningfully apart. Where two sit at nearly the same angle from a pixel, which
one wins is decided by the last bit of the accumulation.

## Pipeline, end to end

_To be filled from `pipeline.txt` and `memory_mode.txt`._

| device | memory | variant | fps | p50 ms | p99 ms | upload ms | sam ms | detect ms |
|--------|--------|---------|----:|-------:|-------:|----------:|-------:|----------:|
| | | | | | | | | |

## Detection quality on HyperBlood

_To be filled from `scripts/eval_detection.py`._

```sh
scripts/fetch_hyperblood.sh
python3 scripts/prepare_hyperblood.py
build/src/hsi_detect --source=envi \
  --envi=data/hyperblood_prepared/F_1.hdr \
  --library=data/hyperblood_targets.csv \
  --frames=1 --warmup=0 --dump-angle=/tmp/F_1.f32
python3 scripts/eval_detection.py /tmp/F_1.f32 \
  data/hyperblood_prepared/F_1_gt.u8 --width=696 --height=520
```

The number to look at is not the ROC AUC against background — that is easy.
It is the per-class mean-angle table: how far blood separates from ketchup,
tomato concentrate, beetroot juice and the two red paints. Those are the
classes colour cannot separate, and they are why this dataset was chosen.

| scene | AUC (blood) | mean angle: blood | ketchup | tomato | beetroot | paints |
|-------|------------:|------------------:|--------:|-------:|---------:|-------:|
| F_1 | | | | | | |

## Notes on interpretation

* An unpinned run measures the governor, not the code. Both scripts pin clocks
  where they have the privileges; if they warn that they could not, the numbers
  are a lower bound of unknown tightness.
* On the Orin, check the peak temperatures the script prints at the end. A
  thermal ceiling reached mid-run invalidates everything measured after it.
* Use `--prefill` to measure the GPU and omit it to measure the whole system.
  The synthetic generator writes tens of millions of floats per frame and can
  easily be the bottleneck; both numbers are worth having, but they answer
  different questions.
