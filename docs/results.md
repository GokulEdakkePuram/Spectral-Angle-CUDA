# Results

**Status: no GPU measurements yet.** Nothing here has run on a GPU. The
performance tables hold the ceiling the model predicts, the commands that
produce the real numbers, and what each should be checked against.

The correctness and detection-quality sections below *are* measured — on the
CPU reference, against real HyperBlood data. Those results stand on their own,
and the GPU is expected to reproduce the angle map to within the tolerances in
the correctness table.

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

### Host side, measured

The ENVI read and the CPU reference were cross-checked against an independent
numpy implementation on real HyperBlood cubes:

| check | result |
|-------|--------|
| BIL→BSQ conversion vs numpy    | bit-exact (max diff 0.0) |
| angle map vs float64 numpy SAM | max 6.5e-7 rad, mean 1.1e-7 rad |
| winning target index           | 0 disagreements in 361 224 pixels |

That fixes the reference the GPU is measured against to real data rather than
to itself.

### GPU, to be filled from `correctness_*.txt`

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

Measured on the CPU reference, so these hold regardless of which GPU runs them.

```sh
scripts/fetch_hyperblood.sh
python3 scripts/prepare_hyperblood.py

# --target=blood matters: with a full library the angle map holds the best
# match over all targets, which is not the same quantity as blood-likeness.
build/src/hsi_score --cube=data/hyperblood_prepared/A_1.hdr \
  --library=data/hyperblood_targets.csv --target=blood \
  --threshold=0.10 --dump-angle=/tmp/A_1_blood.f32

python3 scripts/eval_detection.py /tmp/A_1_blood.f32 \
  data/hyperblood_prepared/A_1_gt.u8 --width=696 --height=520 --target=1
```

All 14 scenes, blood as the target, threshold 0.10 rad. Two signature regimes,
because they answer different questions:

* **on-scene** — the signature is the mean of that scene's own labelled blood.
  This is the operator-in-the-loop mode: mark target pixels in the current
  frame, then detect.
* **global** — one signature averaged over all 14 scenes, carried in from
  elsewhere. This is the ship-it-in-a-config-file mode.

| scene | on-scene AUC | prec | recall | global AUC | prec | recall |
|-------|-------------:|-----:|-------:|-----------:|-----:|-------:|
| A_1   | 0.7019 | 0.9924 | 0.1804 | 0.5979 | 0.8097 | 0.2355 |
| B_1   | 0.7662 | 0.8791 | 0.0662 | 0.5516 | 0.6893 | 0.0530 |
| C_1   | 0.7891 | 0.0826 | 0.0036 | 0.7218 | 0.8006 | 0.1315 |
| D_1   | 0.8104 | 0.9027 | 0.7274 | 0.9979 | 0.9997 | 0.1446 |
| E_1   | 0.7296 | 0.5793 | 0.0813 | 0.6894 | 0.3385 | 0.0590 |
| E_7   | 0.8096 | 0.2083 | 0.1009 | 0.6020 | 0.0039 | 0.0013 |
| E_21  | 0.7426 | 0.0501 | 0.0613 | 0.6721 | 0.0227 | 0.0051 |
| **F_1**  | **0.9945** | **1.0000** | 0.7073 | 0.6940 | 0.3351 | 0.0388 |
| F_1a  | 0.9930 | 0.6865 | 0.9438 | 0.9586 | 0.7250 | 0.2864 |
| F_1s  | 0.9964 | 0.8660 | 0.8790 | 0.9805 | 0.5377 | 0.2038 |
| F_2   | 0.9991 | 0.9275 | 0.9774 | 0.9412 | 0.5559 | 0.1913 |
| F_7   | 0.9838 | 0.9655 | 0.9076 | 0.9301 | 0.5992 | 0.1709 |
| F_21  | 0.9979 | 0.9847 | 0.9380 | 0.9191 | 0.3439 | 0.0694 |
| F_2k  | 0.9959 | 0.9161 | 0.9517 | 0.9122 | 0.1609 | 0.0431 |
| **mean** | **0.8793** | | | **0.7977** | | |

### What the table says

**The scene split matters more than the signature.** The `F_*` scenes — the
"frame" images, blood on a controlled background — reach 0.98–0.999 AUC with
recall around 0.9 when calibrated on scene. The `A`–`E` mock-up scenes, with
cluttered natural backgrounds, sit at 0.70–0.81. Same kernel, same signature
procedure; the difference is entirely how much the background overlaps the
target in spectral direction. SAM has no spatial context to fall back on when
it does.

**On-scene calibration is worth about 0.08 AUC on average**, and much more than
that at a fixed operating point. F_1 goes from 0.694 AUC with the global
signature to 0.9945 with its own, and its precision at 0.10 rad from 0.34 to
1.00.

### F_1 in detail — blood against five red lookalikes

The case this dataset exists for: blood alongside ketchup, artificial blood,
beetroot juice, poster paint, tomato concentrate and acrylic paint, all red,
none separable by colour. On-scene signature, 0.10 rad:

| metric | value |
|--------|------:|
| ROC AUC | 0.9945 |
| precision | 1.0000 — **0 false positives in 320 000 negatives** |
| recall | 0.7073 |

| class | mean angle | min angle |
|-------|-----------:|----------:|
| **blood** (target) | **0.0916** | 0.0216 |
| beetroot_juice     | 0.1857 | 0.1623 |
| poster_paint       | 0.2062 | 0.1724 |
| acrylic_paint      | 0.2144 | 0.1120 |
| ketchup            | 0.2374 | 0.1901 |
| artificial_blood   | 0.2795 | 0.1865 |
| tomato_concentrate | 0.3744 | 0.2324 |
| background         | 0.4573 | 0.1606 |

The margin is not luck: blood averages 0.0916 rad while the *closest single
pixel* of the nearest confuser, beetroot juice, sits at 0.1623 — above the
threshold. That separation is the entire argument for the spectral angle, and
it is what the rest of this project exists to compute quickly.

### The operational caveat

Ranking transfers between scenes; the threshold does not. Global-signature AUC
stays respectable almost everywhere, so the angle keeps ordering blood ahead of
the rest — but precision at a *fixed* 0.10 rad swings from 1.00 to 0.004
depending on the scene. A deployed detector cannot ship one constant threshold
with one library. It needs the threshold set per scene, or derived from the
angle distribution of the frame itself.

None of this is a claim about the state of the art. A single mean endmember is
the weakest reasonable signature, and published results that do better on this
dataset use per-scene signatures, spatial context or learned features — none of
which this pipeline provides. What it claims is that the angle is computed
correctly and fast, and that where the signature fits, the calls it makes are
right.

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
