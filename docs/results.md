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
numpy implementation on a real HyperBlood cube (`A_1`, 520×696×113):

| check | result |
|-------|--------|
| BIL→BSQ conversion vs numpy         | bit-exact (max diff 0.0) |
| angle map vs float64 numpy SAM      | max 1.1e-6 rad, mean 4.2e-8 rad |
| winning target index                | 0 disagreements in 361 920 pixels |

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

### F_1 — blood against five red lookalikes

The scene worth caring about. It carries blood alongside ketchup, artificial
blood, beetroot juice, poster paint, tomato concentrate and acrylic paint —
all red, none of them separable by colour.

Signature = mean of that scene's labelled blood pixels, threshold 0.10 rad:

| metric | value |
|--------|------:|
| ROC AUC (blood vs rest) | 0.9945 |
| precision               | 1.0000 |
| recall                  | 0.7055 |
| F1                      | 0.8273 |
| tp / fp / fn            | 20934 / **0** / 8737 |

| class              | pixels | mean angle | min angle |
|--------------------|-------:|-----------:|----------:|
| **blood** (target) |  29671 | **0.0916** | 0.0216 |
| beetroot_juice     |  13986 |     0.1857 | 0.1623 |
| poster_paint       |   9333 |     0.2062 | 0.1724 |
| acrylic_paint      |   8328 |     0.2144 | 0.1120 |
| ketchup            |  14177 |     0.2374 | 0.1901 |
| artificial_blood   |   9603 |     0.2795 | 0.1865 |
| uncertain_blood    |    581 |     0.3220 | 0.1834 |
| tomato_concentrate |   8996 |     0.3744 | 0.2324 |
| background         | 255170 |     0.4573 | 0.1606 |

Zero false positives across 320 174 negative pixels, of which 54 427 are the
red lookalikes. The margin is not luck: blood averages 0.0916 rad while the
*closest single pixel* of the nearest confuser, beetroot juice, sits at 0.1623
— above the threshold. This is the case the spectral angle exists for, and it
is the result the rest of the project is built to compute quickly.

### Cross-scene: the threshold does not transfer

The table above uses a signature taken from the same scene, which is a fair
operating mode — an operator marking target pixels in the current frame — but
it is not the same as a signature carried in from elsewhere. Both were run:

| signature from | scored on | AUC | precision @0.10 | recall @0.10 |
|----------------|-----------|----:|----------------:|-------------:|
| F_1            | F_1       | 0.9945 | 1.0000 | 0.7055 |
| A_1 + F_1      | F_1       | 0.9935 | 0.9999 | 0.4458 |
| A_1            | F_1       | 0.8816 | 0.5115 | 0.2168 |
| F_1            | A_1       | 0.8236 | 1.0000 | 0.0588 |
| A_1 + F_1      | A_1       | 0.7909 | 1.0000 | 0.0540 |
| A_1            | A_1       | 0.7026 | 0.9926 | 0.1776 |

Two things fall out of this, and both are operational rather than numerical.

**Ranking transfers; the threshold does not.** AUC stays between 0.79 and 0.99
across every combination, so the angle keeps ordering blood ahead of everything
else even with a foreign signature. But precision at a *fixed* 0.10 rad
collapses from 1.00 to 0.51 when A_1's signature is used on F_1. A deployed
detector therefore cannot ship one constant threshold with one library — it
needs the threshold set per scene, or an adaptive one derived from the angle
distribution of the frame.

**A_1 is hard for a reason that is not the algorithm's.** Note that A_1 scores
*better* with F_1's signature (0.82) than with its own (0.70). Its blood
pixels scatter 0.22 rad from their own class mean — wider than the gap between
its blood and background means — because the class spans different ages,
thicknesses and substrates, with thin regions letting the backing through. A
single mean endmember is simply a poor summary of that class, and F_1's
fresher blood happens to give a better-conditioned one.

None of this is a claim about the state of the art. A single mean endmember is
the weakest reasonable signature, and published results on this dataset that do
better use per-scene signatures, spatial context or learned features — none of
which this pipeline provides. What it claims is that the angle is computed
correctly and fast, and that where the signature fits, the detections are
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
