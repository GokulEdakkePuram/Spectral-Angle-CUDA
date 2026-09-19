# Results

Measured on an **RTX 3090** (sm_86, 82 SMs, 936.1 GB/s peak, CUDA 13.0, driver
580.142) in a rented container, and on the CPU reference for detection quality.

**Not yet measured on the Orin.** Everything about zero-copy and unified memory
below is still a prediction.

## What to run

```sh
scripts/build.sh 86              # one arch; the default builds seven
scripts/profile_gpu.sh           # discrete GPU
scripts/profile_jetson.sh        # AGX Orin
```

Both write a timestamped directory under `profiles/`. Correctness runs first
and gates everything after it.

## Correctness

### GPU vs the CPU reference

All four variants, both shapes, on the RTX 3090. Tolerance is 2e-3 rad for the
fp32 paths and 5e-3 for fp16.

| variant | 96×128×24, 4 targets | 520×696×113, 8 targets | verdict |
|---------|---------------------:|-----------------------:|---------|
| `baseline`  | 1.80e-05 | 5.80e-05 | PASS |
| `optimized` | 1.80e-05 | 5.80e-05 | PASS |
| `half`      | 1.72e-04 | 1.27e-04 | PASS |
| `bip`       | 1.80e-05 | 5.80e-05 | PASS |

Zero significant label mismatches in every case. The fp32 paths land ~35×
inside their tolerance and fp16 ~40× inside its own, so the thresholds are not
doing any work here — the kernels simply agree with double precision.

fp16 is better than expected. Accumulating in fp32 while storing in fp16 costs
about 1e-4 rad, which is three orders of magnitude below any sensible detection
threshold.

### Host side

The ENVI read and the CPU reference were cross-checked against an independent
numpy implementation on real HyperBlood cubes:

| check | result |
|-------|--------|
| BIL→BSQ conversion vs numpy    | bit-exact (max diff 0.0) |
| angle map vs float64 numpy SAM | max 6.5e-7 rad, mean 1.1e-7 rad |
| winning target index           | 0 disagreements in 361 224 pixels |

## Kernel throughput

512×512×128 (134.2 MB), 4 targets, RTX 3090:

| variant | ms | GB/s | % of peak | vs baseline |
|---------|---:|-----:|----------:|------------:|
| `baseline`  | 0.650 | 825.5 | 88% | 1.00× |
| `optimized` | 0.164 | 819.3 | 88% | **3.97×** |
| `half`      | 0.098 | 684.3 | 73% | **6.63×** |
| `bip`       | 1.836 | 292.4 | 31% | 0.35× |

BIP→BSQ transpose: 0.421 ms at 636.9 GB/s counting read and write.

### Scaling with target count

The number that decides whether amortising the cube read across targets works.
`optimized` and `half` hold roughly constant while `baseline` scales linearly,
because its target loop is outermost and re-reads the cube every time.

| targets | baseline | optimized | half | opt speedup | opt % of peak |
|--------:|---------:|----------:|-----:|------------:|--------------:|
|  1 | 0.161 | 0.158 | 0.095 | 1.02× | 91% |
|  2 | 0.319 | 0.161 | 0.098 | 1.99× | 89% |
|  4 | 0.653 | 0.164 | 0.098 | 3.98× | 87% |
|  8 | 1.317 | 0.277 | 0.192 | 4.76× | **52%** |
| 16 | 2.642 | 0.531 | 0.384 | 4.97× | **54%** |

### Scaling with band count

| bands | MB | baseline | optimized | half | bip | opt % of peak |
|------:|---:|---------:|----------:|-----:|----:|--------------:|
|  16 |  16.8 | 0.037 | 0.029 | 0.024 | 0.065 | 62% |
|  32 |  33.6 | 0.158 | 0.047 | 0.035 | 0.331 | 76% |
|  64 |  67.1 | 0.318 | 0.088 | 0.059 | 0.871 | 82% |
| 113 | 118.5 | 0.575 | 0.149 | 0.093 | 2.201 | 85% |
| 128 | 134.2 | 0.653 | 0.164 | 0.099 | 1.813 | 87% |
| 224 | 234.9 | 1.166 | 0.278 | 0.158 | 7.133 | 90% |

At 16 bands `baseline` reports **193% of peak**, which looks like a broken
measurement and is not one. Holding iterations at 100, 1000 and 5000 gives
183%, 179% and 172% — it does not amortise away, so it is not launch overhead.

It is L2 reuse, and it is real. `baseline`'s target loop is outermost *per
thread*, so each thread re-reads its own spectrum `num_targets` times back to
back. The working set of one resident wave is roughly
`126k threads × bands × 4 B`, which at 16 bands is ~8 MB against the 3090's
6 MB L2 — so three of the four passes largely hit cache. At 128 bands the same
figure is 64 MB, nothing is retained, and the number falls back to true DRAM
bandwidth.

So the *time* is trustworthy; it is the `% of peak` column that misleads,
because it divides useful bytes by DRAM peak while some of those bytes never
came from DRAM. Read that column as meaningless wherever
`resident_threads × bands × 4 B` approaches L2 size, and note this is also why
`optimized` looks poor at 16 bands (62%): it reads the cube once, so there is
no redundancy for L2 to absorb, and 0.029 ms is too little work to saturate
anything.

## Pipeline, end to end

512×512×128, 4 targets, 3 streams, `--memory=copy`, 200 frames prefilled:

| variant | fps | p50 ms | p99 ms | upload | sam | detect | download |
|---------|----:|-------:|-------:|-------:|----:|-------:|---------:|
| `baseline`  | 79.9 | 37.46 | 38.39 | 36.61 | 0.671 | 0.023 | 0.029 |
| `optimized` | 79.7 | 37.65 | 39.18 | 37.19 | 0.182 | 0.022 | 0.030 |
| `half`      | 80.6 | 37.17 | 37.96 | 36.61 | 0.351 | 0.021 | 0.028 |

**The pipeline is PCIe-bound and the kernel is invisible.** Upload is 36.6 ms
against 0.18 ms of SAM — the kernel is 0.5% of the frame. All three variants
land within 1% of each other because none of them touches the bottleneck.

Aggregate throughput of 80 fps × 134.2 MB is 10.7 GB/s, and 11.1–11.2 GB/s at
one stream. That is a PCIe Gen3 ×16 link running flat out; nothing above the
link can help.

(`nvidia-smi --query-gpu=pcie.link.gen.current` reports Gen1 here, which is a
red herring — the link power-manages down when idle, and the query was run
between transfers. The sustained transfer rate is the honest measurement.)

### Streams

| streams | fps | p50 ms |
|--------:|----:|-------:|
| 1 | 82.7 | 11.99 |
| 2 | 84.7 | 23.55 |
| 3 | 80.8 | 37.09 |
| 4 | 75.9 | 52.80 |
| 6 | 77.8 | 51.38 |

Throughput is flat and latency grows linearly with stream count — the signature
of one serialised resource. Extra streams queue behind the same copy engine
instead of overlapping with anything. **On a discrete GPU `--streams=1` is
strictly better**: same throughput at a third of the latency. The default of 3
is wrong for this path and is only likely to pay off on the Orin, where there
is no copy to serialise on.

## The predictions, scored

Written down in this file before any of them were measured.

| # | prediction | verdict |
|---|------------|---------|
| 1 | zero-copy cuts memory traffic ~3× on Orin | **untested** — needs the Orin |
| 2 | `optimized` approaches 8× over `baseline` at 8 targets | **falsified** — 4.8× |
| 3 | BIP costs ~8× in load efficiency | **partial** — 31% vs 88% of peak achieved; the sector metric needs `ncu` |
| 4 | fp16 ≈2× at the kernel, ≈nothing end to end | **confirmed**, both halves |

**Why 2 failed — and why the obvious explanation is wrong.** The speedup tracks
target count cleanly to 4× and then stops. `optimized` holds 87–91% of peak up
to four targets and falls to 52% at eight.

Register pressure was the natural suspect and `cuobjdump -res-usage` rules it
out:

| instantiation | registers | occupancy on sm_86 |
|---------------|----------:|-------------------:|
| `sam_opt_kernel<1>`  | 40 | 100% |
| `sam_opt_kernel<2>`  | 39 | 100% |
| `sam_opt_kernel<4>`  | 44 | 87.5% |
| `sam_opt_kernel<8>`  | 55 | 75% |
| `sam_half_kernel<4>` | 59 | 75% |

75% occupancy does not cost a memory-bound kernel 40% of its bandwidth — such
kernels usually saturate well below that. And `sam_half_kernel<4>` sits at the
same 75% while holding 73–75% of peak at every target count, including the ones
where `optimized` collapses.

What separates them is the ratio of constant-memory reads to global loads in
the inner loop. Each band step issues one 16-byte load and `TT` reads of
`c_targets`:

| kernel | LDC : 16-byte load | % of peak |
|--------|-------------------:|----------:|
| `opt<4>`, 4 px/thread  | 4:1 | 87% |
| `half<4>`, 8 px/thread | 4:1 | 73–75% |
| `opt<8>`, 4 px/thread  | **8:1** | **52%** |

Every 4:1 configuration works and the single 8:1 configuration does not. That
points at constant-cache request rate rather than occupancy, though confirming
it needs `ncu`, which this host blocked.

If that is right, the fix is the opposite of the obvious one: **more** pixels
per thread at TT=8, not fewer. Eight pixels per thread would issue two loads
per band step against the same eight constant reads, restoring 4:1. Halving the
pixels — the first thing register pressure would suggest — would make it 8:1
against an 8-byte load and should be worse still. Untested either way.

The dispatch choice is unaffected: at 8 targets one TT=8 pass at 0.277 ms still
beats two TT=4 passes at 2 × 0.164 = 0.328 ms.

**Why 3 is only partial.** `ncu` returned `ERR_NVGPUCTRPERM` — the rented
container blocks performance counters, so the per-sector load-efficiency metric
was never collected. What the wall clock does say is that `bip` achieves 31% of
peak against `optimized`'s 88%, about 2.8× worse, and that it degrades with
band count (32% at 128 bands, 14% at 224) as the stride grows. That is
consistent with the predicted cause without confirming the mechanism.

**Why 4 confirmed.** 1.66× at the kernel rather than a clean 2×, because `half`
only reaches 73–79% of peak against `optimized`'s 87–91% — half the bytes, but
less work in flight per thread to hide latency with. End to end it is 80.6 fps
against 79.7, which is noise, exactly as predicted for an fp32 source.

### Two things nobody predicted

**The baseline was already at the roofline.** It runs at 87–90% of peak at
every band count. At one target `optimized` beats it by 1.02× — float4 loads
and constant-memory broadcast together buy **2%**. The entire 4–5× win is
amortising the cube read across targets, and nothing else. Coalescing is what
matters; vectorising on top of already-coalesced access does almost nothing.

**Small frames cannot be measured this way.** The 193%-of-peak row is a warning
about the harness, not a result about the kernel.

## What the Orin still has to answer

Everything above is a discrete GPU behind a PCIe link, which is the
configuration this project was *not* built for. The measurements that only
exist on the Orin:

* whether zero-copy removes the upload stage entirely, as designed
* whether, with the copy gone, the variant choice becomes visible end to end —
  on this box it was buried under a 36 ms transfer
* whether more than one stream helps once there is no copy engine to serialise
  on
* what fraction of a 40 ms frame budget is left after detection

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
