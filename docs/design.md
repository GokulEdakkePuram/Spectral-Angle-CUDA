# Design notes

## The performance model

Everything about this pipeline follows from one ratio. Per pixel, per target,
SAM reads `bands` float32 values and does two fused multiply-adds — one for the
dot product, one for the pixel's own norm:

```
bytes  = bands * 4
flops  = bands * 2        (per target)
ratio  = 0.5 FLOP/byte    (at one target)
```

Every GPU this targets has a ridge point one to two orders of magnitude above
that. An A100 is around 100 FLOP/byte; the AGX Orin, with roughly 200 GB/s of
LPDDR5 against several TFLOP/s, is in the same territory. SAM at one target is
nowhere near it.

So the kernel's runtime is `bytes_moved / achieved_bandwidth`, and there are
exactly two ways to make it faster: move fewer bytes, or move the same bytes
closer to peak. Nothing else on the list matters. This is why the variants are
what they are, and why the benchmark reports achieved GB/s and percent of peak
rather than FLOP/s — a FLOP/s figure here would just be the byte count in
different units.

A useful consequence: the floor is knowable in advance. A 512×512×128 float32
cube is 134 MB. At 200 GB/s that is 0.67 ms, so about 1500 frames per second is
the ceiling for the kernel alone on an Orin, and any measurement far below that
has an explanation worth finding.

## Where the bytes go

**Layout.** One thread per pixel over a band-sequential cube: on each band step
thread `p` reads `cube[b * pixels + p]`, so the 32 threads of a warp read 32
consecutive floats and the load coalesces into four 32-byte sectors. The same
kernel over a band-interleaved-by-pixel cube reads at a stride of `bands`, so
each thread pulls its own sector and the warp fetches 32 of them to use 128
bytes. The `bip` variant exists to put a number on that rather than assert it.

Sensors that deliver BIP are handled by a tiled transpose on the device, staged
through shared memory so both the read and the write coalesce, with the tile
padded by one column so the transposed access spreads across all 32 banks. On
the Orin this is much cheaper than transposing on the CPU cores, which would
otherwise become the bottleneck.

**Amortising across targets.** Reading a pixel's spectrum costs `bands * 4`
bytes; scoring it against one more target costs two more FLOPs per band and no
extra bytes. So the cost of a second target should be nearly zero — but only if
the kernel holds accumulators for several targets across one pass of the cube.
The baseline does not: its target loop is outermost, so it re-reads the whole
cube per target. That is deliberate, and it is what makes `baseline` at one
target a fair comparison (both read the cube exactly once, isolating
vectorisation and constant memory) and `baseline` at eight targets an honest
picture of what failing to amortise costs.

The optimized kernel is templated on the number of targets per pass and
dispatched at the largest instantiated width that fits — 8, 4, 2 or 1 — so the
common library sizes run in a single pass with no predication in the inner
loop. Eight targets times four pixels is 32 accumulator registers, which is
about where occupancy starts to pay for further widening. Libraries larger than
eight take extra passes, each one a re-read; the running best is carried in the
output buffer between them.

**Halving the input.** On a kernel this bandwidth-starved, moving to fp16 is
the only change that can approach a 2× speedup, and one 16-byte load then
covers eight pixels instead of four. Accumulation stays in fp32: fp16 sums over
100+ bands lose enough mantissa to move the angle by more than a detection
threshold cares about.

One caveat, stated plainly because the benchmark will show it. In the pipeline
the source produces fp32 and the cube is narrowed on the device, so end to end
the fp16 path reads 4 bytes, writes 2 and reads 2 — worse than just reading 4.
`half` is a kernel-level result unless the sensor itself delivers fp16, which
is why `bench_sam` measures kernels in isolation and the pipeline reports
stages separately.

## Constant memory for the target library

On any given band step every thread in a warp reads the same target element.
That is the broadcast pattern the constant bank is built for: one cache line
serves the whole warp at close to register latency. 32 KB of the 64 KB bank is
staged, which covers, say, 32 targets across 256 bands. Beyond that the launcher
falls back to multiple passes.

Target norms are uploaded as reciprocals so the kernel multiplies rather than
divides.

## Numerical choices

**One `acosf` per pixel.** Candidates are ranked on cosine and converted once
at the end. `acos` is monotonically decreasing, so ranking on cosine gives the
same winner, and this removes all but one transcendental per pixel.

**Degenerate pixels never branch.** A pixel of all zeros has no direction. Its
`rsqrt` is forced to 0, giving a cosine of 0, whose `acos` is exactly `pi/2` —
the largest angle SAM can return for non-negative reflectance, and therefore
one that never trips a sane threshold. The right answer falls out of the
arithmetic with no special case.

**The reference is double precision.** `sam_cpu` accumulates in double
throughout. A float reference would drift alongside the kernels and turn the
correctness test into a comparison of two approximations.

**Tie-breaking is specified, not incidental.** Where two targets sit at nearly
the same angle from a pixel, which one wins is decided by the last bit of the
accumulation. The benchmark reports label mismatches as total and significant,
and only counts the ones where the two candidates were genuinely apart.
Likewise non-maximum suppression breaks ties by linear index, identically on
CPU and GPU, so a plateau of equal angles emits exactly one detection and the
two implementations can be compared element for element.

## Pipeline

Frames run in slots, each with its own stream and buffers, so while the GPU
works on frame N the host is producing frame N+1. The only blocking point is
reclaiming a slot about to be overwritten.

Only the compacted detection list is read back, never the angle map. Compaction
uses one atomic per warp rather than one per detection: a cluttered frame can
push tens of thousands of pixels through the threshold, and having each hit the
same counter serialises the tail of the kernel. Instead the warp ballots,
elects a leader to reserve the whole warp's slots in one `atomicAdd`, and each
lane writes at its rank in that reservation.

Non-maximum suppression is on by default because any target wider than one
pixel trips the threshold across its whole footprint. Without it the output is
thousands of pixels rather than a handful of targets, and the readback stops
being small.

## Zero-copy on Jetson

The AGX Orin's CPU and GPU share one physical LPDDR5 pool. An H2D copy there
reads from LPDDR5 and writes to LPDDR5 — at ~160 MB per frame it is the largest
single cost in the pipeline, and it buys nothing. `MemoryMode::ZeroCopy`
allocates the staging buffer with `cudaHostAllocMapped` and hands the kernels
the mapped device pointer, removing the copy entirely.

Two details make it actually work:

* Device cubes carry a band-plane stride padded to eight elements, so every
  plane starts 16-byte aligned and the vector loads are legal past band zero.
  On a discrete GPU the 2D upload fixes the layout in transit; with zero-copy
  there is no transit, so frame sources are handed the stride to write at.
* `cudaSetDeviceFlags(cudaDeviceMapHost)` has to run before anything creates
  the context, so the capability check and the flag come first in setup.

On a discrete GPU the same mode turns every kernel read into a PCIe
transaction, which is why it is a flag rather than a default and why
`hsi_detect` prints whether the device is integrated.

## What is not here

* **No spatial context.** SAM scores each pixel independently. Real detectors
  usually follow it with a matched filter or ACE against a background
  covariance, which is a different and much less bandwidth-bound problem.
* **No sub-pixel unmixing.** A pixel containing 20% target and 80% background
  has an angle somewhere between the two, and SAM cannot say which.
* **No tracking.** Detections are per frame, with no association across time.
* **No atmospheric correction.** Signatures must be in the same space as the
  data — both reflectance or both radiance.
