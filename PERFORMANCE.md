# Speed benchmark

Measured locally on 2026-09-12, with all 166,700 neurons and 25,582,938 retained
edges. Eight warm-up ticks, then 40 ticks at 50 simulated ms per tick. Startup,
checkpoint saving, and episode resets are excluded. Short measurements fluctuate;
the live and fixed-input replay results are reported separately.

| Mode | Before ms/tick | After ms/tick | Speedup |
|---|---:|---:|---:|
| Live Minecraft, normal non-recorded training | 221.79 | 217.14 | 1.021x |
| Same captured inputs, neural/controller replay | 218.96 | 206.95 | 1.058x |

Live simulated seconds per wall second improved from 0.2254 to 0.2303.

| Live component, ms/tick | Before | After |
|---|---:|---:|
| Minecraft/CraftGround step | 7.69 | 7.53 |
| R1-R6 retinal sampling | 5.43 | 1.04 |
| Native simulation, including plasticity | 202.00 | 201.86 |
| Python brain/native-call overhead, including R8 | 4.34 | 4.49 |
| Reward/dashboard metrics | 0.58 | 0.59 |
| JSON logging | 0.13 | 0.12 |
| Recording: additional cost on selected ticks | 718.72 | 735.24 |

Recording was separately sampled over four ticks using the existing 1080p output.
Its cost excludes recorder construction/close and includes rendering, encoding
pipe writes and activity serialization. It is not part of the non-recorded total.
Profiling uses an instrumented native library in replay, separately from normal
timings: after optimisation, active integration costs 99.69 ms/tick, synaptic
delivery 88.42 ms (including 4.84 ms plasticity), and full scans/tables 6.30 ms.
These instrumentation numbers are not additive to the live table.

## Changes and validation

* Exact: specialize the common one-timestep native evolution case, retaining
  general elapsed-time handling and the original floating-point expression.
* Exact: sample one held RGB image once per control tick; retain every 10 ms
  retinal filter/current update, native timestep, and spike schedule.
* Checkpoint numerical identity is unchanged; `native_runtime` in new run model
  metadata records the actual accelerated executable separately.
* The full-network regression compares all state/weight bytes, controller output
  and 60 Hz spike bins, tests actual depression (4,184 changed plastic edges),
  frozen evaluation and checkpoint loading in both directions.
* An AVX2 batch experiment passed exactness but was slower; it was discarded.
  No approximate mode, pruning, rate neurons or changed learning rule was added.

[Aimbug](https://github.com/slickdomi/aimbug) informed the batching investigation.
Its reduced/rate-based visual model was not adopted because it would change this
model. The next worthwhile experiment is moving video rendering/encoding to an
offline replay of recorded observations, retaining the same simulation inputs.

## Reproduce

Run these individually in PowerShell from the project directory; do not run
benchmarks concurrently. A live run launches a separate Minecraft on port 8031.

```powershell
. ./env.ps1
$env:OPENBLAS_NUM_THREADS='1'
& $python scripts/benchmark_speed.py --live --reference --label reference-live
& $python scripts/benchmark_speed.py --live --label optimized-live
& $python scripts/benchmark_speed.py --reference --label reference-replay
& $python scripts/benchmark_speed.py --label optimized-replay
& $python scripts/benchmark_speed.py --profile-native --label native-profile
& $python scripts/test_speed_exact.py
```

Results and the first captured input trace are under `artifacts/speed/`. Use a
fresh label for each live run because recording directories must be new. Replay
defaults to the 40 measured ticks available in that trace. The regression can
also run without a captured trace, using seeded synthetic frames. Frozen
pre-optimisation adapters are retained under `scripts/fixtures/speed_reference`.
`FLYCRAFT_REFERENCE_KERNEL=1` disables only the native specialization for
diagnostics; use benchmark `--reference` for the complete original input path.
