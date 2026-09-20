# Scripts reference

All scripts are run from the project root. Unless noted, load the environment
first:

```powershell
. .\env.ps1
& $python scripts\<name>.py
```

"Live Minecraft" means the script launches/controls a real CraftGround client.

---

## Setup, build and runtime utilities

| Script | Purpose | Live MC | Outputs |
|---|---|---|---|
| `download.py` | Downloads/verifies JDK 21, LLVM-mingw and MaleCNS connectome data. | no | `.tools/`, `vendor/doomfly/connectome_data/malecns_v1/` |
| `build_kernel.py` | Compiles the unchanged upstream `kernel.cpp` to a Windows DLL with hashes. | no | `vendor/doomfly/outputs/doom/libneural.so(.json)` |
| `prepare_runtime.py` | Windows CraftGround build/run patches (static link, socket cleanup, graceful exit, window aspect ratio, framebuffer presentation). | no | patched files in `.venv` |
| `prepare_training_runtime.py` | Exposes genuine block-breaking progress + target telemetry via the integer statistics channel. | no | patched mixin/Kotlin in `.venv` |

---

## Verification and unit tests (offline)

| Script | Purpose | Outputs |
|---|---|---|
| `test_brain.py` | Full-connectome native sanity test with synthetic RGB. | `artifacts/neural-proof.json` |
| `test_decoder.py` | Bilateral symmetry, bounds, polarity, legacy equivalence, checkpoint compatibility. | stdout |
| `test_bilateral_decoder_v2.py` | Natural-scene v2 calibration numerics + v1 back-compat. | stdout |
| `test_bilateral_decoder_v3.py` | v3 closed-loop intercept solver numerics. | stdout |
| `test_mirrored_calibration_v4.py` | Mirrored-pair start geometry. | stdout |
| `test_action_credit.py` | Action-conditioned credit unit tests (fake brain). | stdout |
| `test_readout_learning.py` | Supervised readout teacher geometry + encoder/MLP imitation. | stdout |
| `test_training.py` | Upstream LTD, reward deltas, PPL101 pulse, checkpoint continuation, random starts, R8 structure. | `artifacts/training-proof/report.json` |
| `test_speed_exact.py` | Bit-for-bit regression: reference vs optimized network, plasticity, frozen mode, bidirectional checkpoints. | `artifacts/speed/exactness.json` |
| `verify_baseline.py [folder]` | Audits a baseline artifact folder (action mapping, movement, attack, RGB→retina hashes, unchanged weights). | `<folder>/verification.json` |
| `verify_r8_input.py` | R8 adapter structure + vector-current smoke test. | stdout |
| `verify_visuals.py <folder>` | Validates output videos and per-neuron spike hashes. | `<folder>/visuals-verification.json` |

---

## Live Minecraft integration tests

| Script | Purpose | Outputs |
|---|---|---|
| `test_game.py` | RGB capture, turning, walking, arena reset, attack-based breaking. | `artifacts/craftground-proof/` |
| `test_training_live.py` | Tiny end-to-end train/resume/branch/evaluate/replay + offline playback. | `artifacts/training-proof/live-report.json` + run dirs |
| `benchmark_speed.py --live` | Live per-tick speed benchmark (see below). | `artifacts/speed/` |

---

## Decoder calibration and validation (live Minecraft)

| Script | Purpose | Outputs |
|---|---|---|
| `calibrate_bilateral_decoder.py --config <in> --output-config <out>` | Fits the v3 scalar intercept from live yaw-only episodes. | new v3 config; `artifacts/decoder-calibration/closed-loop-v3.json` |
| `calibrate_bilateral_mirrored.py --config <in> --output-config <out>` | From-scratch mirrored-pair calibration (fresh centers/scales + intercept). | new v3 config; `artifacts/decoder-calibration/mirrored-from-scratch-v4.json` |
| `validate_bilateral_bias.py --config <cfg>` | Held-out frozen residual-bias gate (exits nonzero on failure). | `artifacts/decoder-calibration/frozen-bias-validation-v3.json` |
| `validate_bilateral_mirrored.py --config <cfg>` | Held-out mirrored-pair bias gate. | `artifacts/decoder-calibration/mirrored-bias-validation-v4.json` |
| `compare_decoders.py` | Compares legacy vs bilateral yaw on identical recorded spikes (no live run). | `artifacts/decoder-calibration/comparison.json` |
| `compare_weight_effect.py` | Runs two baseline episodes, then replays exact RGB under frozen vs +50% weights. | `artifacts/weight-effect-2d/` |

See [decoder.md](decoder.md) for the mapping and limitations.

---

## Diagnostics (live Minecraft unless noted)

| Script | Purpose |
|---|---|
| `diagnose_kc_inputs.py` | KC input connectivity, presynaptic activity and weighted input over 200 ticks. |
| `diagnose_kc_drive.py` | Incoming KC edge weights and per-tick net/positive/negative drive. |
| `diagnose_kc_live.py` | KC spike counts, membrane-voltage distribution, eligibility, threshold proximity. |
| `diagnose_kc_peak.py` | High-resolution (1 ms substep) search for the highest observed KC voltage. |
| `diagnose_learning_circuit.py` | Static structural check of the learning circuit (no live MC). |

These print to stdout and write no artifacts.

---

## Benchmarking

`benchmark_speed.py`

```powershell
& $python scripts/benchmark_speed.py --live --reference --label reference-live
& $python scripts/benchmark_speed.py --live --label optimized-live
& $python scripts/benchmark_speed.py --reference --label reference-replay
& $python scripts/benchmark_speed.py --label optimized-replay
& $python scripts/benchmark_speed.py --profile-native --label native-profile
```

`--live` needs live Minecraft; otherwise it replays the cached
`artifacts/speed/input-trace.npz`. `--reference` uses the frozen pre-optimisation
adapters. `--profile-native` builds an instrumented native library. See
[performance.md](performance.md).

---

## Filming

`film_arena_step_by_step.py`

```powershell
& $python scripts/film_arena_step_by_step.py --step-delay 20 --tree-delay 60 --tree-options 12 --seed 41027
```

Exports FlyCraft's real arena (and one genuine randomized target tree) as a
player-anchored Minecraft 1.21 filming datapack with relative coordinates and
scheduled build steps. Offline (no live MC). Outputs
`artifacts/flycraft-film-datapack/` (`.mcfunction` files, `pack.mcmeta`,
`README.txt`), `artifacts/flycraft-film-datapack.zip` and
`artifacts/flycraft-film-commands.txt`.

---

## Test fixtures

`scripts/fixtures/speed_reference/` freezes the pre-optimisation learning
adapter and R8 input processing for bitwise regression and replay benchmarking.
`learning.py` and `r8_visual.py` use the same pinned DOOMFLY graph and original
observed kernel as production, but do **not** install the accelerated kernel or
reuse retinal samples. They are loaded dynamically by `test_speed_exact.py` and
must not be updated to match an optimisation — they are the independent
reference. See `scripts/fixtures/speed_reference/README.md`.
