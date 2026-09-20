# FlyCraft / Minecraft Fruit Fly

FlyCraft connects a reconstructed **MaleCNS fruit-fly nervous system** (166,700
neurons, 25,582,938 retained synapses, from the pinned DOOMFLY project) to
**Minecraft 1.21** through CraftGround. Minecraft RGB is the fly's only sensory
input; the fly's descending-neuron activity drives Minecraft yaw / forward /
attack.

The project is a research and visualisation testbed, not a claim that a real fly
learns Minecraft. Every learning mechanism here is either an upstream
experimental rule or a clearly-labelled engineered one. See
[docs/scientific-caveats.md](docs/scientific-caveats.md) before describing
results.

This README is the entry point and quick reference. The details live in
[docs/](#documentation-index).

> **New here?** Start with [docs/getting-started.md](docs/getting-started.md) —
> a short install-and-run walkthrough of the three main modes.

---

## What the project can do

There are three independent control/learning paths plus the offline viewers:

| Path | What learns | Entry point | Docs |
|---|---|---|---|
| **Frozen baseline** | nothing (fixed connectome) | `.\run-baseline.ps1` | [running-modes](docs/running-modes.md) |
| **Signed reinforcement training** | KC→MBON11 synapses (experimental) | `.\train.ps1` | [running-modes](docs/running-modes.md), [configuration](docs/configuration.md) |
| **Frozen-connectome supervised readout** | a small external motor readout; connectome stays frozen | `.\train-readout.ps1` | [readout-learning](docs/readout-learning.md) |
| **Offline viewers** | n/a | `.\play-activity.ps1`, `run-learning-visualizer.cmd` | [running-modes](docs/running-modes.md) |

---

## Quick start

For a guided first run see [docs/getting-started.md](docs/getting-started.md).
The short version: run everything from the project root in **PowerShell**.

```powershell
# One-time toolchain, data and native-runtime setup (long)
.\setup.ps1

# Verify the frozen model and the Minecraft integration
.\test.ps1

# Run the original frozen fly until Ctrl+C
.\run-baseline.ps1

# Bounded baseline run
.\run-baseline.ps1 -Steps 1200

# Start a new signed-reinforcement training lineage
.\train.ps1 -Mode Fresh -Steps 100000 -CheckpointEvery 10000 -NoPreview

# Continue the latest checkpoint of the latest run
.\train.ps1 -Mode Resume -Checkpoint latest -Steps 50000 -NoPreview

# Frozen evaluation of a checkpoint (no learning)
.\train.ps1 -Mode Evaluate -Checkpoint latest -Episodes 20 -NoPreview

# Train the frozen-connectome supervised readout instead
.\train-readout.ps1 -Mode Fresh -Steps 10000 -Config config/new-arch.json
```

If Windows blocks local scripts:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

---

## The mental model in one diagram

```text
Minecraft RGB (16:9 capture)
    │  center-cropped to 640x480
    ▼
DOOMFLY retinal sampling (R1-R6 luminance, + inferred R8 colour in learning mode)
    ▼
MaleCNS neural simulation   (native kernel, 50 ms of neural time per control tick)
    ▼
Descending-neuron decoder   (legacy BCI  or  balanced bilateral DNp20/DNa02/MBON11)
    ▼
turn / forward / attack
    ▼
Minecraft
```

The external trainer may read Minecraft state to compute reward, success,
termination and logging, but that state is **never** fed back into the fly as
sensory input. See [docs/architecture.md](docs/architecture.md) for the full
description.

---

## Command cheat sheet

```powershell
# --- setup / verification ---
.\setup.ps1                       # full setup
.\setup.ps1 -SkipDataAudit        # skip the long connectome data audit
.\test.ps1                        # neural reference + brain + live game + baseline
.\test.ps1 -NeuralOnly            # neural checks only, no Minecraft

# --- frozen baseline ---
.\run-baseline.ps1                # run until Ctrl+C
.\run-baseline.ps1 -Steps 1200
.\run-baseline.ps1 -Steps 1200 -Record            # Minecraft RGB video
.\run-baseline.ps1 -Steps 1200 -RecordVisuals     # dashboard + retina + activity
.\run-baseline.ps1 -Steps 1200 -RecordActivity    # raw spikes/voltage
.\run-baseline.ps1 -Steps 1200 -RecordTrajectory  # exact 20 Hz pose/action trace
.\run-baseline.ps1 -Steps 1200 -NoPreview
.\run-baseline.ps1 -Steps 1200 -Debug             # periodic snapshots
.\evaluate-baseline.ps1 -Steps 1000               # baseline + closed-loop evidence check

# --- signed reinforcement training ---
.\train.ps1 -Mode Fresh    -Steps 100000 -CheckpointEvery 10000 -NoPreview
.\train.ps1 -Mode Resume   -Checkpoint latest -Steps 50000 -NoPreview
.\train.ps1 -Mode Branch   -Checkpoint ".\artifacts\training\run-...\checkpoints\step-000030000" -Steps 50000 -NoPreview
.\train.ps1 -Mode Evaluate -Checkpoint latest -Episodes 20 -NoPreview
.\train.ps1 -Mode Replay   -Checkpoint latest -Episodes 1

# --- supervised readout training ---
.\train-readout.ps1 -Mode Fresh    -Steps 10000 -Config config/new-arch.json
.\train-readout.ps1 -Mode Resume   -Checkpoint latest -Steps 5000 -Config config/new-arch.json
.\train-readout.ps1 -Mode Evaluate -Checkpoint latest -Episodes 20 -Config config/new-arch.json
.\train-readout.ps1 -Mode Replay   -Checkpoint latest -Episodes 3 -Config config/new-arch.json

# --- offline playback / explainer animation ---
.\play-activity.ps1 ".\artifacts\training\run-...\episode-000020-step-000011691"
.\run-learning-visualizer.cmd

# --- list checkpoints ---
Get-ChildItem ".\artifacts\training\run-*\checkpoints" -Directory
```

`-Steps 0` on the baseline means "run until stopped". See
[docs/running-modes.md](docs/running-modes.md) for every switch and mode.

---

## Documentation index

| Document | Contents |
|---|---|
| [docs/getting-started.md](docs/getting-started.md) | Install and run the three main modes — the gentle overview. |
| [docs/architecture.md](docs/architecture.md) | How the whole system works: data path, neural simulation, decoders, learning rules, recording. |
| [docs/running-modes.md](docs/running-modes.md) | Every way to run the program: baseline, all `train.ps1` modes, readout modes, offline player, visualizer. |
| [docs/setup.md](docs/setup.md) | `setup.ps1`, `env.ps1`, toolchain download, native kernel build, CraftGround runtime patches. |
| [docs/configuration.md](docs/configuration.md) | `training.json` and the `config/*.json` files, field by field. |
| [docs/checkpoints-and-artifacts.md](docs/checkpoints-and-artifacts.md) | Checkpoint format, pointers, run folders, metrics, recorded episodes. |
| [docs/decoder.md](docs/decoder.md) | Legacy BCI vs. balanced bilateral decoder, calibration and limitations. |
| [docs/readout-learning.md](docs/readout-learning.md) | Frozen-connectome supervised motor readout (demonstration → DAgger → autonomous). |
| [docs/visual-input.md](docs/visual-input.md) | DOOMFLY v6 R8 colour visual adapter and the R1-R6 luminance path. |
| [docs/performance.md](docs/performance.md) | Speed benchmark, native kernel specialisation, exactness regression. |
| [docs/scripts.md](docs/scripts.md) | Reference for every script in `scripts/`. |
| [docs/video-and-explainer.md](docs/video-and-explainer.md) | The YouTube/video-production track: pictogram interpreter and motion-canvas. |
| [docs/troubleshooting.md](docs/troubleshooting.md) | Common failures and fixes. |
| [docs/scientific-caveats.md](docs/scientific-caveats.md) | What may and may not be claimed about results. |

---

## Repository layout (tracked source)

```text
README.md                  this file
training.json              default signed-reinforcement training config
model-lock.json            pinned upstream revisions + original weight hash
requirements.lock.txt      pinned Python dependencies
env.ps1                    loads the project-local environment into a shell
setup.ps1                  one-time toolchain/data/native setup
test.ps1                   end-to-end verification
run-baseline.ps1           frozen-baseline launcher
evaluate-baseline.ps1      baseline + closed-loop evidence check
train.ps1                  signed-reinforcement training launcher
train-readout.ps1          supervised-readout training launcher
play-activity.ps1          offline neural-activity player
run-learning-visualizer.cmd   explainer animation launcher
config/                    baseline + training + decoder configurations
src/flycraft/              the Python package (see docs/architecture.md)
scripts/                   build, test, calibration, diagnostic and film tools
docs/                      this documentation set
motion-canvas/             Motion Canvas overlay sub-project (video track)
```

Large/generated pieces are deliberately **not** tracked: `.tools/`, `.venv/`,
`vendor/`, `artifacts/` (see `.gitignore`).

---

## Source/version lock

`model-lock.json` pins the expected upstream revisions and the original neural
weight hash:

```text
DOOMFLY commit:    71ecf53d78eaffaf1a57ed7b0ccf5d458abc9f33
CraftGround commit: 18eba01a87a8481bc4fbb54b42fee1428c0c3719
CraftGround pkg:   2.7.4
MaleCNS neurons:   166,700
Directed edges:    25,582,938
```

The frozen baseline verifies the original fixed weights remain unchanged.
Training saves its mutable experimental state through checkpoints. Do not start
long runs until you have read [docs/configuration.md](docs/configuration.md) and
[docs/scientific-caveats.md](docs/scientific-caveats.md).
