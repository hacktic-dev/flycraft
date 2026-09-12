# Minecraft Fruit Fly / FlyCraft

A Windows research/visualisation project that connects a reconstructed **MaleCNS fruit-fly nervous-system model** from DOOMFLY to **Minecraft 1.21** through CraftGround.

The project has two deliberately separate modes:

- **Frozen baseline** — the original fixed-weight fly model receives Minecraft RGB through the existing retinal sampler and drives Minecraft through the existing descending-neuron decoder. Learning is off.
- **Experimental aversive training** — the same visual/control path is used, but DOOMFLY's existing experimental `gamma1-eligibility-ltd-v1` aversive plasticity model is enabled. Negative outcomes can schedule PPL101 aversive stimulation. Positive reward is currently a **metric only**; no positive-reward plasticity rule is invented by FlyCraft.

This README is primarily a practical reference for running, resuming, evaluating, replaying, recording, and understanding the files produced by the project.

---

### Minecraft aspect ratio / neural crop

FlyCraft asks CraftGround for a **16:9 (854x480 at the default 480 px height) raw render** and then center-crops it to the configured **640x480 4:3** view before retinal sampling and sensory hashing. Human-facing previews, dashboard game panes, and game videos use the uncropped 16:9 render. This avoids the CraftGround/Windows path that otherwise compresses a 16:9 render horizontally into a nominal 640x480 frame. The crop is geometric only: it adds no game-state information and only the corrected center crop is passed to `retinal_samples`; the wider game preview is for viewers only.


## 1. Quick command cheat sheet

Run all commands from the project root in **PowerShell**.

```powershell
# One-time setup
.\setup.ps1

# Verify original frozen model + Minecraft integration
.\test.ps1

# Run the original untrained/frozen fly until Ctrl+C
.\run-baseline.ps1

# Run a bounded baseline
.\run-baseline.ps1 -Steps 1200

# Start a completely new training lineage from the original untrained fly
.\train.ps1 -Mode Fresh -Steps 100000 -CheckpointEvery 10000 -NoPreview

# Same training run, but render recorded/preview dashboards with both membrane voltage and spikes
.\train.ps1 -Mode Fresh -Steps 100000 -CheckpointEvery 10000 -ActivityMode Combined -VoltageSmoothingMs 80 -NoPreview

# Continue the latest checkpoint of the latest run for another 50,000 steps
.\train.ps1 -Mode Resume -Checkpoint latest -Steps 50000 -NoPreview

# Branch into a NEW run from an older checkpoint
.\train.ps1 -Mode Branch `
  -Checkpoint ".\artifacts\training\run-...\checkpoints\step-000050000" `
  -Steps 50000 -NoPreview

# Frozen evaluation of a checkpoint; does not learn or alter the checkpoint
.\train.ps1 -Mode Evaluate `
  -Checkpoint ".\artifacts\training\run-...\checkpoints\step-000050000" `
  -Episodes 20 -NoPreview

# Frozen evaluation + video/visual recording of a checkpoint
.\train.ps1 -Mode Replay `
  -Checkpoint ".\artifacts\training\run-...\checkpoints\step-000050000" `
  -Episodes 1

# Evaluate the best checkpoint pointer from a run
.\train.ps1 -Mode Evaluate `
  -Checkpoint ".\artifacts\training\run-...\checkpoints\best.json" `
  -Episodes 20 -NoPreview

# List checkpoint directories from all runs
Get-ChildItem ".\artifacts\training\run-*\checkpoints" -Directory
```

If Windows blocks local PowerShell scripts, use a process-local bypass:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

---

## 2. Setup

### One-time setup

```powershell
.\setup.ps1
```

`setup.ps1` creates a project-local toolchain and environment. It downloads/pins the upstream DOOMFLY and CraftGround revisions, installs Python 3.11, Java 21 and other local dependencies, downloads/audits the MaleCNS data, builds the native neural kernel and CraftGround native runtime, builds Minecraft's Java side, validates the brain, and prepares the anatomical geometry used by the visualiser.

The project deliberately keeps these large/generated pieces out of Git:

```text
.tools/
.venv/
vendor/
artifacts/
```

To skip the long full data audit on a repeated setup:

```powershell
.\setup.ps1 -SkipDataAudit
```

### Basic verification

```powershell
.\test.ps1
```

This runs the original DOOMFLY numerical reference, the full-connectome test, a live CraftGround action test, and a bounded frozen baseline. `test.ps1` is a **baseline test**, not a learned-checkpoint evaluator.

Neural-only verification, without live Minecraft:

```powershell
.\test.ps1 -NeuralOnly
```

---

## 3. Mental model of the system

The environmental data path is intentionally narrow:

```text
Minecraft RGB
    ↓
DOOMFLY retinal_samples
    ↓
MaleCNS neural simulation
    ↓
existing NeuralControls decoder
    ↓
turn / forward / attack
    ↓
Minecraft
```

The external trainer can inspect Minecraft state to calculate reward, episode termination, logging, and evaluation, but that state is **not fed into the fly as sensory input**.

The current control mapping is based on the existing DOOMFLY decoder:

```text
DNp20 L/R rate difference → camera yaw
DNpe017 combined rate      → forward key
DNpe017 spikes             → attack pulse
```

FlyCraft adds a small neural-only attack accumulator/hold mechanism during training so spike pulses can keep left mouse held long enough to break a Minecraft block.

One control/training step corresponds to **50 ms of simulated neural time** and one Minecraft control tick. The native neural integration itself remains at the upstream finer timestep.

---

# 4. Frozen baseline commands

## Normal baseline

```powershell
.\run-baseline.ps1
```

This starts a fresh original frozen brain and continues until `Ctrl+C`.

Bound the run by Minecraft/neural control steps:

```powershell
.\run-baseline.ps1 -Steps 1200
```

At 20 control ticks per simulated second:

```text
200 steps   = 10 simulated seconds
1,200 steps = 60 simulated seconds
6,000 steps = 5 simulated minutes
```

`-Steps 0` means run until stopped.

## Baseline recording options

```powershell
# Minecraft RGB video
.\run-baseline.ps1 -Steps 1200 -Record

# Full visual dashboard + retina + neural activity videos.
# This also enables activity and trajectory recording.
.\run-baseline.ps1 -Steps 1200 -RecordVisuals

# Raw neural spike/voltage recording without requiring full dashboard video
.\run-baseline.ps1 -Steps 1200 -RecordActivity

# Exact 20 Hz Minecraft pose/action trace only
.\run-baseline.ps1 -Steps 1200 -RecordTrajectory

# Disable interactive preview window
.\run-baseline.ps1 -Steps 1200 -NoPreview

# Save periodic RGB/retina/debug snapshots
.\run-baseline.ps1 -Steps 1200 -Debug
```

Any video/activity recording automatically preserves the Minecraft trajectory too.

## Neural display modes

```powershell
.\run-baseline.ps1 -ActivityMode Spikes
.\run-baseline.ps1 -ActivityMode Voltage
.\run-baseline.ps1 -ActivityMode Combined
```

Voltage display smoothing is visual only; it does not alter the simulation:

```powershell
.\run-baseline.ps1 -ActivityMode Combined -VoltageSmoothingMs 80
.\run-baseline.ps1 -ActivityMode Combined -VoltageSmoothingMs 0
```

## Baseline evaluation

```powershell
.\evaluate-baseline.ps1
```

Default: 300 ticks with debug evidence plus closed-loop verification.

```powershell
.\evaluate-baseline.ps1 -Steps 1000
.\evaluate-baseline.ps1 -Steps 1000 -Record
```

---

# 5. Training: Fresh, Resume, Branch, Evaluate, Replay

The easiest way to think about the five modes is:

```text
Fresh     = original untrained fly → new training timeline
Resume    = latest checkpoint      → continue the SAME timeline
Branch    = chosen checkpoint      → create a NEW timeline
Evaluate  = chosen checkpoint      → frozen tests, no learning
Replay    = chosen checkpoint      → frozen tests + footage
```

## Fresh — start from scratch

```powershell
.\train.ps1 -Mode Fresh -Steps 100000 -CheckpointEvery 10000 -NoPreview
```

`Fresh` always creates a new `artifacts\training\run-...` directory and starts with the original untrained model. Previous runs are left untouched.

If `-Steps` is omitted, the value comes from `training.total_steps` in `training.json`.

Example using all defaults from `training.json`:

```powershell
.\train.ps1 -Mode Fresh -NoPreview
```

## Resume — continue the same run

```powershell
.\train.ps1 -Mode Resume -Checkpoint latest -Steps 50000 -NoPreview
```

For `Resume`, `-Steps` means **additional** steps. If the checkpoint is at 43,817 and you pass `-Steps 50000`, the target is 93,817.

`Resume` deliberately allows only the latest generation of that run. If you want to go back to an older checkpoint, use `Branch` instead.

You can also point at a particular run directory. A run directory resolves to its own `checkpoints\latest.json`:

```powershell
.\train.ps1 -Mode Resume `
  -Checkpoint ".\artifacts\training\run-20260912-103500-123456" `
  -Steps 50000 -NoPreview
```

## Branch — fork from an older checkpoint

```powershell
.\train.ps1 -Mode Branch `
  -Checkpoint ".\artifacts\training\run-...\checkpoints\step-000030000" `
  -Steps 70000 -NoPreview
```

This loads the chosen learned brain state but creates an entirely new `run-...` directory. The parent run is never modified.

A branch writes:

```text
branch.json
```

with the parent checkpoint path.

The branch's `checkpoints\baseline\` means **the branch starting state**. If you branch from step 30,000, that baseline already contains the 30k learned brain; it is not the original virgin fly.

Branching is the recommended way to compare changed reward/plasticity/config settings from the same learned state:

```powershell
.\train.ps1 -Mode Branch `
  -Checkpoint ".\artifacts\training\run-...\checkpoints\step-000030000" `
  -Config ".\training-experiment-B.json" `
  -Steps 30000 -NoPreview
```

### Mid-episode checkpoints

Periodic checkpoints can occur in the middle of an episode. The neural/plastic state is saved exactly, but the Minecraft world itself is not checkpointed. When resuming/branching from such a state, FlyCraft restarts the Minecraft episode and records the unfinished partial episode as `censored: true` rather than pretending it completed normally.

## Evaluate — test a checkpoint without learning

```powershell
.\train.ps1 -Mode Evaluate `
  -Checkpoint ".\artifacts\training\run-...\checkpoints\step-000050000" `
  -Episodes 20 -NoPreview
```

Evaluation:

- restores the selected checkpoint into a separate fly instance;
- freezes learning/plasticity;
- runs randomized held-out starts;
- verifies frozen weights/traces did not change;
- writes an `evaluation.json` report under a new `artifacts\training\evaluate-...` folder.

`-Episodes` overrides `training.evaluation_episodes` for that invocation.

Useful checkpoint targets include a checkpoint directory, a run directory, or a checkpoint pointer JSON:

```powershell
# Explicit numbered checkpoint
.\train.ps1 -Mode Evaluate -Checkpoint ".\artifacts\training\run-...\checkpoints\step-000050000" -Episodes 20 -NoPreview

# Latest checkpoint pointer inside a particular run
.\train.ps1 -Mode Evaluate -Checkpoint ".\artifacts\training\run-...\checkpoints\latest.json" -Episodes 20 -NoPreview

# Best checkpoint selected by automatic evaluation
.\train.ps1 -Mode Evaluate -Checkpoint ".\artifacts\training\run-...\checkpoints\best.json" -Episodes 20 -NoPreview

# Globally latest saved checkpoint
.\train.ps1 -Mode Evaluate -Checkpoint latest -Episodes 20 -NoPreview
```

## Replay — evaluate and record footage

```powershell
.\train.ps1 -Mode Replay `
  -Checkpoint ".\artifacts\training\run-...\checkpoints\step-000050000" `
  -Episodes 1
```

Replay is a frozen evaluation with recording enabled. It is useful for producing comparable video after training without relying only on whichever training episodes happened to be recorded.

A good video workflow is to replay the same progression of checkpoints, for example:

```text
baseline
10k
25k
50k
75k
100k
```

---

# 6. `train.ps1` parameters

```powershell
.\train.ps1 `
  -Mode Fresh|Resume|Branch|Evaluate|Replay `
  -Checkpoint <path-or-latest> `
  -Config <training-json> `
  -Steps <N> `
  -CheckpointEvery <N> `
  -Episodes <N> `
  -NoPreview
```

| PowerShell option | Meaning |
|---|---|
| `-Mode` | `Fresh`, `Resume`, `Branch`, `Evaluate`, or `Replay` |
| `-Checkpoint` | Checkpoint/run/pointer to load. Default: `latest` |
| `-Config` | Alternate training config JSON |
| `-Steps` | Fresh: number of training steps. Resume/Branch: **additional** steps |
| `-CheckpointEvery` | Override `checkpoint_every_steps` for this invocation |
| `-Episodes` | Override `evaluation_episodes`; especially useful for Evaluate/Replay |
| `-NoPreview` | Do not open the interactive dashboard preview window |

`train.ps1` automatically runs `scripts/prepare_training_runtime.py` before training/evaluation so the CraftGround runtime exposes genuine Minecraft block-breaking progress telemetry.

---

# 7. `training.json` reference

The current uploaded project contains this overall structure:

```json
{
  "training": {},
  "environment": {},
  "reward": {},
  "attack": {},
  "plasticity": {},
  "recording": {},
  "dashboard": {}
}
```

## `training`

| Setting | Meaning |
|---|---|
| `total_steps` | Default number of steps to add for a training invocation |
| `max_steps_per_episode` | Episode timeout in 50 ms control ticks |
| `checkpoint_every_steps` | Numbered checkpoint frequency; `0` disables periodic numbered checkpoints |
| `evaluate_every_steps` | Automatic frozen-evaluation interval; `0` disables automatic evaluation |
| `evaluation_episodes` | Number of episodes in each evaluation |
| `seed` | Training random seed |

One step is 50 ms of simulated/control time. `max_steps_per_episode: 200` therefore means a maximum of 10 simulated seconds per episode; `600` would mean 30 simulated seconds.

### Important performance note about the current config

The uploaded `training.json` currently uses:

```text
max_steps_per_episode = 200
evaluate_every_steps = 1000
evaluation_episodes = 5
record_every_episodes = 10
```

Those are relatively frequent evaluations/recordings for a long run. In the worst case, 5 evaluation episodes × 200 ticks every 1,000 training steps adds up to roughly another 1,000 neural/Minecraft ticks per 1,000 counted training steps. That can approximately double the compute devoted to the run before recording overhead.

For a long 100k run, consider deliberately choosing the evaluation/recording cadence you actually want before launching it. For example, if you mainly want periodic evidence rather than dense evaluation:

```json
"checkpoint_every_steps": 10000,
"evaluate_every_steps": 10000,
"evaluation_episodes": 5
```

Then do a larger manual `Evaluate -Episodes 20` or `50` on promising/final checkpoints afterward.

## `environment`

| Setting | Meaning |
|---|---|
| `randomize_player_position` | Randomize player X/Z each episode |
| `randomize_tree_position` | Physically place the training tree at a new randomized valid X/Z position |
| `randomize_yaw` | Randomize player starting yaw |
| `min_tree_distance` | Minimum starting player→tree distance |
| `max_tree_distance` | Maximum starting player→tree distance |
| `player_spawn_radius` | Random player X/Z range around the arena centre |

The arena check requires:

```text
player_spawn_radius + max_tree_distance <= 15
```

The project does **not** create a brand-new Minecraft world every episode. It keeps the running superflat arena, clears/rebuilds the experiment region, places the tree, teleports the player, and switches into survival mode.

The training tree currently consists of a vertical oak-log trunk with persistent oak leaves. One target log coordinate is tracked externally for progress/success telemetry.

## `reward`

| Setting | Meaning |
|---|---|
| `angle_improvement` | Positive score scale when angular error decreases |
| `angle_worsening` | Negative score scale when angular error increases |
| `distance_improvement` | Positive score scale when distance decreases |
| `distance_worsening` | Negative score scale when distance increases |
| `crosshair_on_log` | One-time score when the crosshair acquires the target log |
| `breaking_progress_gain` | Score scale for increased real block-breaking progress |
| `breaking_progress_loss` | Penalty when breaking progress falls/resets |
| `log_broken` | Large score for breaking the target log |
| `step_penalty` | Small per-step score/penalty |

Angle, distance, and breaking rewards are based on **deltas**, so the fly cannot repeatedly farm reward simply by remaining in a good state.

### Reward score vs actual learning signal

This distinction is critical:

```text
positive reward  → logged/evaluated as a training score
negative reward  → may schedule existing PPL101 aversive stimulation
```

The current DOOMFLY model does not implement the requested positive-reward plasticity pathway. FlyCraft therefore does **not** invent one. The learning mode uses DOOMFLY's existing experimental aversive LTD model and labels it experimental.

## `attack`

| Setting | Meaning |
|---|---|
| `threshold` | Neural attack accumulator threshold |
| `decay` | Accumulator decay per tick |
| `hold_steps` | How long attack remains held after activation |

Only neural attack events feed this accumulator. Tree location/progress does not script the attack decision.

## `plasticity`

Current supported model:

```text
gamma1-eligibility-ltd-v1
```

| Setting | Meaning |
|---|---|
| `model` | Must currently be `gamma1-eligibility-ltd-v1` |
| `eta` | Upstream learning-rate/plasticity parameter |
| `current` | PPL101 stimulation current used for aversive events |
| `pulse_ticks` | Length of the aversive pulse in 50 ms ticks |
| `negative_threshold` | Reward must be below `-negative_threshold` to schedule punishment |

## `recording`

| Setting | Meaning |
|---|---|
| `record_every_episodes` | Record every Nth training episode; `0` disables interval recording |
| `record_first_episode` | Always record episode 1 |
| `record_final_episode` | Record episodes in the final training budget window so the actual final episode is captured |
| `record_minecraft` | Write Minecraft `baseline.mp4` for selected episodes |
| `record_dashboard` | Write dashboard, retina, and neural-activity MP4s for selected episodes |
| `record_activity` | Write sparse 60 Hz spikes and quantized 20 Hz membrane voltage data |

Non-recorded episodes still run Minecraft RGB because the fly needs visual input, but they avoid the optional dashboard/video/high-volume activity work.

For long runs, sparse recording can make a large performance difference. For example:

```json
"record_every_episodes": 20
```

will give a useful visual progression without filming every attempt.

## `dashboard`

```json
"reward_rolling_average_episodes": 20
```

controls the rolling-average window in the training dashboard.

---

# 8. Training dashboard

The training dashboard is designed around four visual ideas:

1. Minecraft view
2. simulated retina
3. anatomical neural activity
4. neural decoder/actions + training reward

The reward portion now contains **three separate graphs**:

### Current episode — cumulative reward

A live line that resets at the beginning of every episode and shows the current episode's accumulated reward after each control tick.

```text
CURRENT EPISODE | CUMULATIVE REWARD
```

This is the best graph for seeing what is happening *inside the attempt currently on screen*.

### Completed episode reward

A point plot with one gold point per completed, non-censored training episode.

```text
COMPLETED EPISODE REWARD
```

This shows the noisy episode-to-episode distribution without implying a smooth trend between independent episodes.

### Rolling average

A cyan line showing the rolling average of completed episode rewards over the configured window.

```text
ROLLING AVERAGE | LAST 20 EPISODES
```

This is the main long-term learning/training-trend visual for timelapse footage.

The dashboard also displays current episode/step, current accumulated reward, player distance to the log, aiming error, genuine breaking progress, rolling reward, all-time mean reward, best evaluation success rate, and the neural decoder's turn/forward/attack state.

### Dashboard keyboard controls

When the normal live visualiser is open:

```text
1        Minecraft/game view
2        Retina
3        Neural activity
4        Full dashboard
Arrow    Orbit neural anatomy
Q / E    Roll neural anatomy
+ / -    Zoom
Space    Pause visual display
Esc      Stop
```

---

# 9. Checkpoints

## Location

Every training run has its own immutable checkpoint folder:

```text
artifacts\training\<run>\checkpoints\
```

Typical layout:

```text
artifacts/
└── training/
    ├── latest.json                    global latest checkpoint pointer
    │
    └── run-20260912-103500-123456/
        ├── config.json
        ├── config-<timestamp>.json
        ├── model.json
        ├── reward-history.json
        ├── metrics-<timestamp>.jsonl
        ├── branch.json                only on branched runs
        │
        └── checkpoints/
            ├── baseline/
            │   ├── brain.npz
            │   ├── decoder.npy
            │   └── state.json
            ├── step-000010000/
            ├── step-000020000/
            ├── evaluation-step-0000.../
            ├── latest-step-0000...-.../
            ├── latest.json
            └── best.json
```

## What a checkpoint stores

Each checkpoint directory contains:

### `brain.npz`

Full upstream mutable brain checkpoint data required by the learning model. This includes the learned/plastic state and neural temporal state saved by DOOMFLY's `MemoryBrain.checkpoint()`.

### `decoder.npy`

The filtered state of the existing descending-neuron decoder. This matters because the decoder itself has temporal state.

### `state.json`

Training/controller metadata including the run path, full config snapshot, global training state, history, RNG state, reinforcement state, episode counters, best evaluation tracking, and SHA-256 hashes for checkpoint integrity.

Checkpoint loading verifies the stored binary hashes before restoring them.

## Pointers

```text
artifacts\training\latest.json
```

points to the most recently saved checkpoint across runs.

```text
artifacts\training\<run>\checkpoints\latest.json
```

points to the latest checkpoint in that specific run.

```text
artifacts\training\<run>\checkpoints\best.json
```

points to the best checkpoint found by automatic evaluation. The current selection score is success rate first, then evaluation reward as the tie-breaker.

Pointer JSON files contain paths/names; the actual brain data remains in the checkpoint directory.

## Checkpoint immutability

Numbered/named checkpoint directories are immutable. Saving over an existing checkpoint label throws an error instead of silently replacing it.

---

# 10. Understanding `artifacts\training`

Think of the training artifacts as four categories: **runs**, **checkpoints**, **telemetry**, and **recordings/evaluations**.

## A training run

Every `Fresh` or `Branch` creates:

```text
artifacts\training\run-<timestamp>\
```

`Resume` continues an existing run directory rather than creating a new one.

### `config.json`

The active configuration for the latest invocation of that run.

### `config-<timestamp>.json`

An immutable timestamped snapshot of the config for each fresh/resume/branch invocation. These are useful later when reconstructing exactly how a particular run was configured.

### `model.json`

Records the experimental model identity, kernel/circuit information, initial weight hash, sensory-input statement, and the fact that positive reward is metric-only.

### `branch.json`

Exists for branched runs and identifies the parent checkpoint.

## `reward-history.json`

High-level, episode-oriented history. Each completed/censored episode carries information such as episode number, step range, total reward, starting state, success, final distance, angular error, hit rate, rolling reward, and plasticity summary.

Use this file when you want the high-level learning curve.

## `metrics-<timestamp>.jsonl`

Fine-grained training telemetry: one JSON object per training control tick.

Rows include information such as:

```text
global step
episode
instantaneous reward
episode cumulative reward
individual reward components
external Minecraft state/target metrics
action
aversive pulse state
neural telemetry
```

Every 20 training steps, a row additionally includes plasticity/memory statistics.

A new metrics file is created for each invocation/resume stamp instead of rewriting the previous file.

## `evaluation-<step>.json`

Automatic evaluation report written inside a run after a scheduled frozen evaluation.

## Standalone `evaluate-...` / `replay-...` folders

Manual:

```powershell
.\train.ps1 -Mode Evaluate ...
```

creates:

```text
artifacts\training\evaluate-<timestamp>\evaluation.json
```

Manual Replay creates a similar `replay-<timestamp>` folder plus recorded episode folders.

---

# 11. Understanding a recorded training/replay episode

Selected episodes are stored in directories such as:

```text
artifacts\training\run-...\episode-000020-step-000011691\
```

or under a standalone `replay-...` folder.

Depending on recording settings, the folder can contain:

```text
baseline.mp4                    Minecraft RGB footage
dashboard.mp4                   full 1080p dashboard
retina.mp4                      retina visualisation
neural-activity.mp4             anatomical activity visualisation

episode.json                   episode summary
reward-history.json             completed reward history available when recording started
steps-00000.jsonl.gz            per-tick controller/game/training telemetry

activity-60hz.jsonl.gz          sparse spike counts by MaleCNS body ID
voltage-20hz/                   quantised membrane-voltage chunks
neuron-layout.npz               recorded anatomical mapping
anatomy.json                    geometry metadata
skeleton-*.npz                  selected descending-neuron skeletons
visuals.json                    visualiser metadata

minecraft-trajectory.jsonl.gz   exact 20 Hz pose/action stream
minecraft-trajectory-meta.json  trajectory format/config metadata
```

### `episode.json`

Open this first when inspecting a recorded episode. It is the compact summary of what happened in that attempt.

### `steps-00000.jsonl.gz`

Contains controller, action, before/after state, training-panel numbers and neural telemetry for the recorded episode.

### `activity-60hz.jsonl.gz`

Sparse neural spike activity. Each frame stores only active neurons as `[body_id, spike_count]` pairs.

### `voltage-20hz/`

Visualization-only membrane-potential recording. One quantized `uint8` value per neuron per 50 ms control tick, chunked/compressed on disk. The actual neural simulation still uses its full-precision state.

### `minecraft-trajectory.jsonl.gz`

One pose/action row per 20 Hz Minecraft tick plus an initial pose. This is intended for later cinematic/fake-player replay work and is independent of neural activity recording.

---

# 12. Offline neural-activity playback

For a recording that contains `activity-60hz.jsonl.gz`:

```powershell
.\play-activity.ps1 ".\artifacts\...\activity-60hz.jsonl.gz"
```

You can also pass the recording directory:

```powershell
.\play-activity.ps1 ".\artifacts\...\episode-000020-step-000011691"
```

Select activity mode:

```powershell
.\play-activity.ps1 <recording> -ActivityMode Spikes
.\play-activity.ps1 <recording> -ActivityMode Voltage
.\play-activity.ps1 <recording> -ActivityMode Combined
```

Display-only voltage smoothing:

```powershell
.\play-activity.ps1 <recording> -VoltageSmoothingMs 80
```

Older recordings without `voltage-20hz` automatically fall back to spike-only view.

### Offline player controls

```text
Space              Play/pause
Click/drag timeline Seek
Left/Right          ±1 second
Shift+Left/Right    ±5 seconds
, / .               Previous/next frame
[ / ]               Playback speed
Mouse left-drag     Rotate anatomy
Mouse right-drag    Roll anatomy
Mouse wheel         Zoom
W/A/S/D             Pitch/yaw anatomy
Q/E                 Roll anatomy
1                   Spikes
2                   Voltage
3                   Combined
R                   Reset camera
Home/End            Start/end
Esc                 Close
```

The first use builds a disk-backed seek cache in `.activity-player-cache` beside the recording.

---

# 13. Automatic training evaluation and `best`

When `evaluate_every_steps` is nonzero, training periodically saves an `evaluation-step-...` checkpoint and runs frozen randomized evaluation episodes.

The report includes aggregate and per-episode values including:

```text
success_rate
reward
mean_distance
final_distance
angular_error
target_hit_rate
target_break_rate
seconds
frozen_verified
```

The run's `checkpoints\best.json` is updated when the new evaluation score is better. Current comparison order:

```text
1. success rate
2. reward as tie-breaker
```

For a serious final comparison, it is usually better to manually evaluate selected checkpoints with more episodes after training:

```powershell
.\train.ps1 -Mode Evaluate -Checkpoint ".\...\step-000050000" -Episodes 50 -NoPreview
```

---

# 14. Training tests

## Controlled neural/training infrastructure tests

These do not perform a long Minecraft training run:

```powershell
. .\env.ps1
& $python scripts\test_training.py
```

They check the upstream experimental LTD behavior, original integration consistency, positive reward producing no invented pulse, PPL101 aversive stimulation, plastic-state change, frozen-state preservation, exact checkpoint continuation, reward deltas, sustained attack, selective recording, random starts, and history round-trip.

Results are written under:

```text
artifacts\training-proof\
```

## Short live Minecraft training-infrastructure smoke test

```powershell
. .\env.ps1
& $python scripts\test_training_live.py
```

This is a deliberately tiny end-to-end test that checks genuine Minecraft breaking progress/reset, sustained attack, short randomized training, selective recording, resume, branch, frozen evaluation, replay, offline playback, and 1080p60 video output.

Do not confuse either test with a meaningful long learning experiment.

---

# 15. Practical long-run workflow

A sensible workflow for a real experiment is:

```text
1. Run controlled tests.
2. Run a short 1k–5k training sanity check.
3. Inspect reward-history, metrics, checkpoints, and one recorded episode.
4. Confirm wall-clock steps/second.
5. Choose evaluation and recording cadence deliberately.
6. Start the long Fresh run with -NoPreview.
7. Let periodic checkpoints protect progress.
8. If interrupted, Resume latest.
9. After training, Evaluate selected/best checkpoints with a larger sample.
10. Replay chosen checkpoints to generate clean comparison footage.
```

Example long run:

```powershell
.\train.ps1 -Mode Fresh -Steps 100000 -CheckpointEvery 10000 -NoPreview
```

Use `Ctrl+C` to stop cleanly. The `finally` path saves a new latest checkpoint before closing the Minecraft environment.

Avoid force-killing the terminal if possible, especially while recording video, because encoders/checkpoint writers need to finalize cleanly.

---

# 16. Performance and recording

Training speed is strongly affected by optional visualisation/video work.

`-NoPreview` prevents the interactive dashboard window. For training episodes not selected by `record_every_episodes`, the project also avoids the expensive dashboard/video/raw-activity recording path.

Recorded episodes still need to render/write whatever recording types are enabled in `training.json`, even when `-NoPreview` is used.

For a long run, the two largest knobs to watch are:

```text
record_every_episodes
evaluate_every_steps × evaluation_episodes
```

A sparse video cadence plus manual post-training Replay usually gives better footage per unit of compute than recording every training attempt.

---

# 17. Scientific/model caveats

This project is an experiment built around a reconstructed connectome and simplified neural dynamics. The following distinctions should remain explicit in any interpretation/video:

- The MaleCNS wiring/connectome is reconstructed biological anatomy, but the simulated physiology is simplified.
- The Minecraft visual input uses the existing retinal sampling model; it is not a literal claim about the fly's subjective visual experience.
- The descending-neuron → Minecraft control mapping is engineered.
- Minecraft target position, angle, distance, raycast, and breaking progress are external trainer/evaluation telemetry, not fly sensory input.
- Training currently uses DOOMFLY's existing **experimental aversive LTD** mechanism.
- Positive rewards are training/evaluation metrics only; they do not directly modify weights in this implementation.
- Changed plastic weights do not by themselves prove useful learning. Improvement should be judged through frozen randomized evaluation.
- Replay/evaluation freezes plasticity so checkpoint comparisons do not alter the states being compared.

---

# 18. Troubleshooting

## PowerShell refuses to run scripts

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then rerun the command.

## `Run .\setup.ps1 first.`

The project-local Python environment is missing. Run:

```powershell
.\setup.ps1
```

## `Missing genuine breaking telemetry`

Training requires the small CraftGround runtime patch that exposes actual block-breaking progress. `train.ps1` normally prepares it automatically before launching. You can run it manually:

```powershell
. .\env.ps1
& $python scripts\prepare_training_runtime.py
```

Then restart Minecraft/the training command so Gradle rebuilds the changed runtime code.

## Want to know exactly which checkpoint `latest` means

```powershell
Get-Content .\artifacts\training\latest.json
```

For a particular run:

```powershell
Get-Content ".\artifacts\training\run-...\checkpoints\latest.json"
```

## Want the best checkpoint selected by evaluation

```powershell
Get-Content ".\artifacts\training\run-...\checkpoints\best.json"
```

Then evaluate it directly:

```powershell
.\train.ps1 -Mode Evaluate `
  -Checkpoint ".\artifacts\training\run-...\checkpoints\best.json" `
  -Episodes 20 -NoPreview
```

## Want to start over without deleting old data

Just run `Fresh` again:

```powershell
.\train.ps1 -Mode Fresh -Steps 100000 -NoPreview
```

A new timestamped run directory is created; old runs remain intact.

---

# 19. Source/version lock

The project records its expected upstream versions and original neural weight hash in:

```text
model-lock.json
```

The uploaded version pins:

```text
DOOMFLY commit:   71ecf53d78eaffaf1a57ed7b0ccf5d458abc9f33
CraftGround repo: 18eba01a87a8481bc4fbb54b42fee1428c0c3719
CraftGround pkg:  2.7.4
MaleCNS neurons:  166,700
Directed edges:   25,582,938
```

The baseline verifies the original fixed weights remain unchanged. Training uses a separate experimental learning model path and saves its mutable state through checkpoints.


### Minecraft window aspect ratio

The fly capture is configured as **640×480 (4:3)**. `scripts/prepare_runtime.py` also patches CraftGround's launcher so the visible Minecraft client starts at the same 640×480 aspect ratio instead of Minecraft's default 854×480-style 16:9 window. This affects only the human-facing client window; the fly continues to receive the configured 640×480 RGB observation.

After updating an existing checkout with this patch, apply the runtime launcher change once before the next run:

```powershell
. .\env.ps1
& $python scripts\prepare_runtime.py
```

Close any already-running Minecraft client before relaunching.


### Activity mode during training

`train.ps1` accepts the same neural visualization modes as the baseline viewer:

```powershell
.\train.ps1 -Mode Fresh -Steps 1000 -ActivityMode Spikes
.\train.ps1 -Mode Fresh -Steps 1000 -ActivityMode Voltage
.\train.ps1 -Mode Fresh -Steps 1000 -ActivityMode Combined -VoltageSmoothingMs 80
```

`Combined` renders the smoothed cyan/purple membrane-voltage layer together with the gold spike layer in live previews and recorded training dashboards. It is display-only: it does not alter neural state or learning. With `-NoPreview`, the setting still applies to episodes for which `record_dashboard` is enabled.

