# Checkpoints and artifacts

All run output lives under `artifacts/`, which is git-ignored.

---

## 1. Checkpoints

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

### What a checkpoint stores

- **`brain.npz`** — full upstream mutable brain state required by the learning
  model, including learned/plastic state and neural temporal state.
- **`decoder.npy`** — the filtered state of the descending-neuron decoder
  (the decoder itself has temporal state).
- **`state.json`** — run path, full config snapshot, global training state,
  history, RNG state, reinforcement state, episode counters, best-evaluation
  tracking, and SHA-256 hashes for integrity.

Checkpoint loading verifies the stored binary hashes before restoring.

### Pointers

| Pointer | Meaning |
|---|---|
| `artifacts\training\latest.json` | Most recently saved checkpoint across all runs. |
| `artifacts\training\<run>\checkpoints\latest.json` | Latest checkpoint in that run. |
| `artifacts\training\<run>\checkpoints\best.json` | Best checkpoint by automatic evaluation. |

Pointer JSON files contain paths/names; the brain data stays in the checkpoint
directory. Selection score is success rate first, evaluation reward as the
tie-breaker.

### Immutability

Numbered/named checkpoint directories are immutable: saving over an existing
label throws instead of silently replacing it.

### Decoder compatibility

New checkpoints record the decoder configuration and filter rates. `Resume`
rejects decoder changes; use an explicit `Branch` or `Evaluate` override to
compare decoders. Old checkpoints without a decoder block retain legacy BCI
semantics.

---

## 2. `artifacts\training` — runs, checkpoints, telemetry, recordings

`Fresh` and `Branch` create `artifacts\training\run-<timestamp>\`; `Resume`
continues an existing run directory.

### Run files

| File | Meaning |
|---|---|
| `config.json` | Active configuration for the latest invocation. |
| `config-<timestamp>.json` | Immutable snapshot per fresh/resume/branch invocation. |
| `model.json` | Experimental model identity, kernel/circuit info, initial weight hash, visual model, decoder signature, and the learning description. |
| `branch.json` | Present on branched runs; identifies the parent checkpoint. |

### `reward-history.json`

Episode-oriented history. Each completed/censored episode carries episode
number, step range, total reward, mean cumulative reward, starting state,
success, final distance, angular error, hit rate, rolling reward, teaching
counts, plasticity summary and behaviour metrics.

### `metrics-<timestamp>.jsonl`

Fine-grained telemetry: one JSON object per control tick, including global step,
episode, instantaneous/episode reward, reward components, external Minecraft
state, action, aversive state, teaching signal/components, behaviour metrics and
neural telemetry. Every 20 steps a row also includes plasticity/memory
statistics. A new metrics file is created per invocation/resume stamp.

### `evaluation-<step>.json`

Automatic evaluation report written inside a run after a scheduled frozen
evaluation.

### Standalone `evaluate-...` / `replay-...` folders

`train.ps1 -Mode Evaluate` creates
`artifacts\training\evaluate-<timestamp>\evaluation.json`; `Replay` creates a
`replay-<timestamp>` folder plus recorded episode folders.

### Automatic evaluation and `best`

When `evaluate_every_steps` is nonzero, training periodically saves an
`evaluation-step-...` checkpoint and runs frozen randomized evaluation. Reports
include `success_rate`, `reward`, `mean_distance`, `final_distance`,
`angular_error`, `target_hit_rate`, `target_break_rate`, `seconds`,
`frozen_verified`, and behaviour metrics. `checkpoints\best.json` is updated when
the score improves.

For a serious final comparison, manually evaluate selected checkpoints with more
episodes after training.

---

## 3. A recorded episode

Selected episodes are stored in directories such as:

```text
artifacts\training\run-...\episode-000020-step-000011691\
```

or under a standalone `replay-...` folder. Depending on recording settings:

```text
baseline.mp4                    Minecraft RGB footage
dashboard.mp4                   full 1080p dashboard
retina.mp4                      retina visualisation
neural-activity.mp4             anatomical activity visualisation

episode.json                    episode summary
reward-history.json             completed history available when recording started
steps-00000.jsonl.gz            per-tick controller/game/training telemetry

activity-60hz.jsonl.gz          sparse spike counts by MaleCNS body ID
voltage-20hz/                   quantized membrane-voltage chunks
neuron-layout.npz               recorded anatomical mapping
anatomy.json                    geometry metadata
skeleton-*.npz                  selected descending-neuron skeletons
visuals.json                    visualiser metadata

minecraft-trajectory.jsonl.gz   exact 20 Hz pose/action stream
minecraft-trajectory-meta.json  trajectory format/config metadata
```

- **`episode.json`** — open this first; the compact summary of the attempt.
- **`activity-60hz.jsonl.gz`** — each frame stores only active neurons as
  `[body_id, spike_count]`.
- **`voltage-20hz/`** — one quantized `uint8` per neuron per control tick,
  chunked/compressed. Visualisation only; the simulation keeps full precision.
- **`minecraft-trajectory.jsonl.gz`** — one pose/action row per 20 Hz tick plus
  an initial pose, for later cinematic replay.

Play a recording back with `.\play-activity.ps1 <folder>` (see
[running-modes.md](running-modes.md)).

---

## 4. Readout-training artifacts

`train-readout.ps1` writes to:

```text
artifacts\readout-training\
├── latest.json
└── run-...\
    ├── model.json
    ├── config.json
    ├── state.json
    ├── reward-history.json
    ├── readout-latest.npz          the learned readout (weights + optimizer + encoder EMA)
    ├── episode-...\                recorded episodes
    └── autonomous-probes\          optional student-only probe footage
```

The learned object is `readout-latest.npz`. The full connectome is rebuilt from
the fixed config and is never checkpointed as learned state; every episode
checks the connectome weight array is bit-identical to its starting value.

---

## 5. Other artifact folders

| Folder | Written by |
|---|---|
| `artifacts\baseline-*` | `run-baseline.ps1` |
| `artifacts\neural-proof.json` | `scripts/test_brain.py` |
| `artifacts\craftground-proof\` | `scripts/test_game.py` |
| `artifacts\training-proof\` | `scripts/test_training.py`, `test_training_live.py` |
| `artifacts\speed\` | `scripts/benchmark_speed.py`, `test_speed_exact.py` |
| `artifacts\decoder-calibration\` | decoder calibration/validation scripts |
| `artifacts\weight-effect-2d\` | `scripts/compare_weight_effect.py` |
| `artifacts\flycraft-film-datapack\` | `scripts/film_arena_step_by_step.py` |
| `artifacts\interpreter-quality\` | explainer animation exports |
