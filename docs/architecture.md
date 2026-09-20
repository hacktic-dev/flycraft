# Architecture — how FlyCraft works

FlyCraft is a thin, carefully-bounded bridge between two large systems:

- **DOOMFLY / MaleCNS** — a reconstructed fruit-fly connectome (166,700 neurons,
  25,582,938 retained directed edges) with a native C++ integration kernel and a
  simplified leaky-integrate-and-fire physiology.
- **Minecraft 1.21 via CraftGround** — a headless-capable Minecraft client that
  returns RGB frames and accepts human-style actions.

FlyCraft's job is to move **only pixels** into the brain and **only decoded
neural activity** out to the game. Everything else — reward, target geometry,
success detection, video — is external telemetry that observes the loop but is
never injected into it.

---

## 1. The control loop

One iteration is one **control tick**:

```text
 1. Capture Minecraft RGB (16:9, e.g. 854x480)
 2. Center-crop to the configured 640x480 4:3 neural view
 3. retinal_samples(rgb, brain.uv)  -> per-photoreceptor luminance (+ R8 colour in learning mode)
 4. Advance the native MaleCNS simulation by exactly 50 ms of neural time
 5. Decode descending-neuron firing rates -> turn / forward / attack
 6. Send the action to Minecraft; advance the game one tick
 7. (optional) compute reward, log telemetry, record video
```

20 control ticks = 1 simulated second. `max_steps_per_episode: 400` therefore
means a maximum of 20 simulated seconds per training episode.

The native integration itself runs at a finer internal timestep (0.5 ms in the
learning path), but a *control* tick always advances exactly 50 ms. Spike bins
for recording are captured at 60 Hz (three bins per control tick).

### Aspect ratio and the neural crop

CraftGround/Minecraft 1.21 on Windows renders the scene against a 16:9 client
framebuffer. Asking it directly for 640x480 produces a 16:9 image horizontally
compressed into 4:3 pixels. FlyCraft instead:

- captures at the matching 16:9 width (`max(width, ceil(height*16/9))`),
- takes an undistorted **centered 4:3 crop** for retinal sampling and the neural
  dashboard, and
- keeps the **uncropped 16:9 frame** for human-facing previews and video.

The crop is geometric only; it adds no game-state information. See
`game.pixels` and `game.preview_pixels` in `src/flycraft/game.py`.

![The whole loop in one frame](images/baseline-dashboard.png)

*One dashboard frame of the whole loop: Minecraft RGB (top left), the retinal
samples it becomes (top right), the MaleCNS anatomy lighting up (bottom left),
and the decoded descending-neuron rates (bottom right).*

---

## 2. Source map

All runtime code lives in `src/flycraft/`.

| Module | Responsibility |
|---|---|
| `brain.py` | Frozen bridge. Loads the full graph, verifies the original weight hash, runs retinal sampling + one neural step, decodes control. |
| `game.py` | CraftGround boundary: arena setup, RGB capture/crop, telemetry, action mapping. |
| `decoder.py` | Legacy BCI and balanced-bilateral decoders, calibration math, checkpoint signature. |
| `learning.py` | `LearningFly`: the signed/fatigue-selective plasticity brain plus optional action credit and aversive pulses. |
| `r8_visual.py` | DOOMFLY v6 RGB/R8 colour visual adapter layered on the learning brain. |
| `action_credit.py` | Experimental action-conditioned steering credit on KC→MBON11 edges. |
| `run.py` | Frozen-baseline runner. |
| `train.py` | Signed-reinforcement training (fresh/resume/branch/evaluate/replay). |
| `training_game.py` | Episode setup, randomized starts, target/breaking telemetry. |
| `training_metrics.py` | Reward components, balanced teaching signal, attack accumulator, behaviour stats. |
| `training_checkpoints.py` | Immutable checkpoint save/load with hashes and atomic pointers. |
| `training_recording.py` | Selective per-episode video/activity/trajectory recording. |
| `readout_learning.py` | Frozen-connectome supervised readout: neural feature encoder + MLP + DAgger curriculum. |
| `imitation_train.py` | Training/evaluation driver for the supervised readout. |
| `visuals.py` | Read-only dashboard, anatomical activity rendering, video writers, playback. |
| `activity.py` | Spike observer, sparse 60 Hz spike writer, quantized 20 Hz voltage writer/reader. |
| `activity_player.py` | Interactive offline player for recorded activity. |
| `trajectory.py` | Exact 20 Hz Minecraft pose/action trace writer. |
| `anatomy.py` | Fetches published MaleCNS soma/skeleton positions (never inferred from graph layout). |
| `fast_kernel.py` | Exact native specialisation of the common one-timestep integration case. |
| `mirrored_starts.py` | Left/right mirrored start geometry for decoder calibration. |
| `validate_teaching.py` | Dependency-light symmetry check for the balanced teaching transform. |

---

## 3. The neural simulation

`FrozenFly` (`brain.py`) and `LearningFly` (`learning.py`) both wrap the same
upstream native kernel:

- `FrozenFly` freezes `brain.weight` (read-only) and asserts
  `n == 166700` and `len(weight) == 25582938`.
- `LearningFly` uses DOOMFLY's experimental learning model and applies
  FlyCraft's plasticity rule outside the kernel.

### Visual input

- **Baseline**: `doom.game.retinal_samples` maps the 640x480 crop to R1-R6
  luminance photoreceptors only.
- **Learning**: `R8VisualMemoryBrain` adds DOOMFLY v6's inferred **R8 colour**
  channels — R8p from linear-sRGB blue, R8y from linear-sRGB green — and forces
  existing R8→aMe12 contacts excitatory while keeping their magnitudes. RGB is
  advanced in ≤10 ms neural chunks, matching the v6 adapter. See
  [visual-input.md](visual-input.md).

### Timing and exactness

- The native kernel is built by `scripts/build_kernel.py` from the pinned
  `vendor/doomfly/doom/kernel.cpp`.
- `fast_kernel.py` compiles a byte-for-byte-equivalent specialisation of the
  common "elapsed one timestep" integration path. It changes no arithmetic order
  and no checkpoint ABI; it is verified by `scripts/test_speed_exact.py`.
- `FLYCRAFT_REFERENCE_KERNEL=1` disables the specialisation for diagnostics.

---

## 4. Decoding neural activity into actions

There are two steering decoders, selected by the `decoder` block of a config:

### Legacy BCI (default when no `decoder` block is present)

`doom.engine.NeuralControls(mode='bci')` maps annotated readout cells:

```text
DNp20 L/R rate difference → camera yaw
DNpe017 combined rate      → forward key
DNpe017 spikes             → attack pulse
```

`map_action` multiplies the legacy `turn` value by `yaw_gain` (negative in
`config/baseline.json`).

### Balanced bilateral (v1 / v2 / v3)

An engineered, calibratable opponent decoder. It keeps the existing 100 ms rate
filter and forward/attack mappings but replaces steering with a normalized
left/right opponent signal from a chosen neuron pair (DNp20, DNa02 or MBON11).
Full details, calibration and limitations are in [decoder.md](decoder.md).

### Attack persistence

Raw attack is a spike pulse. FlyCraft adds a neural-only `SustainedAttack`
accumulator so a pulse can hold left-mouse long enough to break a block. Only
neural attack events feed it; target location/progress never script the attack
decision.

---

## 5. Reward, teaching and learning rules

FlyCraft deliberately separates three things that are easy to conflate.

### Reward (evaluation metric only)

`training_metrics.reward_components` computes an episode score from external
telemetry: angle improvement/worsening, distance improvement/worsening,
crosshair acquisition, genuine breaking-progress gain/loss, the terminal log
break, and a small per-step penalty. This score drives the dashboard and
evaluation. It is **not** delivered to the fly.

### Teaching (the plasticity signal)

`training_metrics.teaching_signal` builds a **balanced raw-delta** signal that
is independent of the (intentionally asymmetric) reward weights. Equal-and-
opposite physical changes in aim, distance or breaking progress produce
equal-and-opposite teaching. Components are summed, squashed with `tanh`, and
values below a deadband are treated as neutral. A successful log break is
positive-only (it is the terminal success event).

### Learning rules

1. **Run-2D fatigue-selective signed plasticity** (`gamma1-fatigue-selective-signed-v4`)
   — the default `train.ps1` rule. Sparse, transient KC "winners" above a
   per-KC baseline are selected (winner fraction, KC cap, activity floor), with
   a soft decaying ranking penalty ("fatigue") to encourage turnover. Positive
   teaching potentiates eligible KC→MBON11 edges, negative teaching depresses
   them. Updates are bounded (log-domain per-event cap) and weights are limited
   to 0.5×–1.5× baseline, slowly relaxing toward 1.0× baseline. Strong negative
   teaching can additionally schedule a short **PPL101 aversive** current pulse.

2. **Action-conditioned steering credit** (`action-conditioned-steering-v2`,
   optional, `plasticity.action_credit`) — eligibility from the same sparse KC
   winners is tagged by the left/right yaw actually expressed. An aim
   reward-prediction error then strengthens edges whose postsynaptic MBON
   channel supports the action that just succeeded, with per-channel
   homeostatic renormalisation. This is explicitly an **engineered** rule, not a
   claimed reconstructed motor-learning circuit. It needs a bilateral MBON11 (or
   calibrated-downstream) decoder.

3. **Frozen-connectome supervised readout** (`frozen-connectome-readout-v1`) —
   a completely separate path used by `train-readout.ps1`. The connectome is
   frozen; a small MLP learns to translate neural spike features into
   yaw/forward/attack using a scripted expert's labels. See
   [readout-learning.md](readout-learning.md).

### Plasticity health

Training periodically logs a `PLASTICITY OK/WARNING/CRITICAL/COLLAPSE` line with
the distribution of plastic-edge efficacy, credit exposure and update events.
`abort_on_collapse` stops the run if the plastic population saturates or drifts
past the configured thresholds.

---

## 6. Recording and visualisation

Recording is read-only and never changes simulation state.

- **Dashboard / retina / neural-activity video**: 1080p60 via `visuals.py`.
  Game and retina frames are held for three 60 Hz output frames per 50 ms tick.
- **Spikes**: `activity-60hz.jsonl.gz`, sparse `[body_id, spike_count]` per
  active neuron.
- **Voltage**: `voltage-20hz/`, one quantized `uint8` per neuron per control
  tick (visualisation only; the simulation keeps full precision).
- **Trajectory**: `minecraft-trajectory.jsonl.gz`, exact 20 Hz pose/action.
- **Per-tick telemetry**: `steps-*.jsonl.gz`.

Anatomical positions come from the published MaleCNS EM dataset
(`anatomy.py`), never from graph-layout inference. Missing positions are left
missing rather than fabricated.

---

## 7. What is verified

- `scripts/test_brain.py` — full-connectome native sanity test.
- `scripts/test_game.py` — live CraftGround turn/walk/attack/reset.
- `scripts/verify_baseline.py` — audits a baseline run's closed-loop evidence
  (action mapping, movement, attack, RGB→retina hashes, unchanged weights).
- `scripts/test_speed_exact.py` — bit-exact regression of the native
  specialisation.
- `scripts/test_training.py` / `test_training_live.py` — training infrastructure.
- `scripts/test_decoder.py`, `test_action_credit.py`,
  `test_readout_learning.py`, `test_bilateral_decoder_v2/v3.py`,
  `test_mirrored_calibration_v4.py` — dependency-light unit tests.

See [scripts.md](scripts.md) for the complete script reference.
