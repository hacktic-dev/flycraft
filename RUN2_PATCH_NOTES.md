# FlyCraft Run 2C — selective plasticity + balanced teaching

## Current Run-2C changes

Run 2C supersedes the Run-2B description below. It uses
`gamma1-selective-signed-v3`: a 120 ms activity trace relative to a 1,500 ms
per-KC baseline, with competitive credit limited by a 25% winner fraction and
256-KC cap. Balanced raw-delta teaching remains separate from task scoring.
Mean-efficacy drift now participates in health alarms, and collapse aborts are
enabled by default. Frozen evaluation preserves the new credit baseline and age.

The dashboard shows signed teaching and PPL101 events alongside a separate
completed-episode performance curve and rolling average. The earlier component
bar dashboard description below is historical.

Validation: `python -m flycraft.validate_teaching` and
`python -m doom_learning.validate_flycraft_v3` pass locally. The latter and the
v3 learning implementation are local changes in the separate, ignored
`vendor/doomfly` repository; this FlyCraft commit does not package them. A fresh
checkout requires those DOOMFLY patches as well.

## Historical Run-2B notes

This is the updated Run-2 patch. It keeps the reversible KC→MBON11 plasticity introduced for Run 2, but fixes the positive bias found in the first 2,000-step diagnostic and redesigns the dashboard around signals the fly actually receives.

## 1. Plasticity remains reversible

The underlying Run-2 rule is unchanged:

- recent KC activity uses a 300 ms credit/eligibility trace;
- positive teaching potentiates eligible KC→MBON11 edges;
- negative teaching depresses eligible KC→MBON11 edges;
- updates are bounded and occur once per 50 ms control outcome;
- weights are limited to 0.5×–1.5× their reconstructed baseline;
- weights slowly relax toward 1.0× baseline with a 1,800 s simulated-time constant;
- fast neural state, decoder state and eligibility are reset between Minecraft episodes while learned weights are preserved;
- sufficiently strong negative teaching can additionally schedule the separate short PPL101 aversive neural pulse.

Positive teaching is an explicit experimental FlyCraft signal. It is **not** claimed to be a reconstructed appetitive DAN pathway.

## 2. Reward and teaching are now fully separate

The first Run-2 patch derived teaching from the already weighted reward components. That accidentally inherited the reward asymmetry:

- angle reward: +0.05 improvement / -0.025 worsening;
- distance reward: +0.10 improvement / -0.05 worsening;
- break-progress reward: +2.0 gain / -1.0 loss.

This meant equal-and-opposite physical behavior did not cancel neurally. Run 2B fixes this.

The existing `reward` section is still used for episode scoring/evaluation and is unchanged. Plasticity now uses a separate `teaching` section and starts from raw physical changes:

```json
"teaching": {
  "model": "balanced-raw-delta-v1",
  "angle_scale_deg": 15.0,
  "distance_scale_blocks": 0.25,
  "progress_scale": 0.1,
  "angle_weight": 0.25,
  "distance_weight": 0.35,
  "progress_weight": 0.75,
  "crosshair_weight": 0.2,
  "success_weight": 1.0,
  "deadband": 0.03
}
```

For aim, distance, breaking progress and crosshair acquisition/loss, equal-and-opposite physical changes now produce equal-and-opposite teaching components. The components are summed, passed through `tanh`, and values smaller than the deadband are treated as neutral.

A successful log break remains intentionally positive-only because it is the terminal task success event. Crosshair disappearance caused by the log being successfully broken is not counted as a negative crosshair-loss event.

The generic per-step time penalty remains score-only and never teaches the fly.

## 3. Console output now exposes teaching balance

The 20-step status line includes per-episode teaching-event counts:

```text
Teach +0.13 | TeachEv +28/-35/057
```

This lets you immediately see whether positive, negative and neutral outcomes are all occurring instead of inferring that from occasional sampled `Teach` values.

The existing plasticity-health alarm remains unchanged and still reports every 100 steps.

## 4. Dashboard now shows what the fly actually experiences

The large `REWARD` display and reward-history graphs have been removed from the training dashboard because the reward scalar is an external evaluation metric and is not delivered to the fly.

The lower dashboard now shows:

- the exact signed KC→MBON11 teaching value on the current control tick;
- whether it is POSITIVE, NEGATIVE or NEUTRAL;
- how many plastic edges are currently eligible enough to be updated;
- separate AIM, DIST, BREAK, XHAIR and SUCCESS teaching-component bars;
- the raw physical deltas that caused those components;
- a full per-episode signed teaching-event timeline;
- green spikes for positive plastic teaching;
- red spikes for negative plastic teaching;
- purple markers for actual PPL101 aversive stimulation;
- a separate indication when a PPL101 pulse has only been scheduled for the next tick;
- cumulative positive / negative / neutral teaching-event counts for the episode.

Frozen evaluation/replay explicitly displays that teaching is off.

The neural-activity view label has also been changed from the obsolete `AVERSIVE LTD` wording to `SIGNED REVERSIBLE (EXPERIMENTAL)`.

## 5. Validation

A dependency-light validator was added:

```powershell
python -m flycraft.validate_teaching
```

It verifies exact positive/negative symmetry for isolated aim, distance, breaking-progress and crosshair changes. Expected output begins:

```text
BALANCED TEACHING CHECK: PASS
```

The existing plasticity-rule smoke test still passes:

```powershell
python -m doom_learning.validate_flycraft_v2
```

In the packaged source, 500 consecutive synthetic negative teaching events leave the plastic population at mean ~0.885× baseline with 0% at the floor, and 500 matched positive events restore it to 1.000×.

## 6. Recommended next run

Because the teaching policy has changed, start **Fresh** rather than resuming the first 2,000-step diagnostic:

```powershell
.\train.ps1 -Mode Fresh -Steps 2000 -CheckpointEvery 1000 -NoPreview
```

Watch both:

```text
TeachEv +.../-.../0...
```

and:

```text
PLASTICITY OK | ...
```

For this diagnostic, the main goals are:

- both positive and negative teaching events occur;
- the mean efficacy stays near 1.0 rather than steadily marching upward or downward;
- p10/median/p90 begin to develop some spread rather than all 4,184 edges moving identically;
- floor/ceiling saturation remains essentially 0%;
- broad drift outside 0.75×–1.25× remains near 0%.

## Files changed in Run 2B

Compared with the previous Run-2 patch:

- `training.json`
- `src/flycraft/training_metrics.py`
- `src/flycraft/train.py`
- `src/flycraft/visuals.py`
- `src/flycraft/validate_teaching.py` (new)

The DOOMFLY/underlying reversible-plasticity files are unchanged from the previous Run-2 patch.
