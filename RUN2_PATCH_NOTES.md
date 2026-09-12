# FlyCraft Run 2 — reversible signed plasticity

## What changed

Run 1 used one-way PPL101-gated LTD. Repeated punishment could hit every eligible KC→MBON11 edge many times per PPL101 spike, eligibility persisted for 1 s, punishment pulses could be continually extended, positive reward did not alter plasticity, and weights could only move down to 0.1× baseline. The result was catastrophic saturation.

Run 2 keeps the same connectome, KC→MBON11 plastic-edge selection, Minecraft sensory path, R8 visual adapter and DN decoder, but replaces the plastic update with an explicitly experimental signed rule:

- recent KC activity is tracked with a 300 ms eligibility trace;
- weak trace below 1 Hz-equivalent is ignored;
- positive task outcomes potentiate eligible edges;
- negative task outcomes depress eligible edges;
- each teaching event is bounded and occurs once per 50 ms control outcome, not once per PPL101 spike;
- weights are bounded to 0.5×–1.5× their reconstructed baseline;
- weights slowly relax toward 1.0× baseline with a 1,800 s simulated-time constant;
- fast neural state, decoder state and eligibility are reset between Minecraft episodes while learned weights are preserved.

The positive signal is an external experimental teaching gate derived from Minecraft outcomes. It is **not** presented as a reconstructed appetitive DAN pathway. PPL101 remains only as a milder aversive neural perturbation for sufficiently strong negative outcomes.

## Teaching signal

The normal reward score is unchanged for evaluation. For plasticity, `step_penalty` is excluded. Angle/distance improvement, crosshair acquisition, breaking-progress gain and success contribute positive teaching; the corresponding worsenings/loss contribute negative teaching. A small deadband suppresses jitter, then the signal is smoothly bounded to [-1,+1].

Default teaching settings:

- deadband: 0.002 reward units
- scale: 0.02 reward units
- positive gain: 1.0
- negative gain: 1.0

## Aversive pulse

PPL101 stimulation no longer directly performs the weight update. It is a neural perturbation scheduled only when the signed teaching signal is <= -0.35.

Defaults:

- current: +8 (was +30)
- duration: 2 control ticks = 100 ms (was 4 = 200 ms)
- cooldown: 4 control ticks = 200 ms
- a new negative event cannot extend a pulse that is already active.

## Plasticity alarm

Every 100 training steps PowerShell prints a line such as:

`PLASTICITY OK | step 1,000 | mean 0.997 | p10/med/p90 ... | floor 0.0% | ceiling 0.0% | saturated 0.0% | outside 0.75-1.25 0.0%`

It escalates to `WARNING`, `CRITICAL`, or `COLLAPSE` when too many edges reach the hard bounds or drift outside 0.75×–1.25× baseline. `abort_on_collapse` defaults to false, so it warns loudly without killing a run. Set it to true if automatic termination is preferred.

## Recommended first launch

1. Apply both the FlyCraft and DOOMFLY changes.
2. Delete FlyCraft's `.tools/learning-kernel` and `.tools/fast-learning-kernel` directories so the Windows DLLs are rebuilt from the patched kernel source.
3. In the DOOMFLY Python environment, run:

   `python -m doom_learning.validate_flycraft_v2`

   Expected: `Run-2 plasticity smoke test: PASS`.
4. Start a **fresh**, short run first, not a resume/branch from Run 1:

   `./train.ps1 -Mode Fresh -Steps 2000 -CheckpointEvery 1000 -NoPreview`

5. Check the `PLASTICITY ...` lines at steps 100, 200, ... before committing to a long run. A healthy early run should have essentially 0% at the floor/ceiling and should not show broad migration outside 0.75×–1.25× baseline.
6. If the 2k run is healthy, extend to 5k and compare frozen evaluation against the virgin baseline before starting a 100k run.

## Files added/changed

DOOMFLY:
- `doom_learning/flycraft_rule_v2.py` (new)
- `doom_learning/flycraft_v2.py` (new)
- `doom_learning/validate_flycraft_v2.py` (new)
- `doom_learning/kernel.cpp` (small guard so legacy eligibility is not updated when native LTD is disabled)

FlyCraft:
- `training.json`
- `src/flycraft/learning.py`
- `src/flycraft/r8_visual.py`
- `src/flycraft/training_metrics.py`
- `src/flycraft/train.py`
