# Visual input — the DOOMFLY v6 R8 adapter

FlyCraft's learning path adds a colour visual input to the original luminance
photoreceptors, ported from DOOMFLY's pinned `doom_learning_v6.visual`
implementation. This changes **training sensory input only**; it does not
replace FlyCraft's plasticity rule.

`R8VisualMemoryBrain` (`src/flycraft/r8_visual.py`) implements the adapter on top
of `FatigueSelectiveMemoryBrain`:

- the original **3,335 R1-R6 luminance inputs** remain active;
- **811 inferred R8 colour inputs** are added;
- **R8p samples the blue channel** and **R8y samples the green channel**, after
  exact sRGB-to-linear conversion;
- existing **R8→aMe12 edges are forced excitatory** while retaining their
  recorded magnitudes;
- RGB visual input is advanced in **≤10 ms neural chunks**, matching the v6
  adapter, even when a control tick is 50 ms.

Minecraft game-state telemetry is still not supplied to the brain.

These are modelling assumptions, not validated fly photoreceptor physiology.
DOOMFLY itself labels this visual model experimental and unvalidated.

---

## Verifying the adapter

```powershell
. .\env.ps1
& $python scripts\verify_r8_input.py
```

Expected headline output includes:

```text
mapped R8 total: 811
R8 VISUAL ADAPTER STRUCTURE OK
R8 VECTOR STIMULATION SMOKE TEST OK
```

The smoke test exercises the vector-valued current shape used by the R8 adapter
(the v1 neural step previously rejected vector currents).

---

## Important: start a fresh training run

The visual model and initial weight configuration have changed. Old v1-only
checkpoints are intentionally **not** compatible with this model. The previous
run showed zero changed plastic edges, so there was no learned synaptic state to
preserve.

```powershell
.\train.ps1 -Mode Fresh -Steps 1000
```

Watch `KC_spikes` / plasticity in the generated metrics before committing to a
long run. Adding the v6 visual adapter does not guarantee KC recruitment or
successful learning.

---

## Provenance

`R8VisualMemoryBrain` keeps:

- `pre_visual_weight_sha256` — hash of the untouched pinned whole-graph weights,
  so `model-lock.json` can still verify the expected upstream graph was loaded
  before the v6 visual sign assumption is applied;
- `visual_report` — mapped R8p/R8y counts, known-unmapped counts, projection
  confidence, corrected-edge hash, and the coordinate-inference/spectrum/
  physiology descriptions. This is written into `model.json` and recorded
  episode `visuals.json`.
