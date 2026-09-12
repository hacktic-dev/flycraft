# DOOMFLY v6 visual-input patch

This patch changes **training sensory input only**. It does not replace FlyCraft's existing `gamma1-eligibility-ltd-v1` plasticity rule.

Training now follows the visual adapter in the pinned DOOMFLY `doom_learning_v6.visual` implementation:

- original 3,335 R1-R6 luminance inputs remain active;
- 811 inferred R8 color inputs are added;
- R8p samples the blue channel, R8y samples the green channel, after sRGB-to-linear conversion;
- existing R8-to-aMe12 edges are forced to excitatory sign while retaining their recorded magnitudes;
- RGB visual input is advanced in <=10 ms neural chunks, matching the v6 adapter.

Minecraft game-state telemetry is still not supplied to the brain.

## Important: start a fresh training run

The visual model and initial weight configuration have changed. Old v1-only checkpoints are intentionally not compatible with this model. The previous run showed zero changed plastic edges, so there is no learned synaptic state to preserve.

After copying the patch, verify the adapter:

```powershell
. .\env.ps1
& $python scripts\verify_r8_input.py
```

Expected headline output includes:

```text
mapped R8 total: 811
R8 VISUAL ADAPTER STRUCTURE OK
```

Then start a short fresh run first:

```powershell
.\train.ps1 -Mode Fresh -Steps 1000
```

Watch `KC_spikes` / plasticity in the generated metrics before committing to a long run. Adding the v6 visual adapter does not guarantee KC recruitment or successful learning; DOOMFLY itself labels this visual model experimental and unvalidated.
