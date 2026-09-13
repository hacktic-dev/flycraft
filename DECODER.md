# Experimental calibrated bilateral decoder

Use the optional `config/training-bilateral.json` configuration. Normal
`training.json` and checkpoints without a decoder configuration retain legacy BCI
behaviour. No neuron, edge, sensory input or plasticity rule is changed.

## Evidence and limitations

[Rayshubskiy et al., Neural circuit mechanisms for steering control in walking
Drosophila](https://elifesciences.org/articles/102230) reports that bilateral
DNa02 firing-rate differences predict steering. This supports using opponent
left/right signals, but does not validate DNp20 as a steering neuron or prescribe
our game gains, normalization or deadband.

In the two recorded FlyCraft baseline trials, right DNa02 and both DNp09 cells
were silent. Simply switching to the biological-role decoder would therefore
leave one-sided steering and no DNp09 forward drive on those inputs. Although
[Bidaye et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC9435592/) supports DNp09's
walking role in flies, that does not guarantee useful activity in this model.

DNp20 is active on both sides, but uniform-gray calibration produced right
30.3917 Hz versus left 7.0185 Hz. The experiment therefore keeps the existing
DNp20 pair and compensates its neutral response imbalance. These are engineering
reference gains, not measured biological calibration. DNa02 is configurable,
but would require usable bilateral activity and its own nonzero calibration.

## Exact mapping

The existing 100 ms rate filter remains unchanged. With fixed reference rates:

```
L = left_rate / left_reference_rate
R = right_rate / right_reference_rate
opponent = (R - L) / (R + L + 1)
command = sign(opponent) * max(abs(opponent) - 0.05, 0) / 0.95
yaw_degrees = command * 120 * control_interval_seconds
```

The regularizer suppresses amplification at low activity; the soft deadband
removes small differences without an abrupt jump. Silence produces zero turn.
Equal normalized activity cancels. Positive yaw means right in Minecraft; the
old negative `yaw_gain` is bypassed, while the game's yaw cap is still respected.
Polarity is explicit and configurable. The relationship between DNp20 side and
useful visual steering still needs a closed-loop experiment.

References were measured with untrained weights, uniform RGB 128 at 640x480,
20 warm-up control ticks and 60 measurement ticks. They remain fixed during
training/evaluation; no rewards, target coordinates, or action labels enter this
calibration. Calibration under gray does not guarantee balance on natural scenes.

Forward and attack remain unchanged in this isolated steering experiment. Their
shared DNpe017 source remains a separate limitation; this change does not solve
independent approach/attack control or establish improved learning.

## Run and reproduce

Measured on identical recorded spikes (degrees per 50 ms control tick; negative
is left):

| Input | Legacy mean yaw | Bilateral mean yaw |
|---|---:|---:|
| Uniform-gray measurement window | -4.908 | +0.271 |
| Baseline episode 1 | -4.041 | +0.204 |
| Baseline episode 2 | -1.606 | -0.589 |

This demonstrates reduced net bias on these inputs, not improved targeting or
learning. Forward, attack and filtered readout values were identical. Synthetic
symmetry, silence, direction, bounds, legacy-equivalence and checkpoint tests
pass. The existing dashboard panel was rendered and inspected with the new mode.

```powershell
# One short fresh trial with the candidate decoder:
.\train.ps1 -Mode Fresh -Config config/training-bilateral.json -Steps 400 -NoPreview

# Frozen comparison using old learned weights (use the actual checkpoint path):
.\train.ps1 -Mode Evaluate -Config config/training-bilateral.json -Checkpoint artifacts/training/run-20260913-014051-109918_2d/checkpoints/evaluation-step-000060000 -Episodes 2 -NoPreview

. ./env.ps1
& $python scripts/test_decoder.py
& $python scripts/calibrate_bilateral_decoder.py
& $python scripts/compare_decoders.py
```

The last script uses the existing `artifacts/weight-effect-2d/tick-results.json`
recordings to compare decoder commands without rerunning the neural simulation.
It reports command balance, not task performance. Calibration writes evidence to
`artifacts/decoder-calibration`, never overwrites the configuration automatically.

New checkpoints record the decoder configuration as well as its filter rates.
`Resume` rejects decoder changes. Explicit `Branch` or `Evaluate` configuration
overrides permit comparisons and reset decoder filters. Old checkpoints retain
legacy semantics by default. The dashboard labels the active mapping and shows
its opponent signal; research and engineering assumptions remain separate.
