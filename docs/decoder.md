# Steering decoders

FlyCraft can decode descending-neuron activity into Minecraft yaw in two ways.
The decoder is chosen by the optional `decoder` block of a training config.
Normal `training.json` (no `decoder` block) and old checkpoints keep legacy BCI
behaviour. No neuron, edge, sensory input or plasticity rule is changed by
choosing a decoder.

---

## 1. Legacy BCI (default)

`doom.engine.NeuralControls(mode='bci')` uses the upstream readouts:

```text
DNp20 L/R rate difference → camera yaw
DNpe017 combined rate      → forward
DNpe017 spikes             → attack
```

`map_action` multiplies the legacy `turn` by the game config's `yaw_gain`
(negative in `config/baseline.json`). Forward, attack and the 100 ms rate filter
are unchanged.

---

## 2. Balanced bilateral (v1 / v2 / v3)

An engineered opponent decoder that keeps the existing 100 ms rate filter and
the forward/attack mappings but replaces steering with a normalized left/right
signal from a chosen neuron pair (`DNp20`, `DNa02`, or `MBON11`).

### v1 — fixed reference rates

```
L = left_rate / left_reference_hz
R = right_rate / right_reference_hz
opponent = (R - L) / (R + L + regularizer)
command  = sign(opponent) * max(abs(opponent) - deadband, 0) / (1 - deadband)
yaw_degrees = command * max_yaw_deg_per_second * control_interval_seconds
```

The regularizer suppresses amplification at low activity; the soft deadband
removes small differences without an abrupt jump. Silence produces zero turn;
equal normalized activity cancels. Positive yaw means right in Minecraft; the
legacy negative `yaw_gain` is bypassed, while the game's yaw cap is respected.

### v2 — robust natural-scene normalization

```
L = (left_rate  - left_center_hz)  / left_scale_hz
R = (right_rate - right_center_hz) / right_scale_hz
opponent = tanh((R - L - opponent_offset) / response_scale)
command  = deadband(opponent) * polarity
```

Centers/scales are estimated per side with a robust (MAD, falling back to std)
scale, and `opponent_offset` is solved so the mean post-deadband command over
the calibration bank is zero. A symmetric response scale preserves steering
variation.

### v3 — v2 plus a learned motor intercept

v3 adds a scalar `command_offset` solved from **closed-loop** Minecraft
trajectories, applied after `tanh` so it is interpretable in normalized
command units:

```
opponent = tanh((R - L - opponent_offset) / response_scale)
command  = deadband(opponent - command_offset) * polarity
```

### Polarity

`polarity` is `-1` or `+1` and is explicit/configurable. The relationship
between a chosen cell side and useful visual steering still requires a
closed-loop experiment.

---

## 3. Calibration and validation tools

| Script | Purpose | Live Minecraft |
|---|---|---|
| `scripts/test_decoder.py` | Unit tests: symmetry, silence, bounds, polarity, legacy equivalence, checkpoint compatibility. | no |
| `scripts/test_bilateral_decoder_v2.py` | Numerical checks of natural-scene v2 calibration and v1 back-compat. | no |
| `scripts/test_bilateral_decoder_v3.py` | Numerical checks of the v3 closed-loop intercept solver. | no |
| `scripts/test_mirrored_calibration_v4.py` | Geometry checks of mirrored-pair starts. | no |
| `scripts/calibrate_bilateral_decoder.py` | Fits the v3 scalar intercept from live yaw-only episodes. | yes |
| `scripts/calibrate_bilateral_mirrored.py` | From-scratch mirrored-pair calibration (fresh centers/scales + intercept). | yes |
| `scripts/validate_bilateral_bias.py` | Held-out frozen validation gate on residual steering bias. | yes |
| `scripts/validate_bilateral_mirrored.py` | Held-out mirrored-pair validation gate. | yes |
| `scripts/compare_decoders.py` | Compares legacy vs bilateral yaw on identical recorded spikes. | no |
| `scripts/compare_weight_effect.py` | Replays exact recorded RGB under frozen vs +50% weights. | partly |

Calibration writes evidence to `artifacts/decoder-calibration` and never
overwrites a config automatically. `compare_decoders.py` reports command balance,
not task performance.

`mirrored_starts.py` provides exact left/right mirrored episode geometry
(player position, target block centre and yaw), preserving player–target
distance while reversing the signed target angle, so calibration uses matched
scenes instead of unrelated random episodes.

---

## 4. Evidence and limitations

[Rayshubskiy et al., *Neural circuit mechanisms for steering control in walking
Drosophila*](https://elifesciences.org/articles/102230) reports that bilateral
DNa02 firing-rate differences predict steering. That supports using opponent
left/right signals, but does not validate DNp20 as a steering neuron or prescribe
FlyCraft's gains, normalization or deadband.

In the recorded FlyCraft baseline trials, right DNa02 and both DNp09 cells were
silent, so a naive biological-role decoder would leave one-sided steering and no
DNp09 forward drive on those inputs. DNp20 is active on both sides, but
uniform-gray calibration produced right 30.3917 Hz versus left 7.0185 Hz, so the
decoder compensates the neutral response imbalance. These are engineering
reference gains, not measured biological calibration.

References were measured with untrained weights, uniform RGB 128 at 640x480,
20 warm-up control ticks and 60 measurement ticks, and remain fixed during
training/evaluation. No rewards, target coordinates or action labels enter the
calibration. Calibration under gray does not guarantee balance on natural scenes.

Forward and attack remain unchanged; their shared DNpe017 source is a separate
limitation and is not solved by the bilateral decoder.

---

## 5. Reproduce

```powershell
# One short fresh trial with a candidate decoder:
.\train.ps1 -Mode Fresh -Config config/training-bilateral.json -Steps 400 -NoPreview

# Frozen comparison using old learned weights (use the actual checkpoint path):
.\train.ps1 -Mode Evaluate -Config config/training-bilateral.json `
  -Checkpoint artifacts/training/run-.../checkpoints/evaluation-step-000060000 `
  -Episodes 2 -NoPreview

. .\env.ps1
& $python scripts/test_decoder.py
& $python scripts/calibrate_bilateral_decoder.py --config config/training-bilateral-v2.json --output-config config/training-bilateral-v3.json
& $python scripts/compare_decoders.py
```

`compare_decoders.py` uses existing `artifacts/weight-effect-2d/tick-results.json`
recordings to compare decoder commands without rerunning the simulation. It
reports command balance, not task performance.

New checkpoints record the decoder configuration as well as its filter rates.
`Resume` rejects decoder changes; explicit `Branch`/`Evaluate` overrides permit
comparisons and reset decoder filters. The dashboard labels the active mapping
and shows its opponent signal.
