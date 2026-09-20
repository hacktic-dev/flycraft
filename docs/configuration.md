# Configuration reference

Two families of config exist:

- `training.json` — the default signed-reinforcement training config.
- `config/*.json` — the baseline game config plus alternate training/decoder
  configs.

`train.ps1` uses `training.json` unless `-Config` is given. `run-baseline.ps1`
always uses `config/baseline.json` for the game. `train-readout.ps1` uses
`config/new-arch.json` by default.

---

## 1. `training.json`

Top-level sections: `training`, `environment`, `reward`, `attack`, `plasticity`,
`teaching`, `recording`, `dashboard`, and optionally `decoder`.

### `training`

| Setting | Meaning |
|---|---|
| `total_steps` | Default number of steps to add for a training invocation. |
| `max_steps_per_episode` | Episode timeout in 50 ms control ticks (400 = 20 simulated seconds). |
| `checkpoint_every_steps` | Numbered-checkpoint frequency; `0` disables periodic numbered checkpoints. |
| `evaluate_every_steps` | Automatic frozen-evaluation interval; `0` disables. |
| `evaluation_episodes` | Episodes per evaluation. |
| `seed` | Training random seed (also seeds the game). |

### `environment`

| Setting | Meaning |
|---|---|
| `randomize_player_position` | Randomize player X/Z each episode. |
| `randomize_tree_position` | Physically place the training tree at a new randomized valid X/Z. |
| `randomize_yaw` | Randomize player starting yaw. |
| `min_tree_distance` / `max_tree_distance` | Player→tree distance range. |
| `player_spawn_radius` | Random player X/Z range around the arena centre. |

The arena requires `player_spawn_radius + max_tree_distance <= 15`; validation
enforces this. FlyCraft does **not** create a new Minecraft world per episode:
it clears/rebuilds the experiment region in the running superflat arena, places
the tree, teleports the player and switches to survival.

### `reward` (evaluation metric only)

| Setting | Meaning |
|---|---|
| `angle_improvement` / `angle_worsening` | Score scales for decreasing/increasing angular error. |
| `distance_improvement` / `distance_worsening` | Score scales for decreasing/increasing distance. |
| `crosshair_on_log` | One-time score when the crosshair acquires the target log. |
| `breaking_progress_gain` / `breaking_progress_loss` | Score scales for real block-breaking progress changes. |
| `log_broken` | Large terminal score for breaking the target log. |
| `step_penalty` | Small per-step penalty. |

Angle, distance and breaking rewards are **deltas**, so the fly cannot farm
reward by remaining in a good state. Reward never feeds the brain; it only
scores/evaluates.

### `attack`

| Setting | Meaning |
|---|---|
| `threshold` | Neural attack accumulator threshold. |
| `decay` | Accumulator decay per tick. |
| `hold_steps` | How long attack stays held after activation. |

Only neural attack events feed this accumulator.

### `plasticity`

The signed-reinforcement learning rule.

| Setting | Meaning |
|---|---|
| `model` | `gamma1-fatigue-selective-signed-v4` (required by `train.ps1`). |
| `eta` | Learning rate / plasticity scale. |
| `eligibility_tau_ms` | Fast KC credit/eligibility trace time constant. |
| `eligibility_baseline_tau_ms` | Per-KC slow baseline for winner selection. |
| `eligibility_reference_hz` | Reference rate for the trace. |
| `eligibility_gate_hz` | Minimum activity to consider a KC. |
| `eligibility_winner_fraction` | Fraction of active KCs eligible per event (25%). |
| `eligibility_max_kcs` | Hard cap on eligible KCs per event (256). |
| `eligibility_activity_floor` | Activity floor below which a KC is ignored. |
| `selection_fatigue_tau_ms` / `selection_fatigue_strength` | Soft decaying ranking penalty encouraging winner turnover. |
| `recovery_tau_seconds` | Time constant for weights relaxing toward 1.0× baseline. |
| `minimum_fraction` / `maximum_fraction` | Hard weight bounds (0.5×–1.5×). |
| `max_log_update_per_event` | Per-event log-domain update cap. |
| `current` | PPL101 stimulation current for aversive events. |
| `pulse_ticks` | Aversive pulse length in control ticks. |
| `cooldown_ticks` | Minimum gap between aversive pulses. |
| `aversive_signal_threshold` | Negative teaching magnitude that schedules a pulse. |
| `health_check_every_steps` | Plasticity health print cadence. |
| `warning_/critical_/abort_saturation_fraction` | Efficacy-saturation alarm thresholds. |
| `warning_/critical_/abort_extreme_fraction` | Fraction outside 0.75×–1.25× alarm thresholds. |
| `warning_/critical_/abort_mean_drift` | Mean-efficacy drift alarm thresholds. |
| `abort_on_collapse` | Abort the run when the collapse threshold is reached. |
| `action_credit` | Optional object enabling the action-conditioned steering rule (see below). |

### `teaching`

The balanced raw-delta teaching transform, independent of `reward`.

| Setting | Meaning |
|---|---|
| `model` | `balanced-raw-delta-v1`. |
| `angle_scale_deg` / `distance_scale_blocks` / `progress_scale` | Normalization scales for each delta before `tanh`. |
| `angle_weight` / `distance_weight` / `progress_weight` | Component weights. |
| `crosshair_weight` | Weight for crosshair acquire/loss transitions. |
| `success_weight` | Weight for the terminal log break. |
| `deadband` | Signals below this magnitude are treated as neutral. |

Equal-and-opposite physical changes produce equal-and-opposite teaching.
`python -m flycraft.validate_teaching` verifies this.

### `recording`

| Setting | Meaning |
|---|---|
| `record_every_episodes` | Record every Nth training episode; `0` disables interval recording. |
| `record_first_episode` | Always record episode 1. |
| `record_final_episode` | Record in the final budget window so the actual final episode is captured. |
| `record_minecraft` | Write Minecraft `baseline.mp4`. |
| `record_dashboard` | Write dashboard, retina and neural-activity MP4s. |
| `record_activity` | Write sparse 60 Hz spikes and quantized 20 Hz voltage. |

Non-recorded episodes still capture RGB (the fly needs it) but skip the
expensive dashboard/video/activity work.

### `dashboard`

| Setting | Meaning |
|---|---|
| `reward_rolling_average_episodes` | Rolling-average window in the dashboard. |

### `decoder` (optional)

Absent = legacy BCI. See [decoder.md](decoder.md). Fields depend on mode:

- `balanced-bilateral-v1`: `left_reference_hz`, `right_reference_hz`,
  `regularizer`, `deadband`, `max_yaw_deg_per_second`, `polarity`, `neuron_type`.
- `balanced-bilateral-v2`: `left_center_hz`, `right_center_hz`, `left_scale_hz`,
  `right_scale_hz`, `opponent_offset`, `response_scale`, plus the common fields.
- `balanced-bilateral-v3`: v2 plus a learned scalar `command_offset`.

### `plasticity.action_credit` (optional)

Enables action-conditioned steering credit. Key fields:

| Setting | Meaning |
|---|---|
| `model` | `action-conditioned-steering-v2`. |
| `mode` | `direct-opponent` (decoder reads MBON11 directly) or `calibrated-downstream` (explicit MBON turn-effect signs). |
| `tau_ms` | Action-eligibility trace time constant. |
| `prediction_tau_ms` | Reward-prediction baseline time constant. |
| `max_yaw_deg_per_tick` | Yaw scale used to weight action eligibility. |
| `action_deadband_deg` | Yaw below this does not add directional credit. |
| `eta`, `max_log_update` | Update scale/cap. |
| `minimum_fraction` / `maximum_fraction` | Weight bounds. |
| `mbon_turn_effect` | `{"L": ±1, "R": ±1}` for `calibrated-downstream`. |

`direct-opponent` requires an MBON11 bilateral decoder; `calibrated-downstream`
requires explicit, experimentally-measured signs (do not infer them from
anatomy).

---

## 2. Performance note on cadence

The default `training.json` uses relatively frequent evaluation/recording. In
the worst case, `evaluation_episodes × max_steps_per_episode` every
`evaluate_every_steps` can add a large amount of compute before recording
overhead. For a long run, choose the cadence deliberately, for example:

```json
"checkpoint_every_steps": 10000,
"evaluate_every_steps": 10000,
"evaluation_episodes": 5,
"record_every_episodes": 20
```

Then run a larger manual `Evaluate -Episodes 20`/`50` on promising/final
checkpoints. See [performance.md](performance.md).

---

## 3. `config/` files

| File | Purpose |
|---|---|
| `config/baseline.json` | Frozen-baseline game config: `learning:false`, 640x480, port 8023, legacy `yaw_gain:-1.75`, `max_yaw_degrees:10.5`, `forward_threshold:0.5`, a red-concrete target at `(3,9)`. Also used as the game config for training (with `seed` overridden). |
| `training.json` | Default signed-reinforcement training config (root file). |
| `config/training-bilateral.json` | Training config with a `balanced-bilateral-v1` DNp20 decoder calibrated on uniform gray. |
| `config/training-bilateral-v2.json` | Training config with a `balanced-bilateral-v2` DNp20 decoder calibrated on natural task scenes. |
| `config/training-bilateral-v3.json` | Training config with a `balanced-bilateral-v3` decoder (v2 + closed-loop learned intercept). |
| `config/new-arch.json` | Config used by `train-readout.ps1` and for the action-credit path: v3 decoder plus `plasticity.action_credit` in `calibrated-downstream` mode. |
| `config/readout-learning.json` | Readout config variant that turns on per-episode recording and autonomous probes (`record_every_training_episode`, `autonomous_probe_after_episode`) and shortens teacher stop/attack distances. |

### Game config fields (`config/baseline.json`)

| Field | Meaning |
|---|---|
| `learning` | Must be `false` for the baseline runner. |
| `seed` | World/arena seed. |
| `width` / `height` | Neural view size (640x480); capture width is derived to 16:9. |
| `port` | CraftGround port (baseline 8023). |
| `neural_ms_per_tick` | Must be exactly `50.0`. |
| `yaw_gain` | Legacy decoder yaw multiplier (negative). |
| `max_yaw_degrees` | Per-tick yaw cap. |
| `forward_threshold` | Decoder forward activation threshold. |
| `attack_enabled` | Allow attack output. |
| `debug_every` | Debug-snapshot cadence. |
| `target_block`, `target_x`, `target_z` | Baseline target pillar. |

Training builds its own arena and target tree per episode
(`training_game.reset_episode`), independent of the baseline target.
