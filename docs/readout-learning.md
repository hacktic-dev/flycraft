# Frozen-connectome supervised motor readout

This is a separate learning path from KC→MBON11 plasticity. It does **not**
alter any fly synaptic weight.

Pipeline:

```text
Minecraft RGB
  -> existing fly retina / R8 adapter
  -> fixed full MaleCNS neural dynamics
  -> neural spike features
  -> trainable 1-hidden-layer motor readout
  -> yaw / forward / attack
```

During training a scripted expert can see the external target geometry and
produces the desired yaw/walk/attack label. The student never receives target
coordinates, distance, angle, raycast or breaking progress in its feature
vector. Its only input is simulated neural spike activity.

Before training, a fresh run executes three teacher-only preflight episodes and
requires at least two successful log breaks. If the expert itself is wrong,
training aborts immediately.

![Supervised readout dashboard](images/readout-dashboard.png)

*The readout dashboard during an autonomous probe: the student's yaw/walk/attack
versus the teacher's labels, held-out imitation metrics (loss, yaw MAE, walk and
attack accuracy, imitation score) and the live task state.*

Training is deliberately staged for reliability:

1. **DEMONSTRATE**: teacher controls Minecraft; every neural state is labelled.
2. **DAGGER**: student gets an increasing share of control while the teacher
   continues to label the states the student visits.
3. **AUTONOMOUS**: student controls the game alone.

The DAgger handoff is quality-gated, so a weak student cannot suddenly take 100%
control just because a step counter elapsed.

---

## Run it

```powershell
# Smoke-test the readout math (no Minecraft)
. .\env.ps1
& $python scripts\test_readout_learning.py

# Train
.\train-readout.ps1 -Mode Fresh -Steps 10000 -Config "config/new-arch.json"

# Resume (the model resumes; the finite replay buffer starts empty)
.\train-readout.ps1 -Mode Resume -Checkpoint latest -Steps 5000 -Config "config/new-arch.json"

# Student-only evaluation (expert used only for optional diagnostic labels)
.\train-readout.ps1 -Mode Evaluate -Checkpoint latest -Episodes 20 -Config "config/new-arch.json"

# Autonomous recorded footage
.\train-readout.ps1 -Mode Replay -Checkpoint latest -Episodes 3 -Config "config/new-arch.json"
```

Artifacts go to `artifacts\readout-training\run-...\`. The learned object is
`readout-latest.npz`. The full connectome is rebuilt from the fixed config and
is never checkpointed as learned state. Every episode checks that the connectome
weight array is bit-identical to its starting value.

---

## How the student works

`readout_learning.py` implements:

- **`NeuralFeatureEncoder`** — a CountSketch of the full CNS spike counts plus
  explicit retinotopic R1-R6/R8 spike pools, concatenated with a temporal EMA.
  Spike counts are log-scaled by rate.
- **`MotorMLP`** — a one-hidden-layer MLP (default 192 units) with `tanh` yaw and
  sigmoid walk/attack heads, Adam, gradient clipping and positive-class
  weighting for walk/attack.
- **`ReplayBuffer`** — a finite ring buffer for training and a separate
  validation buffer.
- **`ReadoutLearner`** — DAgger scheduling, quality gates, blending and
  checkpointing.

The privileged teacher (`expert_action`) turns the external target geometry into
a yaw command, a forward decision and an attack decision, with conservative
reach thresholds so it does not stare/attack from out of range.

---

## Optional tuning

Defaults work without changing `config/new-arch.json`. To override them, add a
top-level `readout_learning` object, for example:

```json
{
  "readout_learning": {
    "model": "frozen-connectome-readout-v1",
    "hidden_units": 192,
    "teacher_preflight_episodes": 3,
    "teacher_preflight_required_successes": 2,
    "hash_dim": 768,
    "demo_samples": 2200,
    "dagger_full_student_samples": 7500,
    "learning_rate": 0.001,
    "attack_positive_weight": 6.0
  }
}
```

`config/readout-learning.json` is a ready-made variant that additionally enables
per-episode recording and autonomous probes and shortens the teacher's
stop/attack distances.

Useful dashboard targets before trusting autonomous behaviour:

- held-out imitation score: preferably ≥ 85%
- yaw MAE: preferably well below 1 degree/tick
- walk/attack accuracy: preferably > 90%, while checking class balance in logs
- student share: should climb only as validation quality improves

---

## Scientific wording

Accurate description:

> The connectome was frozen. I trained a small motor readout to translate the
> fly simulation's neural activity into Minecraft controls using demonstrations,
> then let the learned controller act autonomously.

Do not claim that biological fly synapses learned Minecraft in this mode.
