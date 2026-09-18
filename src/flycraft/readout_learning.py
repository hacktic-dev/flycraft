"""Frozen-connectome supervised motor readout for FlyCraft.

The fly connectome is never modified by this module. Minecraft target geometry is
used only by ``expert_action`` to produce supervised labels. Student features are
computed exclusively from simulated neural spike counts.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

MODEL = 'frozen-connectome-readout-v1'

DEFAULTS = {
    'model': MODEL,
    'seed': 73191,
    'max_yaw_deg_per_tick': 6.0,
    'teacher_yaw_gain': 0.22,
    'teacher_walk_angle_deg': 18.0,
    'teacher_stop_distance': 3.15,
    'teacher_attack_angle_deg': 7.0,
    'teacher_attack_distance': 4.25,
    'student_walk_threshold': 0.50,
    'student_attack_threshold': 0.42,
    # Neural feature encoder. Full-CNS CountSketch + retinotopic pools, with EMA.
    'hash_dim': 768,
    'retina_grid_width': 16,
    'retina_grid_height': 12,
    'rate_cap_hz': 250.0,
    'feature_ema_tau_ms': 250.0,
    # Small learned readout.
    'hidden_units': 192,
    'learning_rate': 0.001,
    'weight_decay': 1e-5,
    'batch_size': 64,
    'train_every_samples': 16,
    'gradient_steps': 12,
    'warmup_train_samples': 384,
    'yaw_loss_weight': 2.0,
    'walk_positive_weight': 2.0,
    'attack_positive_weight': 6.0,
    'gradient_clip': 5.0,
    # Replay / validation.
    'replay_capacity': 18000,
    'validation_capacity': 2500,
    'validation_every_n': 10,
    'validation_batch': 768,
    # Teacher-forcing -> DAgger curriculum.
    'teacher_preflight_episodes': 3,
    'teacher_preflight_required_successes': 2,
    'demo_samples': 2200,
    'dagger_full_student_samples': 7500,
    'minimum_dagger_student_share': 0.10,
    'quality_gate_1': 0.68,
    'quality_gate_2': 0.78,
    'quality_gate_3': 0.86,
    # Video/diagnostic automation.
    'record_every_training_episode': False,
    'autonomous_probe_after_episode': False,
    'autonomous_probe_episodes': 1,
}


def validated_config(config=None):
    c = {**DEFAULTS, **(config or {})}
    if c.get('model') != MODEL:
        raise ValueError(f'Unknown readout-learning model: {c.get("model")}')
    ints = ('seed', 'hash_dim', 'retina_grid_width', 'retina_grid_height',
            'hidden_units', 'batch_size', 'train_every_samples', 'gradient_steps',
            'warmup_train_samples', 'replay_capacity', 'validation_capacity',
            'validation_every_n', 'validation_batch', 'teacher_preflight_episodes',
            'teacher_preflight_required_successes', 'demo_samples', 'dagger_full_student_samples',
            'autonomous_probe_episodes')
    for k in ints:
        if int(c[k]) <= 0:
            raise ValueError(f'{k} must be positive')
        c[k] = int(c[k])
    if c['teacher_preflight_required_successes'] > c['teacher_preflight_episodes']:
        raise ValueError('teacher_preflight_required_successes cannot exceed teacher_preflight_episodes')
    if c['dagger_full_student_samples'] <= c['demo_samples']:
        raise ValueError('dagger_full_student_samples must exceed demo_samples')
    for k in ('max_yaw_deg_per_tick', 'teacher_yaw_gain', 'teacher_walk_angle_deg',
              'teacher_stop_distance', 'teacher_attack_angle_deg',
              'teacher_attack_distance', 'rate_cap_hz', 'feature_ema_tau_ms',
              'learning_rate', 'batch_size', 'yaw_loss_weight',
              'walk_positive_weight', 'attack_positive_weight', 'gradient_clip'):
        if not math.isfinite(float(c[k])) or float(c[k]) <= 0:
            raise ValueError(f'{k} must be finite and positive')
    for k in ('student_walk_threshold', 'student_attack_threshold',
              'minimum_dagger_student_share', 'quality_gate_1', 'quality_gate_2',
              'quality_gate_3'):
        if not 0 <= float(c[k]) <= 1:
            raise ValueError(f'{k} must be in [0,1]')
    c['record_every_training_episode'] = bool(c.get('record_every_training_episode', False))
    c['autonomous_probe_after_episode'] = bool(c.get('autonomous_probe_after_episode', False))
    return c


def wrap_degrees(value):
    return (float(value) + 180.0) % 360.0 - 180.0


def desired_yaw(state, target):
    tx, _, tz = target
    dx = float(tx) + 0.5 - float(state['x'])
    dz = float(tz) + 0.5 - float(state['z'])
    return math.degrees(math.atan2(-dx, dz))


def expert_action(state, target, config=None):
    """Privileged scripted teacher. Its geometry is label-only, never a feature."""
    c = validated_config(config)
    if not bool(state.get('target_present', True)):
        return {'yaw': 0.0, 'forward': False, 'attack': False, 'error_deg': 0.0}
    error = wrap_degrees(desired_yaw(state, target) - float(state['yaw']))
    max_yaw = float(c['max_yaw_deg_per_tick'])
    yaw = float(np.clip(error * float(c['teacher_yaw_gain']), -max_yaw, max_yaw))
    distance = float(state['distance'])
    on_target = bool(state.get('on_target', False))
    # Do not treat merely seeing the log as permission to stop. Minecraft's
    # raycast can report the log while the player is still too far away to break
    # it. The old policy therefore had a dead zone where it stared/attacked from
    # range forever. Keep approaching until BOTH the log is actually under the
    # crosshair and our conservative reach threshold has been reached.
    attack = bool(
        on_target and
        abs(error) <= float(c['teacher_attack_angle_deg']) and
        distance <= float(c['teacher_attack_distance'])
    )
    # Distance never disables approach by itself. Collision with the trunk is a
    # safer final stop than a guessed geometric stop distance.
    forward = bool(
        not attack and
        abs(error) <= float(c['teacher_walk_angle_deg'])
    )
    return {'yaw': yaw, 'forward': forward, 'attack': attack, 'error_deg': error}


class NeuralFeatureEncoder:
    """CountSketch of the full CNS plus explicit retinotopic neural spike pools."""
    def __init__(self, brain, config=None):
        self.config = validated_config(config)
        c = self.config
        self.n = int(brain.n)
        rng = np.random.default_rng(int(c['seed']))
        self.hash_bucket = rng.integers(0, c['hash_dim'], size=self.n, dtype=np.int32)
        self.hash_sign = rng.choice(np.array([-1.0, 1.0], np.float32), size=self.n)
        self.hash_occupancy = np.bincount(self.hash_bucket, minlength=c['hash_dim']).astype(np.float32)
        self.hash_norm = np.sqrt(np.maximum(self.hash_occupancy, 1.0))

        gw, gh = c['retina_grid_width'], c['retina_grid_height']
        self.grid_size = gw * gh
        self.retina = np.asarray(getattr(brain, 'retina', np.zeros(0, np.int32)), dtype=np.int32)
        self.retina_bins = self._grid_bins(np.asarray(getattr(brain, 'uv', np.zeros((len(self.retina), 2))), dtype=np.float64), gw, gh)
        self.retina_occ = np.bincount(self.retina_bins, minlength=self.grid_size).astype(np.float32) if len(self.retina_bins) else np.ones(self.grid_size, np.float32)

        self.r8 = np.asarray(getattr(brain, 'r8', np.zeros(0, np.int32)), dtype=np.int32)
        r8_uv = np.asarray(getattr(brain, 'r8_uv', np.zeros((len(self.r8), 2))), dtype=np.float64)
        self.r8_bins = self._grid_bins(r8_uv, gw, gh)
        self.r8_channel = np.asarray(getattr(brain, 'r8_channel', np.ones(len(self.r8), np.int32)), dtype=np.int32)
        self.r8_masks = [self.r8_channel == 1, self.r8_channel == 2]
        self.r8_occ = []
        for mask in self.r8_masks:
            self.r8_occ.append(np.bincount(self.r8_bins[mask], minlength=self.grid_size).astype(np.float32) if np.any(mask) else np.ones(self.grid_size, np.float32))

        self.base_dim = int(c['hash_dim'] + self.grid_size * 3)
        self.output_dim = self.base_dim * 2
        self.ema = np.zeros(self.base_dim, np.float32)
        self.initialized = False

    @staticmethod
    def _grid_bins(uv, width, height):
        if uv.ndim != 2 or uv.shape[1] < 2 or not len(uv):
            return np.zeros(0, np.int32)
        out = uv[:, :2].copy()
        for axis, bins in ((0, width), (1, height)):
            finite = np.isfinite(out[:, axis])
            if not np.any(finite):
                out[:, axis] = 0.5
                continue
            lo = float(np.nanmin(out[finite, axis])); hi = float(np.nanmax(out[finite, axis]))
            if hi <= lo + 1e-12:
                out[:, axis] = 0.5
            else:
                out[:, axis] = np.clip((out[:, axis] - lo) / (hi - lo), 0.0, 0.999999)
        x = (out[:, 0] * width).astype(np.int32)
        y = (out[:, 1] * height).astype(np.int32)
        return y * width + x

    def reset_episode(self):
        self.ema.fill(0.0)
        self.initialized = False

    def _scaled_rates(self, counts, seconds, indices=None):
        x = np.asarray(counts if indices is None else counts[indices], dtype=np.float32)
        rates = x / max(float(seconds), 1e-6)
        return np.clip(np.log1p(rates) / math.log1p(float(self.config['rate_cap_hz'])), 0.0, 1.5)

    def encode(self, counts, seconds=0.05):
        counts = np.asarray(counts)
        if counts.shape != (self.n,):
            raise ValueError(f'Expected {self.n} neural counts, got {counts.shape}')
        c = self.config
        active = np.flatnonzero(counts)
        sketch = np.zeros(c['hash_dim'], np.float32)
        if len(active):
            scaled = self._scaled_rates(counts, seconds, active)
            sketch = np.bincount(
                self.hash_bucket[active],
                weights=(scaled * self.hash_sign[active]).astype(np.float64),
                minlength=c['hash_dim'],
            ).astype(np.float32)
            sketch /= self.hash_norm
            sketch = np.tanh(sketch * 1.5).astype(np.float32)

        retina = np.zeros(self.grid_size, np.float32)
        if len(self.retina):
            rv = self._scaled_rates(counts, seconds, self.retina)
            retina = np.bincount(self.retina_bins, weights=rv.astype(np.float64), minlength=self.grid_size).astype(np.float32)
            retina /= np.maximum(self.retina_occ, 1.0)

        colors = []
        for mask, occ in zip(self.r8_masks, self.r8_occ):
            pool = np.zeros(self.grid_size, np.float32)
            if len(self.r8) and np.any(mask):
                rv = self._scaled_rates(counts, seconds, self.r8[mask])
                pool = np.bincount(self.r8_bins[mask], weights=rv.astype(np.float64), minlength=self.grid_size).astype(np.float32)
                pool /= np.maximum(occ, 1.0)
            colors.append(pool)

        base = np.concatenate([sketch, retina, *colors]).astype(np.float32, copy=False)
        alpha = -math.expm1(-float(seconds) * 1000.0 / float(c['feature_ema_tau_ms']))
        if not self.initialized:
            self.ema[:] = base
            self.initialized = True
        else:
            self.ema += np.float32(alpha) * (base - self.ema)
        return np.concatenate([base, self.ema]).astype(np.float32, copy=False)


class ReplayBuffer:
    def __init__(self, capacity, dim):
        self.capacity = int(capacity); self.dim = int(dim)
        self.x = np.empty((self.capacity, self.dim), np.float16)
        self.y = np.empty((self.capacity, 3), np.float32)
        self.size = 0; self.cursor = 0

    def add(self, x, y):
        self.x[self.cursor] = np.asarray(x, np.float16)
        self.y[self.cursor] = np.asarray(y, np.float32)
        self.cursor = (self.cursor + 1) % self.capacity
        self.size = min(self.capacity, self.size + 1)

    def sample(self, rng, count):
        if self.size == 0:
            raise ValueError('Cannot sample empty replay buffer')
        ix = rng.integers(0, self.size, size=min(int(count), self.size))
        return self.x[ix].astype(np.float32), self.y[ix]


class MotorMLP:
    def __init__(self, input_dim, config=None):
        self.config = validated_config(config)
        c = self.config; rng = np.random.default_rng(c['seed'] + 17)
        h = c['hidden_units']
        self.W1 = (rng.standard_normal((input_dim, h)) * math.sqrt(2.0 / max(1, input_dim))).astype(np.float32)
        self.b1 = np.zeros(h, np.float32)
        self.W2 = (rng.standard_normal((h, 3)) * math.sqrt(1.0 / max(1, h))).astype(np.float32)
        self.b2 = np.zeros(3, np.float32)
        self.params = [self.W1, self.b1, self.W2, self.b2]
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]
        self.t = 0

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))

    def forward(self, x, cache=False):
        x = np.asarray(x, np.float32)
        one = x.ndim == 1
        if one: x = x[None, :]
        z1 = x @ self.W1 + self.b1
        h = np.maximum(z1, 0.0)
        z2 = h @ self.W2 + self.b2
        out = np.empty_like(z2)
        out[:, 0] = np.tanh(z2[:, 0])
        out[:, 1:] = self._sigmoid(z2[:, 1:])
        if cache:
            return out, (x, z1, h, z2)
        return out[0] if one else out

    def train_batch(self, x, y):
        c = self.config
        pred, cache = self.forward(x, cache=True); xb, z1, h, z2 = cache
        y = np.asarray(y, np.float32)
        n = max(1, len(xb))
        eps = 1e-6
        yaw_err = pred[:, 0] - y[:, 0]
        yaw_loss = float(np.mean(yaw_err * yaw_err))
        walk_w = 1.0 + (float(c['walk_positive_weight']) - 1.0) * y[:, 1]
        attack_w = 1.0 + (float(c['attack_positive_weight']) - 1.0) * y[:, 2]
        walk_loss = float(np.mean(walk_w * (-(y[:, 1] * np.log(pred[:, 1] + eps) + (1-y[:, 1]) * np.log(1-pred[:, 1] + eps)))))
        attack_loss = float(np.mean(attack_w * (-(y[:, 2] * np.log(pred[:, 2] + eps) + (1-y[:, 2]) * np.log(1-pred[:, 2] + eps)))))
        loss = float(c['yaw_loss_weight']) * yaw_loss + walk_loss + attack_loss

        dz2 = np.zeros_like(pred)
        dz2[:, 0] = (2.0 * float(c['yaw_loss_weight']) / n) * yaw_err * (1.0 - pred[:, 0] ** 2)
        dz2[:, 1] = (walk_w / n) * (pred[:, 1] - y[:, 1])
        dz2[:, 2] = (attack_w / n) * (pred[:, 2] - y[:, 2])
        dW2 = h.T @ dz2 + float(c['weight_decay']) * self.W2
        db2 = dz2.sum(axis=0)
        dh = dz2 @ self.W2.T
        dz1 = dh * (z1 > 0)
        dW1 = xb.T @ dz1 + float(c['weight_decay']) * self.W1
        db1 = dz1.sum(axis=0)
        grads = [dW1, db1, dW2, db2]

        norm = math.sqrt(sum(float(np.sum(g.astype(np.float64) ** 2)) for g in grads))
        clip = float(c['gradient_clip'])
        if norm > clip:
            scale = clip / max(norm, 1e-12)
            grads = [g * scale for g in grads]

        self.t += 1; beta1 = 0.9; beta2 = 0.999; lr = float(c['learning_rate'])
        for p, m, v, g in zip(self.params, self.m, self.v, grads):
            m *= beta1; m += (1-beta1) * g
            v *= beta2; v += (1-beta2) * (g*g)
            mh = m / (1-beta1**self.t); vh = v / (1-beta2**self.t)
            p -= lr * mh / (np.sqrt(vh) + 1e-8)
        return loss


class ReadoutLearner:
    def __init__(self, brain, config=None):
        self.config = validated_config(config)
        self.encoder = NeuralFeatureEncoder(brain, self.config)
        self.model = MotorMLP(self.encoder.output_dim, self.config)
        self.train = ReplayBuffer(self.config['replay_capacity'], self.encoder.output_dim)
        self.validation = ReplayBuffer(self.config['validation_capacity'], self.encoder.output_dim)
        self.rng = np.random.default_rng(self.config['seed'] + 101)
        self.samples = 0; self.gradient_updates = 0
        self.train_loss = None; self.metrics = None
        self.score_history = []

    def reset_episode(self):
        self.encoder.reset_episode()

    def target(self, teacher):
        return np.array([
            float(teacher['yaw']) / float(self.config['max_yaw_deg_per_tick']),
            float(bool(teacher['forward'])), float(bool(teacher['attack']))
        ], np.float32)

    def predict(self, features):
        p = self.model.forward(features)
        return {
            'yaw': float(p[0]) * float(self.config['max_yaw_deg_per_tick']),
            'forward_probability': float(p[1]),
            'attack_probability': float(p[2]),
            'forward': bool(p[1] >= float(self.config['student_walk_threshold'])),
            'attack': bool(p[2] >= float(self.config['student_attack_threshold'])),
        }

    def observe(self, features, teacher):
        self.samples += 1
        target = self.target(teacher)
        if self.samples % self.config['validation_every_n'] == 0:
            self.validation.add(features, target)
        else:
            self.train.add(features, target)
        if self.train.size >= self.config['warmup_train_samples'] and self.samples % self.config['train_every_samples'] == 0:
            self.fit(self.config['gradient_steps'])
            self.evaluate_validation()

    def fit(self, steps=None):
        if self.train.size < max(16, self.config['batch_size']//2):
            return
        losses = []
        for _ in range(int(steps or self.config['gradient_steps'])):
            x, y = self.train.sample(self.rng, self.config['batch_size'])
            losses.append(self.model.train_batch(x, y)); self.gradient_updates += 1
        self.train_loss = float(np.mean(losses))

    def evaluate_validation(self):
        if self.validation.size < 24:
            return None
        x, y = self.validation.sample(self.rng, self.config['validation_batch'])
        p = self.model.forward(x)
        max_yaw = float(self.config['max_yaw_deg_per_tick'])
        yaw_mae = float(np.mean(np.abs(p[:, 0] - y[:, 0])) * max_yaw)
        walk_acc = float(np.mean((p[:, 1] >= self.config['student_walk_threshold']) == (y[:, 1] >= .5)))
        attack_acc = float(np.mean((p[:, 2] >= self.config['student_attack_threshold']) == (y[:, 2] >= .5)))
        yaw_score = max(0.0, 1.0 - yaw_mae / max_yaw)
        score = float((yaw_score + walk_acc + attack_acc) / 3.0)
        self.metrics = {'yaw_mae_deg': yaw_mae, 'walk_accuracy': walk_acc,
                        'attack_accuracy': attack_acc, 'imitation_score': score}
        self.score_history.append(score)
        if len(self.score_history) > 300: self.score_history = self.score_history[-300:]
        return self.metrics

    def scheduled_student_share(self):
        c = self.config
        if self.samples < c['demo_samples']:
            return 0.0
        return float(np.clip((self.samples-c['demo_samples']) /
                             (c['dagger_full_student_samples']-c['demo_samples']), 0.0, 1.0))

    def student_share(self):
        scheduled = self.scheduled_student_share()
        if scheduled <= 0: return 0.0
        score = float((self.metrics or {}).get('imitation_score', 0.0))
        if score < self.config['quality_gate_1']: cap = max(self.config['minimum_dagger_student_share'], 0.15)
        elif score < self.config['quality_gate_2']: cap = 0.35
        elif score < self.config['quality_gate_3']: cap = 0.70
        else: cap = 1.0
        return min(scheduled, cap)

    def phase(self):
        if self.samples < self.config['demo_samples']: return 'DEMONSTRATE'
        if self.student_share() < 0.999: return 'DAGGER'
        return 'AUTONOMOUS'

    def blend(self, teacher, student):
        share = self.student_share()
        yaw = (1.0-share)*float(teacher['yaw']) + share*float(student['yaw'])
        # Use stochastic switches for discrete controls so labels remain meaningful.
        use_student_walk = bool(self.rng.random() < share)
        use_student_attack = bool(self.rng.random() < share)
        return {
            'yaw': float(yaw),
            'forward': bool(student['forward'] if use_student_walk else teacher['forward']),
            'attack': bool(student['attack'] if use_student_attack else teacher['attack']),
            'student_share': float(share),
        }

    def dashboard(self, teacher=None, student=None, controller=None):
        m = self.metrics or {}
        return {
            'model': MODEL, 'phase': self.phase(), 'samples': int(self.samples),
            'train_samples': int(self.train.size), 'validation_samples': int(self.validation.size),
            'gradient_updates': int(self.gradient_updates), 'feature_dim': int(self.encoder.output_dim),
            'hidden_units': int(self.config['hidden_units']), 'train_loss': self.train_loss,
            'val_yaw_mae_deg': m.get('yaw_mae_deg'), 'val_walk_accuracy': m.get('walk_accuracy'),
            'val_attack_accuracy': m.get('attack_accuracy'), 'imitation_score': m.get('imitation_score'),
            'student_share': float(self.student_share()), 'score_history': list(self.score_history),
            'teacher': dict(teacher or {}), 'student': dict(student or {}), 'controller': dict(controller or {}),
        }

    def save(self, path):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {
            'W1': self.model.W1, 'b1': self.model.b1, 'W2': self.model.W2, 'b2': self.model.b2,
            'ema': self.encoder.ema,
            'metadata': np.array(json.dumps({
                'model': MODEL, 'config': self.config, 'samples': self.samples,
                'gradient_updates': self.gradient_updates, 'train_loss': self.train_loss,
                'metrics': self.metrics, 'optimizer_t': self.model.t,
                'feature_dim': self.encoder.output_dim,
            })),
        }
        for i, (m, v) in enumerate(zip(self.model.m, self.model.v)):
            arrays[f'm{i}'] = m; arrays[f'v{i}'] = v
        np.savez_compressed(path, **arrays)

    def load(self, path):
        with np.load(path, allow_pickle=False) as a:
            meta = json.loads(str(a['metadata']))
            if meta.get('model') != MODEL or int(meta.get('feature_dim', -1)) != self.encoder.output_dim:
                raise ValueError('Readout checkpoint/model feature mismatch')
            for name, target in [('W1',self.model.W1),('b1',self.model.b1),('W2',self.model.W2),('b2',self.model.b2),('ema',self.encoder.ema)]:
                value = a[name]
                if value.shape != target.shape: raise ValueError(f'Checkpoint shape mismatch: {name}')
                target[:] = value
            for i in range(4):
                self.model.m[i][:] = a[f'm{i}']; self.model.v[i][:] = a[f'v{i}']
            self.samples = int(meta.get('samples', 0)); self.gradient_updates = int(meta.get('gradient_updates', 0))
            self.train_loss = meta.get('train_loss'); self.metrics = meta.get('metrics'); self.model.t = int(meta.get('optimizer_t', 0))
            self.encoder.initialized = bool(np.any(self.encoder.ema))
        return meta
