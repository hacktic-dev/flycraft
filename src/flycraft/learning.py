"""Run-2D fatigue-selective signed learning adapter with pinned DOOMFLY v6 RGB vision."""
import ctypes
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import doom_learning.brain as native_upstream
from doom.engine import NeuralControls
from doom.game import retinal_samples

from .r8_visual import R8VisualMemoryBrain

ROOT = Path(__file__).resolve().parents[2]


def prepare_kernel():
    """Build the existing native neural kernel plus the read-only spike hook.

    Run-2D plasticity is applied outside this kernel. Native v1 LTD is always
    called with learning_enabled=0 by R8VisualMemoryBrain.
    """
    folder = ROOT/'.tools/learning-kernel'
    folder.mkdir(parents=True, exist_ok=True)
    source = native_upstream.SOURCE.read_text()
    assert source.count('counts[i]++;') == 1
    source = source.replace(
        'extern "C" void memory_advance(',
        'static int32_t* observed_bins=nullptr;\nextern "C" void set_observed_bins(int32_t* p){observed_bins=p;}\nextern "C" void memory_advance('
    )
    source = source.replace(
        'counts[i]++;',
        'counts[i]++;if(observed_bins)observed_bins[((int)((*clock)*dt*60.f/1000.f)%3)*n+i]++;'
    )
    digest = hashlib.sha256(source.encode()).hexdigest()
    dll = folder/'memory.dll'
    meta = folder/'build.json'
    record = json.loads(meta.read_text()) if meta.exists() else {}
    if (not dll.exists() or record.get('observer_sha256') != digest or
            record.get('binary_sha256') != hashlib.sha256(dll.read_bytes()).hexdigest()):
        cpp = folder/'kernel.cpp'
        cpp.write_text(source)
        compiler = next((ROOT/'.tools').glob('llvm-mingw-*/bin/clang++.exe'))
        subprocess.run([
            str(compiler), '-O3', '-std=c++17', '-shared', '-static',
            '-Wl,--export-all-symbols', str(cpp), '-o', str(dll)
        ], check=True)
        record = {
            'model': 'gamma1-fatigue-selective-signed-v4-neural-runtime',
            'source_sha256': hashlib.sha256(native_upstream.SOURCE.read_bytes()).hexdigest(),
            'observer_sha256': digest,
            'binary_sha256': hashlib.sha256(dll.read_bytes()).hexdigest(),
        }
        meta.write_text(json.dumps(record, indent=2))
    # The native binary is intentionally unchanged by v3; refresh provenance even
    # when an already-correct observer DLL can be reused.
    if record.get('model') != 'gamma1-fatigue-selective-signed-v4-neural-runtime':
        record['model'] = 'gamma1-fatigue-selective-signed-v4-neural-runtime'
        meta.write_text(json.dumps(record, indent=2))
    # SelectiveMemoryBrain ultimately calls legacy MemoryBrain.__init__, so patch
    # the legacy module's runtime exactly as before.
    native_upstream.LIBRARY = dll
    native_upstream.build = lambda: record
    return record


class LearningFly:
    def __init__(self, config, decoder_config=None):
        prepare_kernel()
        self.config = config
        self.brain = R8VisualMemoryBrain(
            eta=config['eta'],
            eligibility_tau_ms=config['eligibility_tau_ms'],
            eligibility_baseline_tau_ms=config['eligibility_baseline_tau_ms'],
            eligibility_reference_hz=config['eligibility_reference_hz'],
            eligibility_gate_hz=config['eligibility_gate_hz'],
            eligibility_winner_fraction=config['eligibility_winner_fraction'],
            eligibility_max_kcs=config['eligibility_max_kcs'],
            eligibility_activity_floor=config['eligibility_activity_floor'],
            selection_fatigue_tau_ms=config['selection_fatigue_tau_ms'],
            selection_fatigue_strength=config['selection_fatigue_strength'],
            recovery_tau_seconds=config['recovery_tau_seconds'],
            minimum_fraction=config['minimum_fraction'],
            maximum_fraction=config['maximum_fraction'],
            max_log_update_per_event=config['max_log_update_per_event'],
            dt=float(config.get('dt_ms', 0.5)),
        )
        if os.environ.get('FLYCRAFT_REFERENCE_KERNEL') != '1':
            from .fast_kernel import install
            install(self.brain)

        self.learning = True
        self.pending_ticks = 0
        self.cooldown_ticks = 0
        self.delivered_ticks = 0
        self.positive_events = 0
        self.negative_events = 0
        self.neutral_events = 0
        self.teaching_abs_total = 0.0
        self.last_reinforcement = {
            'signal': 0.0,
            'scheduled_aversive': False,
            'plasticity': {'active_edges': 0, 'mean_activity': 0.0, 'max_activity': 0.0},
        }

        self.manifest = json.loads((native_upstream.GRAPH.parent/'manifest.json').read_text())
        from .decoder import make_decoder
        self.action_credit = None
        readouts = list(self.manifest['readouts'])
        if config.get('action_credit'):
            from .action_credit import ActionCredit
            self.action_credit = ActionCredit(self.brain, config['action_credit'], decoder_config)
            for cell in self.brain.circuit['report']['MBON']:
                readouts.append({'index':cell['index'],'id':cell['id'],'type':'MBON11','side':cell['soma_side']})
        self.decoder = make_decoder(readouts, decoder_config)
        self.initial_weights = hashlib.sha256(self.brain.weight.tobytes()).hexdigest()
        lock = json.loads((ROOT/'model-lock.json').read_text(encoding='utf-8-sig'))
        assert self.brain.pre_visual_weight_sha256 == lock['weight_sha256']

        self.samples = None
        self.last_voltage = self.brain.v.copy()
        self.activity_bins = None
        self.bins = np.zeros((3, self.brain.n), np.int32)
        self.set_bins = self.brain.library.set_observed_bins
        self.set_bins.argtypes = [ctypes.c_void_p]
        self.set_bins.restype = None

    def reset_episode(self):
        """Clear fast neural/controller state while preserving learned weights."""
        self.brain.reset(keep_memory=True)
        self.decoder.rates.fill(0)
        self.pending_ticks = 0
        self.cooldown_ticks = 0
        self.last_reinforcement = {
            'signal': 0.0,
            'scheduled_aversive': False,
            'plasticity': {'active_edges': 0, 'mean_activity': 0.0, 'max_activity': 0.0},
        }

    def step(self, rgb, record=False):
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError('Expected RGB only')
        self.samples = retinal_samples(rgb, self.brain.uv)
        aversive_active = self.learning and self.pending_ticks > 0
        self.bins.fill(0)
        self.set_bins(self.bins.ctypes.data if record else None)

        # Evaluation/replay must be observationally frozen, including traces.
        frozen_names = ('eligibility', 'eligibility_last', 'modulation', 'modulation_last',
                        'credit_trace', 'credit_baseline', 'credit_age_ms',
                        'selection_fatigue')
        frozen = {k: getattr(self.brain, k).copy() for k in frozen_names} if not self.learning else {}
        try:
            counts, wall = self.brain.rgb_step(
                rgb, 50., learning=self.learning, _retinal_samples=self.samples,
                stimulation=(self.brain.circuit['dan'], self.config['current']) if aversive_active else None
            )
        finally:
            self.set_bins(None)
        for k, value in frozen.items():
            getattr(self.brain, k)[:] = value

        if aversive_active:
            self.pending_ticks -= 1
            self.delivered_ticks += 1
            if self.pending_ticks == 0:
                self.cooldown_ticks = self.config['cooldown_ticks']
        elif self.cooldown_ticks > 0:
            self.cooldown_ticks -= 1

        self.last_counts = counts
        self.last_voltage = self.brain.v.copy()
        self.activity_bins = self.bins if record else None
        if record:
            assert np.array_equal(self.bins.sum(axis=0), counts)

        return self.decoder.decode(counts, .05), {
            'neural_ms': self.brain.sim_ms,
            'neural_compute_seconds': wall,
            'spikes_sha256': hashlib.sha256(counts.tobytes()).hexdigest(),
            'aversive_applied': bool(aversive_active),
            'aversive_current': self.config['current'] if aversive_active else 0.,
            'aversive_pending_ticks': self.pending_ticks,
            'aversive_cooldown_ticks': self.cooldown_ticks,
            'DAN_spikes': counts[self.brain.circuit['dan']].tolist(),
            'KC_spikes': int(counts[self.brain.circuit['kc']].sum()),
            'R8_inputs': int(len(self.brain.r8)),
            'R8_light_mean': float(self.brain.r8_light.mean()),
            'R8_light_max': float(self.brain.r8_light.max()),
        }

    def record_action(self, yaw_degrees):
        if self.learning and self.action_credit:
            self.action_credit.observe(self.last_counts, yaw_degrees)

    def reinforce(self, signal, aim_signal=None):
        """Apply signed plastic teaching and optionally schedule aversive PPL101.

        Positive signals directly potentiate eligible KC->MBON11 edges. Negative
        signals depress them. Only sufficiently strong negative signals schedule
        a short PPL101 current pulse, and pulses cannot be extended while active.
        """
        signal = float(signal)
        if not self.learning:
            return self.last_reinforcement

        if self.action_credit:
            if aim_signal is None:raise ValueError('Action credit requires the aim outcome separately')
            signal = float(aim_signal)
            detail = self.action_credit.teach(signal)
        else:
            detail = self.brain.teach(signal)
        scheduled = False
        if signal > 0:
            self.positive_events += 1
        elif signal < 0:
            self.negative_events += 1
        else:
            self.neutral_events += 1
        self.teaching_abs_total += abs(signal)

        if (not self.action_credit and signal <= -self.config['aversive_signal_threshold'] and
                self.pending_ticks == 0 and self.cooldown_ticks == 0):
            self.pending_ticks = self.config['pulse_ticks']
            scheduled = True

        self.last_reinforcement = {
            'signal': signal,
            'scheduled_aversive': scheduled,
            'plasticity': detail,
        }
        return self.last_reinforcement

    def state(self):
        return {
            'pending_ticks': self.pending_ticks,
            'cooldown_ticks': self.cooldown_ticks,
            'delivered_ticks': self.delivered_ticks,
            'positive_events': self.positive_events,
            'negative_events': self.negative_events,
            'neutral_events': self.neutral_events,
            'teaching_abs_total': self.teaching_abs_total,
        }

    def restore_state(self, state):
        self.pending_ticks = int(state.get('pending_ticks', 0))
        self.cooldown_ticks = int(state.get('cooldown_ticks', 0))
        self.delivered_ticks = int(state.get('delivered_ticks', 0))
        self.positive_events = int(state.get('positive_events', 0))
        self.negative_events = int(state.get('negative_events', 0))
        self.neutral_events = int(state.get('neutral_events', 0))
        self.teaching_abs_total = float(state.get('teaching_abs_total', 0.0))

    def freeze(self):
        self.learning = False
        self.pending_ticks = 0
        self.cooldown_ticks = 0


if __name__ == '__main__':
    print(prepare_kernel())
