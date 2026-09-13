"""Dependency-light checks for FlyCraft action-conditioned credit."""
import copy
import importlib.util
from pathlib import Path
import numpy as np

MODULE = Path(__file__).resolve().parents[1] / 'src' / 'flycraft' / 'action_credit.py'
spec = importlib.util.spec_from_file_location('action_credit', MODULE)
ac = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ac)


class FakeBrain:
    def __init__(self):
        # Two KC inputs per MBON side.  Edge positions 0/1 -> L, 2/3 -> R.
        self.circuit = {
            'edges': np.arange(4, dtype=np.int64),
            'kc': np.arange(4, dtype=np.int32),
            'report': {'MBON': [
                {'index': 10, 'soma_side': 'L'},
                {'index': 11, 'soma_side': 'R'},
            ]},
        }
        self.post = np.array([10, 10, 11, 11], dtype=np.int32)
        self.weight = np.ones(4, dtype=np.float64)
        self.baseline_plastic = self.weight.copy()
        self.fields = []
        self.initial = {}
        self._edge_activity = np.array([1.0, 0.0, 1.0, 0.0])

    def _select_credit(self):
        return self._edge_activity.copy(), {'active_edges': int(np.count_nonzero(self._edge_activity))}

    def configuration_signature(self):
        return {'fake': True}

    def reset(self, keep_memory=False):
        for k, v in self.initial.items():
            getattr(self, k)[:] = v
        if not keep_memory:
            self.weight[:] = self.baseline_plastic

    def memory(self):
        f = self.weight / self.baseline_plastic
        return {'mean_efficacy': float(f.mean())}


def cfg(mode='direct-opponent', **extra):
    out = {'model': ac.MODEL, 'mode': mode, 'eta': 0.1, 'max_log_update': 0.05}
    out.update(extra)
    return out


def decoder(polarity=1, typ='MBON11'):
    return {'mode': 'balanced-bilateral-v1', 'neuron_type': typ, 'polarity': polarity}


def assert_channel_totals_preserved(brain):
    assert np.isclose(brain.weight[:2].sum(), 2.0)
    assert np.isclose(brain.weight[2:].sum(), 2.0)


def test_right_success_reinforces_right_support():
    b = FakeBrain(); rule = ac.ActionCredit(b, cfg(), decoder(+1))
    rule.observe(np.zeros(1), +6.0)
    detail = rule.teach(+0.8)
    # For polarity +1, R supports +yaw and L opposes it.  Selected R input should
    # rise relative to its same-channel competitor; selected L should fall.
    assert b.weight[2] > b.weight[3], b.weight
    assert b.weight[0] < b.weight[1], b.weight
    assert detail['rpe'] > 0
    assert_channel_totals_preserved(b)


def test_left_success_is_opposite():
    b = FakeBrain(); rule = ac.ActionCredit(b, cfg(), decoder(+1))
    rule.observe(np.zeros(1), -6.0)
    rule.teach(+0.8)
    assert b.weight[0] > b.weight[1], b.weight
    assert b.weight[2] < b.weight[3], b.weight
    assert_channel_totals_preserved(b)


def test_decoder_polarity_is_respected():
    b = FakeBrain(); rule = ac.ActionCredit(b, cfg(), decoder(-1))
    rule.observe(np.zeros(1), +6.0)
    rule.teach(+0.8)
    # Polarity -1 reverses which MBON channel supports positive yaw.
    assert b.weight[0] > b.weight[1], b.weight
    assert b.weight[2] < b.weight[3], b.weight
    assert_channel_totals_preserved(b)


def test_failed_right_action_reverses_update():
    b = FakeBrain(); rule = ac.ActionCredit(b, cfg(), decoder(+1))
    rule.observe(np.zeros(1), +6.0)
    rule.teach(-0.8)
    assert b.weight[2] < b.weight[3], b.weight
    assert b.weight[0] > b.weight[1], b.weight
    assert_channel_totals_preserved(b)


def test_downstream_mode_requires_explicit_causal_signs():
    try:
        ac.validated_config(cfg('calibrated-downstream'))
    except ValueError:
        pass
    else:
        raise AssertionError('Missing downstream MBON effect calibration was accepted')

    b = FakeBrain()
    rule = ac.ActionCredit(
        b,
        cfg('calibrated-downstream', mbon_turn_effect={'L': 1, 'R': -1}),
        decoder(+1, 'DNp20'),
    )
    assert rule.turn_effect == {'L': 1, 'R': -1}


def test_episode_reset_preserves_slow_prediction_and_diagnostics_only():
    b = FakeBrain(); rule = ac.ActionCredit(b, cfg(), decoder(+1))
    rule.observe(np.zeros(1), +6.0); rule.teach(+0.8)
    expected = float(b.action_expected[0]); events = b.action_update_events.copy(); weights = b.weight.copy()
    b.action_eligibility[:] = 123
    b.action_ready[0] = 1
    b.reset(keep_memory=True)
    assert float(b.action_expected[0]) == expected
    assert np.array_equal(b.action_update_events, events)
    assert np.array_equal(b.weight, weights)
    assert not np.any(b.action_eligibility)
    assert b.action_ready[0] == 0


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    for test in tests:
        test(); print('PASS', test.__name__)
    print(f'{len(tests)} action-credit tests passed')
