"""Experimental action-conditioned steering credit on existing KC->MBON11 edges.

This module is deliberately an engineered learning rule.  It does not claim that
Drosophila implements Minecraft steering this way.

The important distinction from the Run-2D rule is causal credit:

* eligibility is built only from the sparse/transient KC winners already selected
  by ``FatigueSelectiveMemoryBrain._select_credit``;
* the trace is split by the action that was actually expressed (left/right yaw);
* reward prediction error updates synapses according to whether the MBON channel
  is calibrated to promote the action that just succeeded;
* each MBON channel is renormalised to its original total KC input so learning
  redistributes sensory preference instead of globally increasing/decreasing drive.

Two steering arrangements are supported:

``direct-opponent``
    The game decoder reads bilateral MBON11 directly.  The causal sign is known
    from decoder polarity and therefore needs no extra calibration.  This is the
    cleanest proof-of-learning experiment, but it bypasses downstream DNs.

``calibrated-downstream``
    The game can keep a downstream readout such as DNp20.  The caller must provide
    the measured sign of each MBON11 side's causal effect on game yaw.  Do not infer
    these signs from anatomy alone; obtain them from a controlled perturbation.
"""
import copy
import math

import numpy as np

MODEL = 'action-conditioned-steering-v2'

DEFAULTS = {
    'model': MODEL,
    'mode': 'direct-opponent',
    'tau_ms': 300.0,
    'prediction_tau_ms': 10000.0,
    'max_yaw_deg_per_tick': 6.0,
    'action_deadband_deg': 0.10,
    'eta': 0.02,
    'minimum_fraction': 0.5,
    'maximum_fraction': 1.5,
    'max_log_update': 0.002,
}


def validated_config(config):
    c = {**DEFAULTS, **copy.deepcopy(config or {})}
    if c.get('model') != MODEL:
        raise ValueError(f'Unknown action-credit model: {c.get("model")}')
    if c.get('mode') not in ('direct-opponent', 'calibrated-downstream'):
        raise ValueError('Unknown action-credit mode')
    for k in ('tau_ms', 'prediction_tau_ms', 'max_yaw_deg_per_tick', 'eta',
              'minimum_fraction', 'maximum_fraction', 'max_log_update'):
        if not math.isfinite(float(c[k])) or float(c[k]) <= 0:
            raise ValueError(f'Invalid action credit {k}')
    if not math.isfinite(float(c['action_deadband_deg'])) or float(c['action_deadband_deg']) < 0:
        raise ValueError('Invalid action credit action_deadband_deg')
    if not float(c['minimum_fraction']) < 1 < float(c['maximum_fraction']):
        raise ValueError('Action-credit bounds must contain baseline')

    if c['mode'] == 'calibrated-downstream':
        effects = c.get('mbon_turn_effect')
        if not isinstance(effects, dict) or set(effects) != {'L', 'R'}:
            raise ValueError('calibrated-downstream requires mbon_turn_effect={"L": +/-1, "R": +/-1}')
        if any(effects[s] not in (-1, 1) for s in ('L', 'R')):
            raise ValueError('Each calibrated MBON turn effect must be -1 or +1')
        if effects['L'] == effects['R']:
            raise ValueError('Calibrated MBON channels must have opponent turn effects')
        c['mbon_turn_effect'] = {s: int(effects[s]) for s in ('L', 'R')}
    else:
        # Ignore stale downstream calibration when direct MBON opponent steering is used.
        c.pop('mbon_turn_effect', None)
    return c


def validate(config):
    validated_config(config)


def _renormalize(weights, baseline, lo, hi):
    """Preserve a channel's original total KC input subject to per-edge bounds."""
    weights = np.asarray(weights, dtype=np.float64)
    baseline = np.asarray(baseline, dtype=np.float64)
    target = float(np.sum(baseline, dtype=np.float64))
    lower = baseline * float(lo)
    upper = baseline * float(hi)

    # Find a scalar multiplier after the local update that restores the original
    # channel total while respecting hard bounds.  Monotonicity makes bisection
    # robust and keeps this independent of edge count / baseline magnitudes.
    left = 0.0
    right = max(2.0, float(np.max(upper / np.maximum(weights, 1e-30))))
    for _ in range(48):
        mid = (left + right) / 2.0
        total = float(np.clip(weights * mid, lower, upper).sum(dtype=np.float64))
        if total < target:
            left = mid
        else:
            right = mid
    return np.clip(weights * ((left + right) / 2.0), lower, upper)


class ActionCredit:
    def __init__(self, brain, config, decoder_config=None):
        self.b = brain
        self.config = validated_config(config)
        c = self.config

        cells = brain.circuit['report']['MBON']
        by_side = {s: [x for x in cells if x['soma_side'] == s] for s in ('L', 'R')}
        if any(len(by_side[s]) != 1 for s in ('L', 'R')):
            raise ValueError('Action credit requires exactly one MBON11 cell per side')
        self.sides = {s: int(by_side[s][0]['index']) for s in ('L', 'R')}

        self.edges = brain.circuit['edges']
        post = brain.post[self.edges]
        self.groups = {s: np.flatnonzero(post == i) for s, i in self.sides.items()}
        if any(not len(ix) for ix in self.groups.values()):
            raise ValueError('Both MBON11 channels require existing KC inputs')

        if c['mode'] == 'direct-opponent':
            d = decoder_config or {}
            if d.get('neuron_type') != 'MBON11':
                raise ValueError('direct-opponent action credit requires the MBON11 bilateral decoder')
            polarity = int(d.get('polarity', 1))
            if polarity not in (-1, 1):
                raise ValueError('Decoder polarity must be -1 or +1')
            # Decoder command is polarity * (R - L): increasing R promotes
            # +yaw*polarity and increasing L promotes -yaw*polarity.
            self.turn_effect = {'L': -polarity, 'R': polarity}
        else:
            self.turn_effect = dict(c['mbon_turn_effect'])

        # Store eligibility per *plastic edge*, not merely per KC.  This preserves
        # the actual postsynaptic channel and avoids accidentally giving the same
        # directional credit to both bilateral MBON11 targets of one KC.
        arrays = {
            'action_eligibility': np.zeros((2, len(self.edges)), np.float64),  # 0=left action, 1=right action
            'action_expected': np.zeros(1, np.float64),
            'action_ready': np.zeros(1, np.int32),
            'action_last_rpe': np.zeros(1, np.float64),
            'action_update_events': np.zeros(len(self.edges), np.int32),
            'action_signed_exposure': np.zeros(len(self.edges), np.float64),
            'action_absolute_exposure': np.zeros(len(self.edges), np.float64),
        }
        for name, array in arrays.items():
            if hasattr(brain, name):
                raise ValueError(f'Brain already has action-credit field {name}')
            setattr(brain, name, array)
            brain.fields.append(name)
            brain.initial[name] = array.copy()

        original_signature = brain.configuration_signature
        brain.configuration_signature = lambda: {
            **original_signature(),
            'action_credit': self.config,
            'action_mbon_sides': self.sides,
            'action_mbon_turn_effect': self.turn_effect,
        }

        # Episode reset clears fast action eligibility but preserves the slowly
        # learned reward-prediction baseline and lifetime diagnostics.
        original_reset = brain.reset
        persistent = ('action_expected', 'action_update_events',
                      'action_signed_exposure', 'action_absolute_exposure')

        def reset(keep_memory=False):
            saved = {name: getattr(brain, name).copy() for name in persistent} if keep_memory else None
            original_reset(keep_memory=keep_memory)
            if saved is not None:
                for name, value in saved.items():
                    getattr(brain, name)[:] = value

        brain.reset = reset

        original_memory = brain.memory

        def memory():
            out = original_memory()
            e = brain.action_eligibility
            exposure_q = np.quantile(brain.action_absolute_exposure, [0.10, 0.50, 0.90])
            events_q = np.quantile(brain.action_update_events.astype(np.float64), [0.10, 0.50, 0.90])
            out.update({
                'action_credit_model': MODEL,
                'action_credit_mode': self.config['mode'],
                'expected_aim_improvement': float(brain.action_expected[0]),
                'last_rpe': float(brain.action_last_rpe[0]),
                'left_action_eligibility': float(e[0].sum()),
                'right_action_eligibility': float(e[1].sum()),
                'action_update_events': int(brain.action_update_events.sum()),
                'action_absolute_exposure': float(brain.action_absolute_exposure.sum()),
                'action_signed_exposure': float(brain.action_signed_exposure.sum()),
                # Reuse the generic dashboard keys so health logs continue to
                # describe the active learning rule rather than the disabled
                # Run-2D global teaching exposure.
                'credit_exposure_p10': float(exposure_q[0]),
                'credit_exposure_median': float(exposure_q[1]),
                'credit_exposure_p90': float(exposure_q[2]),
                'credit_exposure_p90_p10_span': float(exposure_q[2] - exposure_q[0]),
                'credit_update_events_p10': float(events_q[0]),
                'credit_update_events_median': float(events_q[1]),
                'credit_update_events_p90': float(events_q[2]),
                'signed_credit_mean': float(brain.action_signed_exposure.mean()),
                'signed_credit_std': float(brain.action_signed_exposure.std()),
                'mbon_turn_effect_L': int(self.turn_effect['L']),
                'mbon_turn_effect_R': int(self.turn_effect['R']),
            })
            return out

        brain.memory = memory

    def observe(self, counts, yaw_degrees, seconds=0.05):
        """Attach sparse sensory eligibility to the action actually expressed."""
        if not math.isfinite(yaw_degrees) or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError('Invalid action interval')
        b = self.b
        c = self.config
        if b.action_ready[0]:
            raise RuntimeError('Previous action has not received its outcome')

        # Eligibility decays even on a straight/no-op action, but only a real yaw
        # above the deadband adds new left/right action credit.
        decay = math.exp(-seconds * 1000.0 / float(c['tau_ms']))
        b.action_eligibility *= decay
        amount = abs(float(yaw_degrees))
        if amount > float(c['action_deadband_deg']):
            edge_activity, detail = b._select_credit()
            # Keep the existing memory/dashboard diagnostics honest even though
            # the old global teach() path is no longer called in this mode.
            if hasattr(b, 'last_credit'):
                b.last_credit = detail
            magnitude = min(1.0, amount / float(c['max_yaw_deg_per_tick']))
            channel = 1 if yaw_degrees > 0 else 0
            b.action_eligibility[channel] += (1.0 - decay) * edge_activity * magnitude

        b.action_ready[0] = 1

    def teach(self, improvement, seconds=0.05):
        """Apply aim RPE to action-conditioned eligibility.

        ``improvement`` is the bounded signed aim component for the immediately
        preceding action.  Positive means the action reduced angular error.
        """
        b = self.b
        c = self.config
        improvement = float(improvement)
        if not math.isfinite(improvement) or not -1.000001 <= improvement <= 1.000001:
            raise ValueError('Bounded finite aim improvement required')
        if not b.action_ready[0]:
            raise RuntimeError('Record the action before its outcome')

        rpe = improvement - float(b.action_expected[0])
        alpha = -math.expm1(-seconds * 1000.0 / float(c['prediction_tau_ms']))
        b.action_expected[0] += alpha * rpe

        # Positive directional trace means "this edge was eligible during a right
        # action"; negative means left.  Multiplying by the calibrated effect of
        # the edge's postsynaptic MBON converts that into "this edge supported the
        # action that was taken".  RPE then strengthens successful support and
        # weakens support for unsuccessful actions.
        directional = b.action_eligibility[1] - b.action_eligibility[0]
        effect = np.empty(len(self.edges), dtype=np.float64)
        for side, indices in self.groups.items():
            effect[indices] = float(self.turn_effect[side])
        policy_credit = directional * effect

        current = b.weight[self.edges].astype(np.float64)
        baseline = b.baseline_plastic.astype(np.float64)
        log_delta = np.clip(
            float(c['eta']) * rpe * policy_credit,
            -float(c['max_log_update']),
            float(c['max_log_update']),
        )
        proposed = current * np.exp(log_delta)

        # Competition/homeostasis is local to each MBON channel: one channel
        # cannot steal total synaptic drive from the other merely by drifting.
        for side, indices in self.groups.items():
            proposed[indices] = _renormalize(
                proposed[indices], baseline[indices],
                float(c['minimum_fraction']), float(c['maximum_fraction'])
            )
        b.weight[self.edges] = proposed.astype(b.weight.dtype)

        exposure = rpe * policy_credit
        active = np.abs(log_delta) > 0
        b.action_update_events += active.astype(np.int32)
        b.action_signed_exposure += exposure
        b.action_absolute_exposure += np.abs(exposure)
        b.action_last_rpe[0] = rpe
        b.action_ready[0] = 0

        return {
            'signal': improvement,
            'rpe': float(rpe),
            'expected_improvement': float(b.action_expected[0]),
            'active_edges': int(np.count_nonzero(active)),
            'left_action_eligibility': float(b.action_eligibility[0].sum()),
            'right_action_eligibility': float(b.action_eligibility[1].sum()),
            'mean_abs_policy_credit': float(np.mean(np.abs(policy_credit))) if len(policy_credit) else 0.0,
            'max_abs_policy_credit': float(np.max(np.abs(policy_credit))) if len(policy_credit) else 0.0,
            'mbon_turn_effect': dict(self.turn_effect),
        }
