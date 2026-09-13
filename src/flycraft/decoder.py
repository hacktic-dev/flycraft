"""Explicit neural-only bilateral readout; no game state or reward inputs."""
import copy, math
import numpy as np
from doom.engine import NeuralControls

LEGACY = {'mode': 'legacy-bci'}


def _finite_positive(c, names):
    for k in names:
        if not math.isfinite(float(c[k])) or float(c[k]) <= 0:
            raise ValueError(f'Invalid decoder {k}')


def validated_config(config=None):
    c = copy.deepcopy(LEGACY if config is None else config)
    if c.get('mode') == 'legacy-bci':
        if c != LEGACY:
            raise ValueError('Unexpected legacy decoder options')
        return c

    mode = c.get('mode')
    if mode not in ('balanced-bilateral-v1', 'balanced-bilateral-v2', 'balanced-bilateral-v3'):
        raise ValueError('Unknown decoder mode')
    if c.get('neuron_type') not in ('DNp20', 'DNa02', 'MBON11'):
        raise ValueError('Unsupported steering readout')

    _finite_positive(c, ('max_yaw_deg_per_second',))
    if mode == 'balanced-bilateral-v1':
        _finite_positive(c, ('left_reference_hz', 'right_reference_hz', 'regularizer'))
    else:
        for k in ('left_center_hz', 'right_center_hz', 'opponent_offset'):
            if not math.isfinite(float(c[k])):
                raise ValueError(f'Invalid decoder {k}')
        _finite_positive(c, ('left_scale_hz', 'right_scale_hz', 'response_scale'))
        if mode == 'balanced-bilateral-v3':
            if not math.isfinite(float(c.get('command_offset', 0.0))):
                raise ValueError('Invalid decoder command_offset')
            c['command_offset'] = float(c.get('command_offset', 0.0))

    if not math.isfinite(float(c['deadband'])) or not 0 <= float(c['deadband']) < 1:
        raise ValueError('Invalid decoder deadband')
    if c.get('polarity') not in (-1, 1):
        raise ValueError('Decoder polarity must be -1 or +1')
    return c


def _deadband(value, deadband):
    value = float(np.clip(value, -1.0, 1.0))
    deadband = float(deadband)
    if abs(value) <= deadband:
        return 0.0
    return math.copysign((abs(value) - deadband) / (1.0 - deadband), value)


def command_from_rates(left_hz, right_hz, config):
    """Pure steering transform used by the live decoder and calibration tools."""
    c = validated_config(config)
    if c['mode'] == 'legacy-bci':
        raise ValueError('Legacy decoder has no bilateral steering transform')

    left = float(left_hz)
    right = float(right_hz)
    if c['mode'] == 'balanced-bilateral-v1':
        l = left / float(c['left_reference_hz'])
        r = right / float(c['right_reference_hz'])
        opponent = (r - l) / (r + l + float(c['regularizer']))
        raw_command = opponent
        extra = {'left_normalized': l, 'right_normalized': r, 'raw_difference': r - l}
    else:
        # V2/V3 share the same bilateral rate normalization. V3 adds one final
        # scalar motor intercept learned from *closed-loop* Minecraft trajectories.
        # Keeping this intercept after tanh makes it directly interpretable in
        # normalized steering-command units and leaves the response shape intact.
        l = (left - float(c['left_center_hz'])) / float(c['left_scale_hz'])
        r = (right - float(c['right_center_hz'])) / float(c['right_scale_hz'])
        raw_difference = r - l
        centered = raw_difference - float(c['opponent_offset'])
        opponent = math.tanh(centered / float(c['response_scale']))
        command_offset = float(c.get('command_offset', 0.0)) if c['mode'] == 'balanced-bilateral-v3' else 0.0
        raw_command = opponent - command_offset
        extra = {
            'left_normalized': l,
            'right_normalized': r,
            'raw_difference': raw_difference,
            'centered_difference': centered,
            'command_offset': command_offset,
            'pre_deadband_command': raw_command,
        }

    command = _deadband(raw_command, float(c['deadband']))
    command *= int(c['polarity'])
    return command, opponent, extra


def _robust_scale(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) < 4 or not np.all(np.isfinite(values)):
        raise ValueError('Calibration requires at least four finite samples per side')
    center = float(np.mean(values))
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median))) * 1.4826
    std = float(np.std(values))
    # Prefer MAD, but firing rates can be heavily quantised at low activity.  The
    # fallback prevents a near-zero scale from exploding small count differences.
    scale = mad if mad > 0.25 else std
    scale = max(scale, 1.0)
    return center, scale


def _mean_command_for_offset(raw_difference, offset, response_scale, deadband):
    squashed = np.tanh((raw_difference - offset) / response_scale)
    mag = np.maximum(0.0, np.abs(squashed) - deadband) / (1.0 - deadband)
    return float(np.mean(np.sign(squashed) * mag))


def solve_command_offset(opponents, *, deadband=0.05, polarity=1):
    """Find a scalar V3 motor intercept whose mean command is zero.

    ``opponents`` must come from the live decoder *before* the V3 command offset.
    The solve includes deadband and polarity, so it matches the quantity that is
    ultimately sent to Minecraft rather than merely centering neural statistics.
    """
    values = np.asarray(opponents, dtype=np.float64)
    if values.ndim != 1 or len(values) < 8 or not np.all(np.isfinite(values)):
        raise ValueError('Closed-loop calibration requires >=8 finite opponent samples')
    if not 0 <= float(deadband) < 1:
        raise ValueError('deadband must be in [0,1)')
    if int(polarity) not in (-1, 1):
        raise ValueError('polarity must be -1 or +1')

    def mean_command(offset):
        shifted = np.clip(values - float(offset), -1.0, 1.0)
        mag = np.maximum(0.0, np.abs(shifted) - float(deadband)) / (1.0 - float(deadband))
        commands = np.sign(shifted) * mag * int(polarity)
        return float(np.mean(commands))

    # The post-deadband mean is monotonic in the subtracted offset. A generous
    # bracket beyond the observed opponent range also handles all-deadband banks.
    lo = float(np.min(values) - 1.0)
    hi = float(np.max(values) + 1.0)
    for _ in range(80):
        mid = (lo + hi) / 2.0
        m = mean_command(mid)
        if m * int(polarity) > 0:
            lo = mid
        else:
            hi = mid
    offset = (lo + hi) / 2.0
    return {
        'command_offset': float(offset),
        'mean_command_after_fit': float(mean_command(offset)),
        'sample_count': int(len(values)),
    }


def calibrate_rate_samples(left_rates, right_rates, *, deadband=0.05,
                           target_median_abs_command=0.25):
    """Fit balanced-bilateral-v2 parameters to frozen natural-scene responses.

    Centers/scales are estimated independently per side.  A symmetric response
    scale preserves useful steering variation, and a final opponent offset is
    solved so the mean post-deadband command over the calibration bank is zero.
    """
    left = np.asarray(left_rates, dtype=np.float64)
    right = np.asarray(right_rates, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1 or len(left) < 8:
        raise ValueError('Calibration requires matching 1-D left/right samples (>=8)')
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError('Calibration samples must be finite')
    if not 0 <= float(deadband) < 1:
        raise ValueError('deadband must be in [0,1)')
    if not 0 < float(target_median_abs_command) < 1:
        raise ValueError('target_median_abs_command must be in (0,1)')

    left_center, left_scale = _robust_scale(left)
    right_center, right_scale = _robust_scale(right)
    l = (left - left_center) / left_scale
    r = (right - right_center) / right_scale
    diff = r - l

    desired_pre_deadband = float(deadband) + float(target_median_abs_command) * (1.0 - float(deadband))
    desired_pre_deadband = min(desired_pre_deadband, 0.95)
    target_atanh = math.atanh(desired_pre_deadband)
    median_abs = float(np.median(np.abs(diff - np.median(diff))))
    response_scale = max(0.25, median_abs / max(target_atanh, 1e-6))

    # Mean command is monotonic in the subtracted offset.  Solve the actual
    # post-tanh/post-deadband mean, rather than merely forcing raw z-score mean 0.
    lo = float(np.min(diff) - 10.0 * response_scale)
    hi = float(np.max(diff) + 10.0 * response_scale)
    for _ in range(80):
        mid = (lo + hi) / 2.0
        mean_command = _mean_command_for_offset(diff, mid, response_scale, float(deadband))
        if mean_command > 0:
            lo = mid
        else:
            hi = mid
    offset = (lo + hi) / 2.0

    commands = []
    for lv, rv in zip(left, right):
        cfg = {
            'mode': 'balanced-bilateral-v2', 'neuron_type': 'DNp20',
            'left_center_hz': left_center, 'right_center_hz': right_center,
            'left_scale_hz': left_scale, 'right_scale_hz': right_scale,
            'opponent_offset': offset, 'response_scale': response_scale,
            'deadband': float(deadband), 'max_yaw_deg_per_second': 120.0, 'polarity': 1,
        }
        commands.append(command_from_rates(lv, rv, cfg)[0])
    commands = np.asarray(commands, dtype=np.float64)

    return {
        'left_center_hz': left_center,
        'right_center_hz': right_center,
        'left_scale_hz': left_scale,
        'right_scale_hz': right_scale,
        'opponent_offset': float(offset),
        'response_scale': float(response_scale),
        'calibration_mean_command': float(commands.mean()),
        'calibration_median_abs_command': float(np.median(np.abs(commands))),
        'sample_count': int(len(left)),
    }


class BilateralControls(NeuralControls):
    def __init__(self, readouts, config):
        super().__init__(readouts, mode='bci')
        self.config = validated_config(config)
        self.side_indices = {
            side: [i for i, r in enumerate(readouts)
                   if r['type'] == self.config['neuron_type'] and r['side'] == side]
            for side in ('L', 'R')
        }
        if any(len(ix) != 1 for ix in self.side_indices.values()):
            raise ValueError('Balanced decoder requires exactly one annotated cell per side')

    def decode(self, counts, seconds):
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError('Positive finite interval required')
        # Preserve the existing 100 ms firing-rate filter, readout serialization,
        # forward and attack mappings. Only bilateral steering is replaced.
        out = super().decode(counts, seconds)
        c = self.config
        left = float(self.rates[self.side_indices['L'][0]])
        right = float(self.rates[self.side_indices['R'][0]])
        command, opponent, extra = command_from_rates(left, right, c)
        out['turn'] = command * float(c['max_yaw_deg_per_second']) * seconds
        out['decoder'] = {
            'mode': c['mode'], 'neuron_type': c['neuron_type'],
            'turn_units': 'degrees_per_tick', 'left_hz': left, 'right_hz': right,
            'opponent': opponent, 'deadband': c['deadband'],
            'max_yaw_deg_per_second': c['max_yaw_deg_per_second'], **extra,
        }
        return out


def make_decoder(readouts, config=None):
    c = validated_config(config)
    if c['mode'] == 'legacy-bci':
        return NeuralControls(readouts, mode='bci')
    return BilateralControls(readouts, c)


def signature(decoder):
    return copy.deepcopy(getattr(decoder, 'config', LEGACY))
