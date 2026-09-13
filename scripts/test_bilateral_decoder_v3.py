"""Cheap numerical tests for the closed-loop V3 steering intercept."""
import math
import numpy as np

from flycraft.decoder import command_from_rates, solve_command_offset, _deadband


def config(offset=0.0):
    return {
        'mode': 'balanced-bilateral-v3',
        'neuron_type': 'DNp20',
        'left_center_hz': 7.0,
        'right_center_hz': 30.0,
        'left_scale_hz': 4.0,
        'right_scale_hz': 8.0,
        'opponent_offset': 0.0,
        'response_scale': 1.0,
        'command_offset': offset,
        'deadband': 0.05,
        'max_yaw_deg_per_second': 120.0,
        'polarity': 1,
    }


def main():
    rng = np.random.default_rng(1234)
    # Deliberately left-biased opponent population, resembling the failure the
    # held-out Minecraft validator exposed. The fit must center motor commands.
    opponents = np.clip(rng.normal(-0.16, 0.28, 2000), -0.95, 0.95)
    fit = solve_command_offset(opponents, deadband=0.05, polarity=1)
    assert abs(fit['mean_command_after_fit']) < 1e-12
    assert fit['command_offset'] < 0

    c0 = config(0.0)
    cf = config(fit['command_offset'])
    # A scalar command intercept must preserve ordering: a more right-biased
    # bilateral rate pattern remains more right-biased after calibration.
    left0 = command_from_rates(11.0, 30.0, c0)[0]
    right0 = command_from_rates(7.0, 38.0, c0)[0]
    leftf = command_from_rates(11.0, 30.0, cf)[0]
    rightf = command_from_rates(7.0, 38.0, cf)[0]
    assert left0 < right0
    assert leftf < rightf
    assert math.isfinite(leftf) and math.isfinite(rightf)

    # Polarity is included in the fitting target.
    fit_inv = solve_command_offset(opponents, deadband=0.05, polarity=-1)
    assert abs(fit_inv['mean_command_after_fit']) < 1e-12
    assert abs(fit_inv['command_offset'] - fit['command_offset']) < 1e-9
    # Strongly skewed samples saturate after offset subtraction. The fitting
    # objective must include the same clipping as the actual runtime transform.
    skewed = np.r_[np.full(90, .95), np.full(10, -.95)]
    for polarity in (-1, 1):
        fitted = solve_command_offset(skewed, polarity=polarity)
        actual = np.mean([polarity * _deadband(v-fitted['command_offset'], .05) for v in skewed])
        assert abs(actual) < 1e-12
    print('PASS bilateral decoder v3 closed-loop intercept numerics')
    print('fitted command_offset', fit['command_offset'])


if __name__ == '__main__':
    main()
