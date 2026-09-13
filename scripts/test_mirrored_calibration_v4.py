"""Cheap tests for mirrored-pair calibration geometry and fresh decoder construction."""
import math
import numpy as np

from flycraft.mirrored_starts import mirror_start, pair_starts, target_error


def distance(s):
    tx, _, tz = s['target']
    return math.hypot(tx + .5 - s['x'], tz + .5 - s['z'])


def main():
    starts = [
        {'x': 1.25, 'z': -2.75, 'yaw': 37.0, 'target': [4, -58, 1]},
        {'x': -3.5, 'z': 4.25, 'yaw': -121.0, 'target': [-1, -58, 0]},
    ]
    for s in starts:
        for axis in ('x', 'z'):
            m = mirror_start(s, axis)
            assert math.isclose(distance(s), distance(m), abs_tol=1e-10)
            assert abs(((target_error(s) + target_error(m) + 180) % 360) - 180) < 1e-9
            assert mirror_start(m, axis)['target'] == s['target']
            assert math.isclose(mirror_start(m, axis)['x'], s['x'], abs_tol=1e-12)
            assert math.isclose(mirror_start(m, axis)['z'], s['z'], abs_tol=1e-12)

    bases = [
        {'x': i / 3 - 2.0, 'z': i / 5 - 1.0, 'yaw': -150.0 + i * 23.0,
         'target': [i - 5, -58, 3 - i]}
        for i in range(12)
    ]
    pairs = pair_starts(bases, axes=('x', 'z'))
    assert len(pairs) == 12
    assert [p['axis'] for p in pairs[:4]] == ['x', 'z', 'x', 'z']
    for p in pairs:
        assert math.isclose(distance(p['a']), distance(p['b']), abs_tol=1e-10)
        assert abs(((target_error(p['a']) + target_error(p['b']) + 180) % 360) - 180) < 1e-9
    print('PASS mirrored calibration v4 geometry')


if __name__ == '__main__':
    main()
