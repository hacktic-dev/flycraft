"""Exact left/right mirrored training starts for decoder calibration.

The calibration bank uses matched geometry instead of unrelated random episodes.
Each base start is reflected across a world axis, including player position,
target block center, and yaw.  The reflection preserves player-target distance
and reverses the signed target angle exactly (up to floating-point tolerance).

These helpers are calibration-only.  They do not alter normal training starts.
"""
import math



def wrap(deg):
    return (float(deg) + 180.0) % 360.0 - 180.0


def target_yaw(start):
    tx, _, tz = start['target']
    dx = tx + 0.5 - float(start['x'])
    dz = tz + 0.5 - float(start['z'])
    return math.degrees(math.atan2(-dx, dz))


def target_error(start):
    return wrap(target_yaw(start) - float(start['yaw']))


def mirror_start(start, axis='x'):
    """Reflect one Minecraft start exactly across x=0 or z=0.

    Block coordinates need the ``-cell-1`` transform because a block at integer
    coordinate ``t`` is centered at ``t + 0.5``. Reflecting that center across
    zero gives the center of block ``-t-1``.
    """
    tx, ty, tz = [int(v) for v in start['target']]
    x, z, yaw = float(start['x']), float(start['z']), float(start['yaw'])
    if axis == 'x':
        mirrored = {
            'x': -x,
            'z': z,
            'yaw': wrap(-yaw),
            'target': [-tx - 1, ty, tz],
        }
    elif axis == 'z':
        mirrored = {
            'x': x,
            'z': -z,
            'yaw': wrap(180.0 - yaw),
            'target': [tx, ty, -tz - 1],
        }
    else:
        raise ValueError("mirror axis must be 'x' or 'z'")

    # Geometry should be an exact reflection: same distance, opposite signed
    # target angle relative to the player's view.
    d0 = math.hypot(tx + 0.5 - x, tz + 0.5 - z)
    mtx, _, mtz = mirrored['target']
    d1 = math.hypot(mtx + 0.5 - mirrored['x'], mtz + 0.5 - mirrored['z'])
    if not math.isclose(d0, d1, rel_tol=0.0, abs_tol=1e-10):
        raise AssertionError('Mirrored start changed player-target distance')
    if abs(wrap(target_error(start) + target_error(mirrored))) > 1e-9:
        raise AssertionError('Mirrored start did not reverse signed target angle')
    return mirrored


def pair_starts(starts, axes=('x', 'z')):
    if not axes or any(a not in ('x', 'z') for a in axes):
        raise ValueError("axes must contain only 'x'/'z'")
    pairs = []
    for i, base in enumerate(starts):
        axis = axes[i % len(axes)]
        pairs.append({
            'pair': i + 1,
            'axis': axis,
            'a': base,
            'b': mirror_start(base, axis),
        })
    return pairs


def mirrored_pairs(rng, environment, count, axes=('x', 'z')):
    from .training_game import randomized_start
    if int(count) <= 0:
        raise ValueError('pair count must be positive')
    starts = [randomized_start(rng, environment) for _ in range(int(count))]
    return pair_starts(starts, axes)
