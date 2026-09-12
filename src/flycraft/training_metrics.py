"""External reward calculation and neural-only attack persistence.

Angles are radians, distance is metres, progress is Minecraft's actual 0..1
fraction. These helpers never receive a brain or choose steering actions.
"""
import math


def target_metrics(player, target):
    dx, dz = target[0] + .5 - player['x'], target[2] + .5 - player['z']
    dy = target[1] + .5 - (player['y'] + 1.62)
    distance = math.sqrt(dx*dx + dy*dy + dz*dz)
    yaw, pitch = math.radians(player['yaw']), math.radians(player['pitch'])
    look = (-math.sin(yaw)*math.cos(pitch), -math.sin(pitch), math.cos(yaw)*math.cos(pitch))
    cosine = sum(a*b for a, b in zip(look, (dx, dy, dz))) / max(distance, 1e-12)
    return {'distance': distance, 'angle': math.acos(max(-1., min(1., cosine)))}


def reward_components(before, after, config):
    """Positive approach/alignment deltas; progress reset penalized except success.

    Crosshair reward is paid on target acquisition, not every stationary tick.
    A successful break has a final progress of 1 even though Minecraft clears
    its breaking-progress field on completion. Target presence must come from
    external block telemetry, never from a missed raycast alone.
    """
    success = bool(before['target_present'] and not after['target_present'])
    def delta(value, positive, negative):
        return abs(value) * config[positive if value >= 0 else negative]
    change = (1. if success else after['progress']) - before['progress']
    parts = {
        'angle': delta(before['angle']-after['angle'], 'angle_improvement', 'angle_worsening'),
        'distance': delta(before['distance']-after['distance'], 'distance_improvement', 'distance_worsening'),
        'crosshair': config['crosshair_on_log'] if after['on_target'] and not before['on_target'] else 0.,
        'breaking_progress': delta(change, 'breaking_progress_gain', 'breaking_progress_loss'),
        'success': config['log_broken'] if success else 0.,
        'step': config['step_penalty'],
    }
    return sum(parts.values()), parts


class SustainedAttack:
    """Only neural attack events influence the accumulator; no target telemetry."""
    def __init__(self, config):
        self.config = config
        if not 0 <= config['decay'] < 1 or config['threshold'] <= 0 or config['hold_steps'] < 0:
            raise ValueError('Invalid attack configuration')
        self.accumulator = 0.
        self.remaining = 0

    def step(self, neural_attack):
        self.accumulator = self.accumulator*self.config['decay'] + float(bool(neural_attack))
        if self.accumulator >= self.config['threshold']:
            self.remaining = self.config['hold_steps']
        active = self.accumulator >= self.config['threshold'] or self.remaining > 0
        self.remaining = max(0, self.remaining-1)
        return active


class EpisodeMotionStats:
    """Read-only episode behavior metrics for detecting learned task structure.

    These values are observational only: they never feed the controller, reward,
    plasticity or sensory input. `path_efficiency` is signed useful progress
    toward the target divided by horizontal path length: +1 is a straight
    approach, ~0 is circling/no net progress, and negative values mean the fly
    has moved away overall.

    `aim_within_30_fraction` and the conditional W/attack metrics are intended
    specifically to test whether behavior becomes contingent on having the log
    roughly in front of the fly rather than occurring indiscriminately.
    """
    AHEAD_ANGLE_RAD = math.radians(30.0)

    def __init__(self):
        self.steps = 0
        self.path_length = 0.0
        self.start_target_distance = None
        self.current_target_distance = None
        self.closest_target_distance = None
        self.signed_yaw = 0.0
        self.absolute_yaw = 0.0
        self.forward_ticks = 0
        self.attack_ticks = 0
        self.aim_error_sum = 0.0
        self.ahead_ticks = 0
        self.not_ahead_ticks = 0
        self.forward_ahead_ticks = 0
        self.forward_not_ahead_ticks = 0
        self.attack_ahead_ticks = 0
        self.attack_not_ahead_ticks = 0
        self.on_target_ticks = 0
        self.max_break_progress = 0.0

    @staticmethod
    def _horizontal_target_distance(state):
        tx, _, tz = state['target']
        return math.hypot(tx + .5 - state['x'], tz + .5 - state['z'])

    def update(self, before, after, action):
        if self.start_target_distance is None:
            self.start_target_distance = self._horizontal_target_distance(before)
        self.current_target_distance = self._horizontal_target_distance(after)
        self.path_length += math.hypot(after['x'] - before['x'], after['z'] - before['z'])
        yaw = float(action.get('camera_yaw', 0.0))
        self.signed_yaw += yaw
        self.absolute_yaw += abs(yaw)

        forward = bool(action.get('forward', False))
        attacking = bool(action.get('attack', False))
        self.forward_ticks += int(forward)
        self.attack_ticks += int(attacking)

        angle = abs(float(after['angle']))
        self.aim_error_sum += angle
        ahead = angle <= self.AHEAD_ANGLE_RAD
        if ahead:
            self.ahead_ticks += 1
            self.forward_ahead_ticks += int(forward)
            self.attack_ahead_ticks += int(attacking)
        else:
            self.not_ahead_ticks += 1
            self.forward_not_ahead_ticks += int(forward)
            self.attack_not_ahead_ticks += int(attacking)

        self.on_target_ticks += int(bool(after.get('on_target', False)))
        progress = 1.0 if before.get('target_present', True) and not after.get('target_present', True) else float(after.get('progress', 0.0))
        self.max_break_progress = max(self.max_break_progress, progress)
        distance = float(after.get('distance', self.current_target_distance))
        self.closest_target_distance = distance if self.closest_target_distance is None else min(self.closest_target_distance, distance)
        self.steps += 1
        return self.summary()

    def summary(self):
        start = 0.0 if self.start_target_distance is None else self.start_target_distance
        current = start if self.current_target_distance is None else self.current_target_distance
        progress = start - current
        efficiency = progress / self.path_length if self.path_length > 1e-9 else 0.0
        # Tiny telemetry jitter can put the ratio microscopically outside [-1, 1].
        efficiency = max(-1.0, min(1.0, efficiency))
        normalized_progress = progress / start if start > 1e-9 else 0.0
        n = max(1, self.steps)
        ahead_n = max(1, self.ahead_ticks)
        other_n = max(1, self.not_ahead_ticks)
        closest = current if self.closest_target_distance is None else self.closest_target_distance
        return {
            'path_efficiency': float(efficiency),
            'path_length': float(self.path_length),
            'target_progress': float(progress),
            'normalized_target_progress': float(normalized_progress),
            'start_target_distance': float(start),
            'current_target_distance': float(current),
            'closest_target_distance': float(closest),
            'turn_bias_deg_per_tick': float(self.signed_yaw / n),
            'total_abs_turn_deg': float(self.absolute_yaw),
            'mean_abs_turn_deg_per_tick': float(self.absolute_yaw / n),
            'forward_fraction': float(self.forward_ticks / n),
            'attack_fraction': float(self.attack_ticks / n),
            'mean_aim_error_deg': float(math.degrees(self.aim_error_sum / n)),
            'aim_within_30_fraction': float(self.ahead_ticks / n),
            'ahead_ticks': int(self.ahead_ticks),
            'not_ahead_ticks': int(self.not_ahead_ticks),
            'forward_when_ahead_fraction': float(self.forward_ahead_ticks / ahead_n) if self.ahead_ticks else 0.0,
            'forward_when_not_ahead_fraction': float(self.forward_not_ahead_ticks / other_n) if self.not_ahead_ticks else 0.0,
            'attack_when_ahead_fraction': float(self.attack_ahead_ticks / ahead_n) if self.ahead_ticks else 0.0,
            'attack_when_not_ahead_fraction': float(self.attack_not_ahead_ticks / other_n) if self.not_ahead_ticks else 0.0,
            'on_target_fraction': float(self.on_target_ticks / n),
            'max_break_progress': float(self.max_break_progress),
            'behavior_steps': int(self.steps),
        }


def should_record(episode, config, final=False):
    every = config['record_every_episodes']
    return bool((episode == 1 and config['record_first_episode']) or
                (final and config['record_final_episode']) or (every > 0 and episode % every == 0))
