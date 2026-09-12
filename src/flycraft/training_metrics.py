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


def should_record(episode, config, final=False):
    every = config['record_every_episodes']
    return bool((episode == 1 and config['record_first_episode']) or
                (final and config['record_final_episode']) or (every > 0 and episode % every == 0))
