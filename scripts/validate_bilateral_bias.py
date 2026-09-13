"""Held-out frozen validation for residual left/right steering bias."""
import argparse, json, math
from pathlib import Path
import numpy as np

from flycraft.brain import ROOT
from flycraft.game import make_game, initialize_game, pixels, map_action
from flycraft.learning import LearningFly
from flycraft.training_game import randomized_start, reset_episode, observe


def wrap(deg):
    return (float(deg) + 180.0) % 360.0 - 180.0


def desired_yaw(state):
    tx, _, tz = state['target']
    dx = tx + 0.5 - state['x']
    dz = tz + 0.5 - state['z']
    return math.degrees(math.atan2(-dx, dz))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--episodes', type=int, default=8)
    p.add_argument('--steps', type=int, default=100, help='Yaw-only steps per episode')
    p.add_argument('--max-abs-bias', type=float, default=0.25, help='Fail above this |deg/tick|')
    args = p.parse_args()
    if args.episodes <= 0 or args.steps <= 0 or args.max_abs_bias < 0:
        raise SystemExit('Invalid validation arguments')

    config = json.loads(args.config.read_text(encoding='utf-8-sig'))
    d = config.get('decoder', {})
    if d.get('mode') != 'balanced-bilateral-v3' or d.get('neuron_type') != 'DNp20':
        raise SystemExit('Validation expects balanced-bilateral-v3 DNp20 config')

    game_config = json.loads((ROOT/'config/baseline.json').read_text(encoding='utf-8-sig'))
    game_config['seed'] = config['training']['seed']
    # Deliberately different from the calibration seed: this is a held-out bank.
    rng = np.random.default_rng(int(config['training']['seed']) + 55109)
    starts = [randomized_start(rng, config['environment']) for _ in range(args.episodes)]
    fly = LearningFly(config['plasticity'], d)
    fly.freeze()

    all_turns, left_turns, right_turns = [], [], []
    correct_left = correct_right = left_n = right_n = 0
    episode_biases = []
    env = None
    try:
        env = make_game(game_config)
        initialize_game(env, game_config)
        for e, start in enumerate(starts):
            obs = reset_episode(env, start)
            fly.reset_episode()
            episode_turns = []
            for _ in range(args.steps):
                before = observe(obs, start['target'])
                control, _ = fly.step(pixels(obs, game_config))
                action = map_action(control, game_config)
                action['forward'] = False
                action['attack'] = False
                target_error = wrap(desired_yaw(before) - before['yaw'])
                obs = env.step(action)[0]
                after = observe(obs, start['target'])
                actual_turn = wrap(after['yaw'] - before['yaw'])
                all_turns.append(actual_turn)
                episode_turns.append(actual_turn)
                if target_error < -1.0:
                    left_n += 1; left_turns.append(actual_turn); correct_left += int(actual_turn < 0)
                elif target_error > 1.0:
                    right_n += 1; right_turns.append(actual_turn); correct_right += int(actual_turn > 0)
            ep_bias = float(np.mean(episode_turns))
            episode_biases.append(ep_bias)
            print(f'Episode {e+1}/{args.episodes} frozen yaw bias {ep_bias:+.3f} deg/tick', flush=True)
    finally:
        if env is not None:
            env.close()

    bias = float(np.mean(all_turns)) if all_turns else 0.0
    report = {
        'episodes': args.episodes,
        'steps_per_episode': args.steps,
        'held_out_seed': int(config['training']['seed']) + 55109,
        'command_offset': float(d.get('command_offset', 0.0)),
        'mean_signed_turn_deg_per_tick': bias,
        'mean_abs_turn_deg_per_tick': float(np.mean(np.abs(all_turns))) if all_turns else 0.0,
        'mean_abs_episode_bias_deg_per_tick': float(np.mean(np.abs(episode_biases))) if episode_biases else 0.0,
        'episode_biases_deg_per_tick': episode_biases,
        'target_left': {
            'ticks': left_n,
            'mean_turn': float(np.mean(left_turns)) if left_turns else 0.0,
            'correct_direction_fraction': correct_left/left_n if left_n else None,
        },
        'target_right': {
            'ticks': right_n,
            'mean_turn': float(np.mean(right_turns)) if right_turns else 0.0,
            'correct_direction_fraction': correct_right/right_n if right_n else None,
        },
        'threshold_abs_deg_per_tick': float(args.max_abs_bias),
        'passed_bias_gate': abs(bias) <= float(args.max_abs_bias),
    }
    out = ROOT/'artifacts/decoder-calibration/frozen-bias-validation-v3.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)
    print('REPORT', out, flush=True)
    if not report['passed_bias_gate']:
        raise SystemExit(f'BIAS GATE FAILED: {bias:+.3f} deg/tick; do not train yet')
    print('BIAS GATE PASSED', flush=True)


if __name__ == '__main__':
    main()
