"""Closed-loop frozen calibration for the bilateral DNp20 steering intercept.

This replaces the earlier static-frame calibration. Minecraft advances on every
sample, the decoder's 100 ms rate filter runs normally, and the player's yaw is
actually updated. The only fitted V3 parameter is one scalar command offset.

No plasticity is enabled and target coordinates are never used to calculate the
correction. Randomized targets are present only because they define normal task
scenes. Two passes are used by default: the second pass collects a new visual
trajectory under the first corrected decoder, which reduces distribution shift.
"""
import argparse, copy, json, math
from pathlib import Path
import numpy as np

from flycraft.brain import ROOT
from flycraft.decoder import solve_command_offset
from flycraft.game import make_game, initialize_game, pixels, map_action
from flycraft.learning import LearningFly
from flycraft.training_game import randomized_start, reset_episode, observe


def wrap(deg):
    return (float(deg) + 180.0) % 360.0 - 180.0


def as_v3(decoder):
    if decoder.get('mode') not in ('balanced-bilateral-v2', 'balanced-bilateral-v3'):
        raise ValueError('Closed-loop calibration requires a balanced-bilateral-v2/v3 starting decoder')
    if decoder.get('neuron_type') != 'DNp20':
        raise ValueError('Closed-loop calibration currently expects DNp20')
    out = copy.deepcopy(decoder)
    out['mode'] = 'balanced-bilateral-v3'
    out['command_offset'] = float(out.get('command_offset', 0.0))
    return out


def run_bank(env, game_config, config, decoder, starts, steps):
    """Run frozen yaw-only episodes and return live opponent samples + yaw bias."""
    fly = LearningFly(config['plasticity'], decoder)
    fly.freeze()
    opponents, turns, episode_biases = [], [], []
    for i, start in enumerate(starts):
        obs = reset_episode(env, start)
        fly.reset_episode()
        ep_turns = []
        for _ in range(steps):
            before = observe(obs, start['target'])
            control, _ = fly.step(pixels(obs, game_config))
            opponents.append(float(control['decoder']['opponent']))
            action = map_action(control, game_config)
            action['forward'] = False
            action['attack'] = False
            obs = env.step(action)[0]
            after = observe(obs, start['target'])
            turn = wrap(after['yaw'] - before['yaw'])
            turns.append(turn)
            ep_turns.append(turn)
        episode_bias = float(np.mean(ep_turns))
        episode_biases.append(episode_bias)
        print(f'  episode {i+1}/{len(starts)} bias {episode_bias:+.3f} deg/tick', flush=True)
    return {
        'opponents': np.asarray(opponents, dtype=np.float64),
        'turns': np.asarray(turns, dtype=np.float64),
        'episode_biases': episode_biases,
        'mean_turn_bias': float(np.mean(turns)),
        'mean_abs_turn': float(np.mean(np.abs(turns))),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True,
                   help='Existing balanced-bilateral-v2/v3 training config')
    p.add_argument('--output-config', type=Path, required=True,
                   help='Write a new balanced-bilateral-v3 config')
    p.add_argument('--episodes', type=int, default=8,
                   help='Randomized closed-loop calibration episodes per pass')
    p.add_argument('--steps', type=int, default=100,
                   help='Yaw-only Minecraft steps per calibration episode')
    p.add_argument('--passes', type=int, default=2,
                   help='Fixed-point passes; pass 2 observes trajectories under the corrected decoder')
    args = p.parse_args()
    if args.episodes < 4 or args.steps < 20 or not 1 <= args.passes <= 4:
        raise SystemExit('Use episodes>=4, steps>=20, passes in [1,4]')

    config = json.loads(args.config.read_text(encoding='utf-8-sig'))
    decoder = as_v3(config.get('decoder', {}))
    game_config = json.loads((ROOT/'config/baseline.json').read_text(encoding='utf-8-sig'))
    game_config['seed'] = config['training']['seed']

    seed = int(config['training']['seed']) + 93017
    rng = np.random.default_rng(seed)
    # Reuse exactly the same randomized starts on every pass. Any change in
    # closed-loop trajectories is therefore caused by the decoder correction.
    starts = [randomized_start(rng, config['environment']) for _ in range(args.episodes)]

    history = []
    env = None
    try:
        env = make_game(game_config)
        initialize_game(env, game_config)
        for pass_index in range(args.passes):
            print(f'\nClosed-loop calibration pass {pass_index+1}/{args.passes} | command_offset={decoder["command_offset"]:+.6f}', flush=True)
            bank = run_bank(env, game_config, config, decoder, starts, args.steps)
            fit = solve_command_offset(
                bank['opponents'], deadband=float(decoder['deadband']), polarity=int(decoder['polarity'])
            )
            history.append({
                'pass': pass_index + 1,
                'input_command_offset': float(decoder['command_offset']),
                'mean_turn_bias_deg_per_tick': bank['mean_turn_bias'],
                'mean_abs_turn_deg_per_tick': bank['mean_abs_turn'],
                'episode_biases_deg_per_tick': bank['episode_biases'],
                'opponent_mean': float(np.mean(bank['opponents'])),
                'opponent_std': float(np.std(bank['opponents'])),
                'fitted_command_offset': float(fit['command_offset']),
                'offline_mean_command_after_fit': float(fit['mean_command_after_fit']),
                'samples': int(fit['sample_count']),
            })
            decoder['command_offset'] = float(fit['command_offset'])
            print(f'  observed mean yaw bias {bank["mean_turn_bias"]:+.3f} deg/tick', flush=True)
            print(f'  fitted command_offset {decoder["command_offset"]:+.6f}', flush=True)
    finally:
        if env is not None:
            env.close()

    decoder['calibration'] = {
        'source': 'frozen baseline brain in closed-loop randomized Minecraft yaw-only episodes',
        'method': 'iterative live opponent-intercept fit',
        'episodes_per_pass': int(args.episodes),
        'steps_per_episode': int(args.steps),
        'passes': int(args.passes),
        'seed': seed,
        'weights': 'fresh untrained baseline',
        'history': history,
    }
    updated = copy.deepcopy(config)
    updated['decoder'] = decoder
    args.output_config.parent.mkdir(parents=True, exist_ok=True)
    args.output_config.write_text(json.dumps(updated, indent=2) + '\n', encoding='utf-8')

    report = {'decoder': decoder, 'history': history}
    out = ROOT/'artifacts/decoder-calibration/closed-loop-v3.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print('\nWROTE', args.output_config, flush=True)
    print('REPORT', out, flush=True)
    print(f'FINAL command_offset {decoder["command_offset"]:+.6f}', flush=True)
    print('Run validate_bilateral_bias.py on the output config before training.', flush=True)


if __name__ == '__main__':
    main()
