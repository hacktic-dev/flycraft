"""From-scratch mirrored-pair calibration for bilateral DNp20 steering.

This intentionally ignores every previously fitted decoder calibration value.
Only fixed design choices (neuron type, deadband, max yaw, polarity) are reused
from the input config.  Fresh DNp20 centers/scales are estimated from matched
mirrored Minecraft scenes using an untrained frozen brain, then a scalar command
intercept is fitted from closed-loop mirrored pairs.

No learned checkpoint, v1 grey reference, v2 natural-scene fit, or v3 command
offset is used as an input to the new calibration.
"""
import argparse, copy, json
from pathlib import Path
import numpy as np
from craftground.environment.action_space import no_op_v2

from flycraft.brain import ROOT
from flycraft.decoder import calibrate_rate_samples, solve_command_offset
from flycraft.game import make_game, initialize_game, pixels, map_action
from flycraft.learning import LearningFly
from flycraft.mirrored_starts import mirrored_pairs, target_error
from flycraft.training_game import reset_episode, observe


def fixed_decoder_choices(config):
    """Return only non-fitted decoder design choices from an arbitrary config."""
    d = config.get('decoder', {})
    neuron_type = d.get('neuron_type', 'DNp20')
    if neuron_type != 'DNp20':
        raise ValueError('Mirrored calibration currently expects DNp20')
    return {
        'neuron_type': neuron_type,
        'deadband': float(d.get('deadband', 0.05)),
        'max_yaw_deg_per_second': float(d.get('max_yaw_deg_per_second', 120.0)),
        'polarity': int(d.get('polarity', 1)),
    }


def probe_decoder(fixed):
    """Neutral decoder used only to expose raw bilateral rates without old fits."""
    return {
        'mode': 'balanced-bilateral-v3',
        **fixed,
        'left_center_hz': 0.0,
        'right_center_hz': 0.0,
        'left_scale_hz': 1.0,
        'right_scale_hz': 1.0,
        'opponent_offset': 0.0,
        'response_scale': 1.0,
        'command_offset': 0.0,
    }


def collect_rate_bank(env, game_config, config, fixed, pairs, warmup, samples):
    """Collect fresh raw DNp20 rates from stationary matched mirrored scenes."""
    fly = LearningFly(config['plasticity'], probe_decoder(fixed))
    fly.freeze()
    left, right, rows = [], [], []
    for pair in pairs:
        pair_rows = []
        for member in ('a', 'b'):
            start = pair[member]
            obs = reset_episode(env, start)
            fly.reset_episode()
            member_left, member_right = [], []
            for tick in range(warmup + samples):
                control, _ = fly.step(pixels(obs, game_config))
                # Deliberately do not use decoder output here. This stage exists
                # only to estimate fresh per-side firing-rate normalization.
                obs = env.step(no_op_v2())[0]
                if tick >= warmup:
                    member_left.append(float(control['decoder']['left_hz']))
                    member_right.append(float(control['decoder']['right_hz']))
            left.extend(member_left); right.extend(member_right)
            pair_rows.append({
                'member': member,
                'target_error_deg': float(target_error(start)),
                'left_mean_hz': float(np.mean(member_left)),
                'right_mean_hz': float(np.mean(member_right)),
            })
        rows.append({'pair': pair['pair'], 'axis': pair['axis'], 'members': pair_rows})
        print(
            f"  rate pair {pair['pair']}/{len(pairs)} | "
            f"A err {pair_rows[0]['target_error_deg']:+.1f}° | "
            f"B err {pair_rows[1]['target_error_deg']:+.1f}°",
            flush=True,
        )
    return np.asarray(left), np.asarray(right), rows


def run_closed_loop_bank(env, game_config, config, decoder, pairs, steps):
    fly = LearningFly(config['plasticity'], decoder)
    fly.freeze()
    opponents, all_turns, pair_rows = [], [], []
    for pair in pairs:
        biases = []
        for member in ('a', 'b'):
            start = pair[member]
            obs = reset_episode(env, start)
            fly.reset_episode()
            turns = []
            for _ in range(steps):
                before = observe(obs, start['target'])
                control, _ = fly.step(pixels(obs, game_config))
                opponents.append(float(control['decoder']['opponent']))
                action = map_action(control, game_config)
                action['forward'] = False
                action['attack'] = False
                obs = env.step(action)[0]
                after = observe(obs, start['target'])
                turn = (float(after['yaw']) - float(before['yaw']) + 180.0) % 360.0 - 180.0
                turns.append(turn); all_turns.append(turn)
            biases.append(float(np.mean(turns)))
        residual = (biases[0] + biases[1]) / 2.0
        pair_rows.append({
            'pair': pair['pair'], 'axis': pair['axis'],
            'a_bias': biases[0], 'b_bias': biases[1],
            'pair_residual': residual,
        })
        print(
            f"  pair {pair['pair']}/{len(pairs)} | A {biases[0]:+.3f} | "
            f"B {biases[1]:+.3f} | residual {residual:+.3f} deg/tick",
            flush=True,
        )
    return {
        'opponents': np.asarray(opponents, dtype=np.float64),
        'turns': np.asarray(all_turns, dtype=np.float64),
        'pairs': pair_rows,
        'mean_turn_bias': float(np.mean(all_turns)),
        'mean_abs_pair_residual': float(np.mean(np.abs([r['pair_residual'] for r in pair_rows]))),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True,
                   help='Training config used only for environment/plasticity and fixed decoder design choices')
    p.add_argument('--output-config', type=Path, required=True)
    p.add_argument('--pairs', type=int, default=20, help='Matched mirror pairs in the calibration bank')
    p.add_argument('--rate-warmup', type=int, default=6, help='Stationary neural/filter warmup ticks per scene')
    p.add_argument('--rate-samples', type=int, default=12, help='Stationary rate samples per scene after warmup')
    p.add_argument('--closed-loop-steps', type=int, default=100)
    p.add_argument('--passes', type=int, default=2, help='Closed-loop intercept refit passes')
    args = p.parse_args()
    if args.pairs < 6 or args.rate_warmup < 2 or args.rate_samples < 4 or args.closed_loop_steps < 20:
        raise SystemExit('Use pairs>=6, rate-warmup>=2, rate-samples>=4, closed-loop-steps>=20')
    if not 1 <= args.passes <= 4:
        raise SystemExit('passes must be in [1,4]')

    config = json.loads(args.config.read_text(encoding='utf-8-sig'))
    fixed = fixed_decoder_choices(config)
    game_config = json.loads((ROOT/'config/baseline.json').read_text(encoding='utf-8-sig'))
    game_config['seed'] = config['training']['seed']

    # A new seed namespace and a freshly constructed decoder make this calibration
    # independent of all previous calibration results.
    seed = int(config['training']['seed']) + 140311
    rng = np.random.default_rng(seed)
    pairs = mirrored_pairs(rng, config['environment'], args.pairs, axes=('x', 'z'))

    env = None
    try:
        env = make_game(game_config)
        initialize_game(env, game_config)

        print('\nStage 1/2: fresh mirrored DNp20 rate normalization (no steering)', flush=True)
        left, right, rate_rows = collect_rate_bank(
            env, game_config, config, fixed, pairs, args.rate_warmup, args.rate_samples
        )
        fit = calibrate_rate_samples(left, right, deadband=fixed['deadband'])
        decoder = {
            'mode': 'balanced-bilateral-v3',
            **fixed,
            'left_center_hz': float(fit['left_center_hz']),
            'right_center_hz': float(fit['right_center_hz']),
            'left_scale_hz': float(fit['left_scale_hz']),
            'right_scale_hz': float(fit['right_scale_hz']),
            'opponent_offset': float(fit['opponent_offset']),
            'response_scale': float(fit['response_scale']),
            'command_offset': 0.0,
        }
        print(
            f"  fresh centers L/R {decoder['left_center_hz']:.3f}/{decoder['right_center_hz']:.3f} Hz | "
            f"scales {decoder['left_scale_hz']:.3f}/{decoder['right_scale_hz']:.3f} Hz",
            flush=True,
        )

        history = []
        print('\nStage 2/2: mirrored closed-loop command centering', flush=True)
        for pass_index in range(args.passes):
            print(f"\nClosed-loop pass {pass_index+1}/{args.passes} | command_offset={decoder['command_offset']:+.6f}", flush=True)
            bank = run_closed_loop_bank(env, game_config, config, decoder, pairs, args.closed_loop_steps)
            solve = solve_command_offset(
                bank['opponents'], deadband=decoder['deadband'], polarity=decoder['polarity']
            )
            history.append({
                'pass': pass_index + 1,
                'input_command_offset': float(decoder['command_offset']),
                'mean_turn_bias_deg_per_tick': bank['mean_turn_bias'],
                'mean_abs_pair_residual_deg_per_tick': bank['mean_abs_pair_residual'],
                'pairs': bank['pairs'],
                'fitted_command_offset': float(solve['command_offset']),
                'offline_mean_command_after_fit': float(solve['mean_command_after_fit']),
                'samples': int(solve['sample_count']),
            })
            decoder['command_offset'] = float(solve['command_offset'])
            print(f"  observed mean yaw bias {bank['mean_turn_bias']:+.3f} deg/tick", flush=True)
            print(f"  mean |pair residual| {bank['mean_abs_pair_residual']:.3f} deg/tick", flush=True)
            print(f"  fitted command_offset {decoder['command_offset']:+.6f}", flush=True)
    finally:
        if env is not None:
            env.close()

    decoder['calibration'] = {
        'version': 'mirrored-pairs-from-scratch-v4',
        'from_scratch': True,
        'inherited_fitted_decoder_values': False,
        'source_config': str(args.config),
        'weights': 'fresh untrained baseline',
        'seed': seed,
        'pairs': int(args.pairs),
        'mirror_axes': ['x', 'z'],
        'rate_warmup_ticks': int(args.rate_warmup),
        'rate_samples_per_scene': int(args.rate_samples),
        'closed_loop_steps_per_member': int(args.closed_loop_steps),
        'closed_loop_passes': int(args.passes),
        'fresh_rate_fit': fit,
        'history': history,
    }

    updated = copy.deepcopy(config)
    updated['decoder'] = decoder
    args.output_config.parent.mkdir(parents=True, exist_ok=True)
    args.output_config.write_text(json.dumps(updated, indent=2) + '\n', encoding='utf-8')

    report = {
        'calibration': decoder['calibration'],
        'decoder': decoder,
        'rate_pairs': rate_rows,
    }
    out = ROOT/'artifacts/decoder-calibration/mirrored-from-scratch-v4.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print('\nWROTE', args.output_config, flush=True)
    print('REPORT', out, flush=True)
    print(f"FINAL command_offset {decoder['command_offset']:+.6f}", flush=True)
    print('Run validate_bilateral_mirrored.py on the output config before training.', flush=True)


if __name__ == '__main__':
    main()
