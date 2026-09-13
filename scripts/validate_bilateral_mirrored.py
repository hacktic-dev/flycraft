"""Held-out mirrored-pair validation for bilateral steering symmetry.

This validation intentionally does not ask an untrained fly to turn toward the
target. It asks only whether matched mirrored scenarios have opposite spontaneous
steering biases and whether their residual averages close to zero.
"""
import argparse, json
from pathlib import Path
import numpy as np

from flycraft.brain import ROOT
from flycraft.game import make_game, initialize_game, pixels, map_action
from flycraft.learning import LearningFly
from flycraft.mirrored_starts import mirrored_pairs
from flycraft.training_game import reset_episode, observe


def run_member(env, fly, game_config, start, steps):
    obs = reset_episode(env, start)
    fly.reset_episode()
    turns = []
    for _ in range(steps):
        before = observe(obs, start['target'])
        control, _ = fly.step(pixels(obs, game_config))
        action = map_action(control, game_config)
        action['forward'] = False
        action['attack'] = False
        obs = env.step(action)[0]
        after = observe(obs, start['target'])
        turn = (float(after['yaw']) - float(before['yaw']) + 180.0) % 360.0 - 180.0
        turns.append(turn)
    return float(np.mean(turns)), turns


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--pairs', type=int, default=12)
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--max-abs-bias', type=float, default=0.25)
    p.add_argument('--max-mean-abs-pair-residual', type=float, default=0.50)
    args = p.parse_args()
    if args.pairs <= 0 or args.steps <= 0 or args.max_abs_bias < 0 or args.max_mean_abs_pair_residual < 0:
        raise SystemExit('Invalid validation arguments')

    config = json.loads(args.config.read_text(encoding='utf-8-sig'))
    d = config.get('decoder', {})
    cal = d.get('calibration', {})
    if d.get('mode') != 'balanced-bilateral-v3' or d.get('neuron_type') != 'DNp20':
        raise SystemExit('Validation expects balanced-bilateral-v3 DNp20 config')
    if cal.get('version') != 'mirrored-pairs-from-scratch-v4' or not cal.get('from_scratch'):
        raise SystemExit('Config was not produced by the from-scratch mirrored calibrator')

    game_config = json.loads((ROOT/'config/baseline.json').read_text(encoding='utf-8-sig'))
    game_config['seed'] = config['training']['seed']
    seed = int(config['training']['seed']) + 240319
    rng = np.random.default_rng(seed)
    pairs = mirrored_pairs(rng, config['environment'], args.pairs, axes=('x', 'z'))

    fly = LearningFly(config['plasticity'], d)
    fly.freeze()
    all_turns, pair_rows = [], []
    env = None
    try:
        env = make_game(game_config)
        initialize_game(env, game_config)
        for pair in pairs:
            a_bias, a_turns = run_member(env, fly, game_config, pair['a'], args.steps)
            b_bias, b_turns = run_member(env, fly, game_config, pair['b'], args.steps)
            all_turns.extend(a_turns); all_turns.extend(b_turns)
            residual = (a_bias + b_bias) / 2.0
            anti_error = abs(a_bias + b_bias)
            pair_rows.append({
                'pair': pair['pair'], 'axis': pair['axis'],
                'a_bias': a_bias, 'b_bias': b_bias,
                'pair_residual': residual,
                'mirror_antisymmetry_error': anti_error,
            })
            print(
                f"Pair {pair['pair']}/{args.pairs} | A {a_bias:+.3f} | B {b_bias:+.3f} | "
                f"residual {residual:+.3f} deg/tick",
                flush=True,
            )
    finally:
        if env is not None:
            env.close()

    bias = float(np.mean(all_turns)) if all_turns else 0.0
    mean_abs_pair_residual = float(np.mean(np.abs([r['pair_residual'] for r in pair_rows])))
    mean_antisymmetry_error = float(np.mean([r['mirror_antisymmetry_error'] for r in pair_rows]))
    passed = (
        abs(bias) <= float(args.max_abs_bias)
        and mean_abs_pair_residual <= float(args.max_mean_abs_pair_residual)
    )
    report = {
        'validation': 'held-out-mirrored-pairs-v4',
        'pairs': int(args.pairs),
        'steps_per_member': int(args.steps),
        'held_out_seed': seed,
        'command_offset': float(d.get('command_offset', 0.0)),
        'mean_signed_turn_deg_per_tick': bias,
        'mean_abs_turn_deg_per_tick': float(np.mean(np.abs(all_turns))) if all_turns else 0.0,
        'mean_abs_pair_residual_deg_per_tick': mean_abs_pair_residual,
        'mean_mirror_antisymmetry_error_deg_per_tick': mean_antisymmetry_error,
        'pair_results': pair_rows,
        'threshold_abs_bias_deg_per_tick': float(args.max_abs_bias),
        'threshold_mean_abs_pair_residual_deg_per_tick': float(args.max_mean_abs_pair_residual),
        'passed_bias_gate': passed,
    }
    out = ROOT/'artifacts/decoder-calibration/mirrored-bias-validation-v4.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)
    print('REPORT', out, flush=True)
    if not passed:
        raise SystemExit(
            f"MIRROR BIAS GATE FAILED: overall {bias:+.3f} deg/tick, "
            f"mean |pair residual| {mean_abs_pair_residual:.3f}; do not train yet"
        )
    print('MIRROR BIAS GATE PASSED', flush=True)


if __name__ == '__main__':
    main()
