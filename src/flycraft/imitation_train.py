"""High-probability FlyCraft learning path: frozen connectome + supervised readout.

Training data are DAgger-style demonstrations. The privileged Minecraft target is
used ONLY by the scripted expert to create labels. The student receives only
simulated fly neural activity from ``ReadoutLearner``.
"""
from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from craftground.environment.action_space import no_op_v2

from .brain import ROOT
from .game import make_game, initialize_game, pixels, preview_pixels, map_action
from .learning import LearningFly
from .readout_learning import ReadoutLearner, expert_action
from .training_game import randomized_start, reset_episode, observe
from .training_metrics import reward_components, should_record, EpisodeMotionStats


def average(values):
    return float(np.mean(values)) if values else 0.0


def json_write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    tmp.replace(path)


def fixed_fly(config):
    """Build the current visual/full-connectome model but remove all synaptic learning."""
    plasticity = copy.deepcopy(config['plasticity'])
    plasticity.pop('action_credit', None)
    fly = LearningFly(plasticity, config.get('decoder'))
    fly.freeze()
    return fly


def assert_connectome_frozen(fly, baseline):
    if not np.array_equal(np.asarray(fly.brain.weight), baseline):
        raise RuntimeError('FIXED-CONNECTOME INVARIANT FAILED: a brain synaptic weight changed')


def resolve_checkpoint(root, value):
    if value == 'latest':
        p = root / 'latest.json'
        if not p.exists(): raise FileNotFoundError('No readout latest.json exists yet')
        value = json.loads(p.read_text(encoding='utf-8'))['checkpoint']
    p = Path(value)
    if p.suffix.lower() == '.json' and p.exists():
        data = json.loads(p.read_text(encoding='utf-8'))
        p = Path(data.get('checkpoint', data.get('model', '')))
    if p.is_dir(): p = p / 'readout-latest.npz'
    if not p.exists(): raise FileNotFoundError(p)
    return p.resolve()


def neural_motor_summary(control, action):
    rates = {}
    for r in control.get('readouts', []):
        rates[(r.get('type'), r.get('side'))] = float(r.get('rate_hz', 0.0))
    return {
        'dnp20_left_hz': rates.get(('DNp20','L'), 0.0),
        'dnp20_right_hz': rates.get(('DNp20','R'), 0.0),
        'dnpe017_left_hz': rates.get(('DNpe017','L'), 0.0),
        'dnpe017_right_hz': rates.get(('DNpe017','R'), 0.0),
        'yaw': float(action.get('camera_yaw', 0.0)),
        'walking': bool(action.get('forward', False)),
        'attacking': bool(action.get('attack', False)),
    }


def dashboard_panel(step, target_steps, episode, episode_step, reward, state, behavior,
                    learner, teacher, student, controller, base_control, action,
                    mode='TRAIN'):
    readout = learner.dashboard(teacher, student, controller)
    if mode != 'TRAIN':
        # Replay/evaluation is always autonomous: the teacher is computed only
        # as a reference label and never contributes to the Minecraft action.
        readout['phase'] = mode
        readout['student_share'] = 1.0
        readout['student_control_share'] = 1.0
        readout['teacher_control_share'] = 0.0
        readout['teacher_role'] = 'diagnostic_only'
        readout['checkpoint_samples'] = int(learner.samples)
        readout['checkpoint_gradient_updates'] = int(learner.gradient_updates)
    return {
        'learning_mode': 'frozen-connectome-readout-v1',
        'step': int(step), 'total': int(target_steps), 'episode': int(episode),
        'episode_step': int(episode_step), 'reward': float(reward),
        'distance': float(state['distance']), 'angle': float(state['angle']),
        'progress': float(state['progress']), 'behavior': behavior,
        'motor': neural_motor_summary(base_control, action),
        'readout_learning': readout,
    }


def apply_controller(base_control, game_config, command):
    action = map_action(base_control, game_config)
    action['camera_yaw'] = float(command['yaw'])
    action['forward'] = bool(command['forward'])
    action['attack'] = bool(command['attack'])
    return action


def clean_episode_world(env):
    """Clear state that otherwise leaks from one Minecraft episode to the next."""
    env.get_wrapper_attr('add_commands')([
        'clear @p',
        'kill @e[type=minecraft:item]',
    ])
    # CraftGround executes queued commands on the next environment step.
    env.step(no_op_v2())


def teacher_preflight(env, fly, config, game_config, rng, frozen_weights, learner_config):
    """Prove that the privileged expert can solve the actual closed-loop task."""
    episodes = int(learner_config['teacher_preflight_episodes'])
    required = int(learner_config['teacher_preflight_required_successes'])
    successes = 0
    print(f'TEACHER PREFLIGHT | requiring {required}/{episodes} successful log breaks', flush=True)
    for ep in range(1, episodes+1):
        start = randomized_start(rng, config['environment'])
        clean_episode_world(env)
        obs = reset_episode(env, start); fly.reset_episode()
        max_break = 0.0
        for tick in range(config['training']['max_steps_per_episode']):
            before = observe(obs, start['target'])
            base_control, _ = fly.step(pixels(obs, game_config))
            command = expert_action(before, start['target'], learner_config)
            action = apply_controller(base_control, game_config, command)
            obs = env.step(action)[0]; after = observe(obs, start['target'])
            max_break = max(max_break, float(after['progress']))
            if not after['target_present']:
                successes += 1; break
        print(f'  teacher {ep}/{episodes} | break {max_break:.0%} | {"SUCCESS" if not after["target_present"] else "timeout"}', flush=True)
        assert_connectome_frozen(fly, frozen_weights)
    if successes < required:
        raise RuntimeError(
            f'Teacher preflight failed ({successes}/{episodes}). Do not train the readout yet: '
            'fix teacher yaw/reach thresholds first.'
        )
    print(f'TEACHER PREFLIGHT PASS | {successes}/{episodes}', flush=True)


def save_training_state(run, learner, state, root):
    model = run / 'readout-latest.npz'
    learner.save(model)
    json_write(run / 'state.json', state)
    json_write(run / 'reward-history.json', state.get('history', []))
    json_write(root / 'latest.json', {'checkpoint': str(model), 'run': str(run)})
    return model


def record_success_tail(env, fly, game_config, rec, obs, start, panel, start_tick, *, ticks=20):
    """Keep successful recordings alive for ~1 second after the log disappears."""
    if rec is None:
        return obs
    # The simulation runs at 50 ms per control tick, so 20 frames is ~1 second.
    # Do not train or alter episode metrics here: this is purely a clean video tail.
    fly.training = panel
    for offset in range(int(ticks)):
        rgb = pixels(obs, game_config); preview_rgb = preview_pixels(obs)
        before = observe(obs, start['target'])
        base_control, neural = fly.step(rgb, record=True)
        action = no_op_v2()
        obs = env.step(action)[0]
        after = observe(obs, start['target'])
        rec.write(
            fly, rgb, preview_rgb, base_control, action, before, after,
            int(start_tick) + offset, neural, panel,
        )
    return obs


def autonomous_probe(env, fly, learner, config, game_config, rng, out_dir, label, *,
                     episodes=1, no_preview=False):
    """Run student-only recorded probes between training episodes for footage."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for ep in range(1, int(episodes)+1):
        start = randomized_start(rng, config['environment'])
        clean_episode_world(env)
        obs = reset_episode(env, start); fly.reset_episode(); learner.reset_episode()
        motion = EpisodeMotionStats(); reward = 0.0; hits = 0
        from .training_recording import EpisodeRecording
        rec = EpisodeRecording(
            out / f'episode-{ep:06d}', fly, config['recording'], game_config,
            observe(obs, start['target']), results, preview=not no_preview,
        )
        try:
            for tick in range(config['training']['max_steps_per_episode']):
                before = observe(obs, start['target']); rgb = pixels(obs, game_config); preview_rgb = preview_pixels(obs)
                base_control, neural = fly.step(rgb, record=True)
                feature = learner.encoder.encode(fly.last_counts, .05)
                student = learner.predict(feature)
                teacher = expert_action(before, start['target'], learner.config)
                command = {'yaw': student['yaw'], 'forward': student['forward'], 'attack': student['attack'], 'student_share': 1.0}
                action = apply_controller(base_control, game_config, command)
                obs = env.step(action)[0]; after = observe(obs, start['target'])
                behavior = motion.update(before, after, action)
                value, _parts = reward_components(before, after, config['reward'])
                reward += value; hits += int(bool(after['on_target']))
                panel = dashboard_panel(
                    tick+1, config['training']['max_steps_per_episode'], ep, tick+1, reward, after, behavior,
                    learner, teacher, student, command, base_control, action, mode='REPLAY',
                )
                fly.training = panel
                rec.write(fly, rgb, preview_rgb, base_control, action, before, after, tick, neural, panel)
                if not after['target_present']:
                    break
            result = {
                'episode': ep, 'label': str(label), 'reward': float(reward), 'start': start, 'steps': tick+1,
                'success': not bool(after['target_present']), 'target_hit_rate': hits / max(1, tick+1),
                'final_distance': float(after['distance']), 'angular_error': float(after['angle']), **motion.summary(),
            }
            if result['success']:
                obs = record_success_tail(env, fly, game_config, rec, obs, start, panel, tick+1)
            results.append(result)
            print(
                f'Probe {label} | student-only {ep}/{episodes} | ' +
                f'Prog {result["normalized_target_progress"]:+.0%} | ' +
                f'Aim<30 {result["aim_within_30_fraction"]:.0%} | ' +
                f'Break {result["max_break_progress"]:.0%} | ' +
                f'{"SUCCESS" if result["success"] else "timeout"}',
                flush=True,
            )
        finally:
            rec.close(results[-1] if len(results) >= ep else {'interrupted': True})
    keys = ('reward','target_hit_rate','final_distance','angular_error','normalized_target_progress',
            'mean_aim_error_deg','aim_within_30_fraction','max_break_progress')
    report = {
        'model': 'frozen-connectome-readout-v1',
        'probe_label': str(label),
        'episodes': results,
        'success_rate': average([float(r['success']) for r in results]),
        **{k: average([float(r[k]) for r in results]) for k in keys},
    }
    json_write(out / 'evaluation.json', report)
    return report


def episode_result(base, after, behavior, hits, steps, learner):
    return {
        **base, 'success': not bool(after['target_present']),
        'target_hit_rate': float(hits / max(1, steps)), 'final_distance': float(after['distance']),
        'angular_error': float(after['angle']), **behavior,
        'readout': learner.dashboard(), 'censored': False,
    }


def evaluate(config, game_config, checkpoint, episodes, *, record=False, no_preview=False):
    fly = fixed_fly(config); frozen_weights = fly.brain.weight.copy()
    learner = ReadoutLearner(fly.brain, config.get('readout_learning'))
    learner.load(checkpoint)
    outroot = ROOT / 'artifacts' / 'readout-training'
    out = outroot / (('replay-' if record else 'evaluate-') + datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
    out.mkdir(parents=True)
    env = make_game(game_config); initialize_game(env, game_config)
    rng = np.random.default_rng(int(config['training']['seed']) + 9000001)
    results = []
    try:
        for ep in range(1, episodes+1):
            start = randomized_start(rng, config['environment'])
            clean_episode_world(env)
            obs = reset_episode(env, start); fly.reset_episode(); learner.reset_episode()
            motion = EpisodeMotionStats(); reward = 0.0; hits = 0; rec = None
            if record:
                from .training_recording import EpisodeRecording
                rec = EpisodeRecording(out/f'episode-{ep:06d}', fly, config['recording'], game_config,
                                       observe(obs,start['target']), results, preview=not no_preview)
            try:
                for tick in range(config['training']['max_steps_per_episode']):
                    before = observe(obs, start['target']); rgb = pixels(obs, game_config); preview_rgb = preview_pixels(obs)
                    base_control, neural = fly.step(rgb, record=bool(rec))
                    feature = learner.encoder.encode(fly.last_counts, .05)
                    student = learner.predict(feature)
                    teacher = expert_action(before, start['target'], learner.config)  # diagnostic label only
                    command = {'yaw': student['yaw'], 'forward': student['forward'], 'attack': student['attack'], 'student_share': 1.0}
                    action = apply_controller(base_control, game_config, command)
                    obs = env.step(action)[0]; after = observe(obs, start['target'])
                    behavior = motion.update(before, after, action); value, parts = reward_components(before, after, config['reward'])
                    reward += value; hits += int(bool(after['on_target']))
                    p = dashboard_panel(tick+1, config['training']['max_steps_per_episode'], ep, tick+1, reward, after, behavior,
                                        learner, teacher, student, command, base_control, action,
                                        mode='REPLAY' if record else 'EVALUATION')
                    fly.training = p
                    if rec: rec.write(fly, rgb, preview_rgb, base_control, action, before, after, tick, neural, p)
                    if not after['target_present']: break
                result = {'episode':ep,'reward':float(reward),'start':start,'steps':tick+1,
                          'success':not bool(after['target_present']),'target_hit_rate':hits/(tick+1),
                          'final_distance':float(after['distance']),'angular_error':float(after['angle']),**motion.summary()}
                if rec and result['success']:
                    obs = record_success_tail(env, fly, game_config, rec, obs, start, p, tick+1)
                results.append(result)
                print(f'Eval {ep}/{episodes} | Prog {result["normalized_target_progress"]:+.0%} | '
                      f'Aim<30 {result["aim_within_30_fraction"]:.0%} | Hit {result["target_hit_rate"]:.0%} | '
                      f'Break {result["max_break_progress"]:.0%} | {"SUCCESS" if result["success"] else "timeout"}', flush=True)
            finally:
                if rec: rec.close(results[-1] if len(results)>=ep else {'interrupted':True})
            assert_connectome_frozen(fly, frozen_weights)
    finally:
        env.close()
    keys = ('reward','target_hit_rate','final_distance','angular_error','normalized_target_progress',
            'mean_aim_error_deg','aim_within_30_fraction','max_break_progress')
    report = {'model':'frozen-connectome-readout-v1','checkpoint':str(checkpoint),'episodes':results,
              'success_rate':average([float(r['success']) for r in results]),
              **{k:average([float(r[k]) for r in results]) for k in keys},
              'connectome_frozen_verified':True}
    json_write(out/'evaluation.json', report)
    print(f'Saved student-only evaluation: {out}', flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('fresh','resume','evaluate','replay'), default='fresh')
    parser.add_argument('--config', type=Path, default=ROOT/'config'/'new-arch.json')
    parser.add_argument('--checkpoint', default='latest')
    parser.add_argument('--steps', type=int)
    parser.add_argument('--episodes', type=int)
    parser.add_argument('--no-preview', action='store_true')
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding='utf-8-sig'))
    game_config = json.loads((ROOT/'config'/'baseline.json').read_text(encoding='utf-8-sig'))
    game_config['seed'] = int(config['training']['seed'])
    outroot = ROOT/'artifacts'/'readout-training'; outroot.mkdir(parents=True, exist_ok=True)

    if args.mode in ('evaluate','replay'):
        checkpoint = resolve_checkpoint(outroot, args.checkpoint)
        return evaluate(config, game_config, checkpoint,
                        int(args.episodes or config['training']['evaluation_episodes']),
                        record=args.mode=='replay', no_preview=args.no_preview)

    fly = fixed_fly(config); frozen_weights = fly.brain.weight.copy()
    learner = ReadoutLearner(fly.brain, config.get('readout_learning'))
    rng = np.random.default_rng(int(config['training']['seed']))
    probe_rng = np.random.default_rng(int(config['training']['seed']) + 12000001)
    if args.mode == 'resume':
        checkpoint = resolve_checkpoint(outroot, args.checkpoint)
        run = checkpoint.parent
        learner.load(checkpoint)
        state_path = run/'state.json'
        state = json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else {'step':0,'episode':0,'history':[]}
        # Replay buffer intentionally restarts; the learned readout itself resumes exactly.
    else:
        run = outroot/datetime.now().strftime('run-%Y%m%d-%H%M%S-%f'); run.mkdir()
        state = {'step':0,'episode':0,'history':[]}
        json_write(run/'model.json', {
            'model':'frozen-connectome-readout-v1',
            'description':'Full MaleCNS neural activity is frozen as a reservoir; only a small supervised motor readout learns. Minecraft target geometry is label-only and is never included in student features.',
            'student_inputs':'simulated neural spike counts only: full-CNS CountSketch + retinotopic R1-R6/R8 spike pools + temporal EMA',
            'teacher_inputs':'privileged target geometry used only to generate yaw/walk/attack labels',
            'connectome_synapses_trainable':False,
            'readout_config':learner.config,
        })

    add_steps = int(args.steps if args.steps is not None else config['training']['total_steps'])
    target_steps = int(state.get('step',0)) + add_steps
    json_write(run/'config.json', {**config, 'readout_learning_effective':learner.config})
    save_training_state(run, learner, state, outroot)
    print(f'READOUT LEARNING | CONNECTOME FROZEN | supervised demonstrations -> DAgger -> student | {run}', flush=True)

    env = make_game(game_config); initialize_game(env, game_config); rec = None
    try:
        if args.mode == 'fresh':
            preflight_rng = np.random.default_rng(int(config['training']['seed']) + 770001)
            teacher_preflight(env, fly, config, game_config, preflight_rng, frozen_weights, learner.config)
        while state['step'] < target_steps:
            start = randomized_start(rng, config['environment'])
            clean_episode_world(env)
            obs = reset_episode(env, start); fly.reset_episode(); learner.reset_episode()
            episode = int(state.get('episode',0))+1; begin = int(state['step'])
            reward = 0.0; cumulative_reward_sum = 0.0; hits = 0; motion = EpisodeMotionStats()
            record = bool(learner.config.get('record_every_training_episode', False)) or should_record(
                episode, config['recording'], final=target_steps-begin<=config['training']['max_steps_per_episode']
            )
            if record:
                from .training_recording import EpisodeRecording
                rec = EpisodeRecording(run/f'episode-{episode:06d}-step-{begin:09d}', fly, config['recording'], game_config,
                                       observe(obs,start['target']), state['history'], preview=not args.no_preview)
            for tick in range(min(config['training']['max_steps_per_episode'], target_steps-begin)):
                before = observe(obs, start['target']); rgb = pixels(obs, game_config); preview_rgb = preview_pixels(obs)
                base_control, neural = fly.step(rgb, record=bool(rec))
                features = learner.encoder.encode(fly.last_counts, .05)
                student = learner.predict(features)
                teacher = expert_action(before, start['target'], learner.config)
                learner.observe(features, teacher)
                command = learner.blend(teacher, student)
                action = apply_controller(base_control, game_config, command)
                obs = env.step(action)[0]; after = observe(obs, start['target'])
                behavior = motion.update(before, after, action)
                value, parts = reward_components(before, after, config['reward'])
                reward += value; cumulative_reward_sum += reward; hits += int(bool(after['on_target']))
                state['step'] += 1
                p = dashboard_panel(state['step'], target_steps, episode, tick+1, reward, after, behavior,
                                    learner, teacher, student, command, base_control, action)
                fly.training = p
                if rec: rec.write(fly, rgb, preview_rgb, base_control, action, before, after, tick, neural, p)
                if state['step'] % 20 == 0:
                    m = learner.metrics or {}
                    loss = 'n/a' if learner.train_loss is None else f'{learner.train_loss:.3f}'
                    yaw = 'n/a' if m.get('yaw_mae_deg') is None else f'{m["yaw_mae_deg"]:.2f}deg'
                    walk = 'n/a' if m.get('walk_accuracy') is None else f'{m["walk_accuracy"]:.0%}'
                    attack = 'n/a' if m.get('attack_accuracy') is None else f'{m["attack_accuracy"]:.0%}'
                    score = 'n/a' if m.get('imitation_score') is None else f'{m["imitation_score"]:.0%}'
                    student_share = learner.student_share()
                    print(
                        f'Step {state["step"]:,}/{target_steps:,} | Ep {episode} | {learner.phase()} | '
                        f'DATA train={learner.train.size:,} val={learner.validation.size:,} | '
                        f'FIT updates={learner.gradient_updates:,} loss={loss} | '
                        f'HELDOUT yawMAE={yaw} walk={walk} attack={attack} score={score} | '
                        f'CTRL student={student_share:.0%} teacher={1.0-student_share:.0%} | '
                        f'TASK prog={behavior["normalized_target_progress"]:+.0%} '
                        f'aim<30={behavior["aim_within_30_fraction"]:.0%} '
                        f'break={behavior["max_break_progress"]:.0%}',
                        flush=True,
                    )
                if state['step'] % 200 == 0: assert_connectome_frozen(fly, frozen_weights)
                if not after['target_present']: break

            learner.fit(32); learner.evaluate_validation()
            mean_cum = cumulative_reward_sum/max(1,tick+1)
            base = {'episode':episode,'start_step':begin,'end_step':state['step'],'reward':float(reward),
                    'mean_cumulative_reward':float(mean_cum),'start':start,'steps':tick+1,'run':str(run)}
            result = episode_result(base, after, motion.summary(), hits, tick+1, learner)
            if rec and result['success']:
                obs = record_success_tail(env, fly, game_config, rec, obs, start, p, tick+1)
            state['history'].append(result); state['episode']=episode
            assert_connectome_frozen(fly, frozen_weights)
            model = save_training_state(run, learner, state, outroot)
            m = learner.metrics or {}
            yaw = 'n/a' if m.get('yaw_mae_deg') is None else f'{m["yaw_mae_deg"]:.2f}deg'
            walk = 'n/a' if m.get('walk_accuracy') is None else f'{m["walk_accuracy"]:.0%}'
            attack = 'n/a' if m.get('attack_accuracy') is None else f'{m["attack_accuracy"]:.0%}'
            score = 'n/a' if m.get('imitation_score') is None else f'{m["imitation_score"]:.0%}'
            student_share = learner.student_share()
            print(
                f'Episode {episode} done | {learner.phase()} | samples={learner.samples:,} | '
                f'HELDOUT yawMAE={yaw} walk={walk} attack={attack} score={score} | '
                f'CTRL student={student_share:.0%} teacher={1.0-student_share:.0%} | '
                f'TASK prog={result["normalized_target_progress"]:+.0%} '
                f'aim<30={result["aim_within_30_fraction"]:.0%} '
                f'break={result["max_break_progress"]:.0%} | '
                f'{"SUCCESS" if result["success"] else "timeout"} | saved {model.name}',
                flush=True,
            )
            if rec: rec.close(result); rec=None

            if bool(learner.config.get('autonomous_probe_after_episode', False)):
                probe_dir = run / 'autonomous-probes' / f'after-episode-{episode:06d}-step-{state["step"]:09d}'
                probe = autonomous_probe(
                    env, fly, learner, config, game_config, probe_rng, probe_dir,
                    label=f'after training episode {episode}',
                    episodes=int(learner.config.get('autonomous_probe_episodes', 1)),
                    no_preview=args.no_preview,
                )
                print(
                    f'Auto-probe after episode {episode} | success {probe["success_rate"]:.0%} | '
                    f'mean break {probe["max_break_progress"]:.0%} | saved {probe_dir}',
                    flush=True,
                )
                assert_connectome_frozen(fly, frozen_weights)
    except KeyboardInterrupt:
        print('Stopping and saving learned readout...', flush=True)
    finally:
        if rec: rec.close({'interrupted':True})
        assert_connectome_frozen(fly, frozen_weights)
        save_training_state(run, learner, state, outroot)
        env.close()
        print(f'Saved {run}', flush=True)


if __name__ == '__main__':
    main()
