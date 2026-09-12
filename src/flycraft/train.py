"""Minecraft aversive training; game telemetry never reaches the sensory adapter."""
import argparse,copy,json,time
from datetime import datetime
from pathlib import Path
import numpy as np
from .brain import ROOT
from .learning import LearningFly
from .game import make_game,initialize_game,pixels,preview_pixels,map_action
from .training_game import randomized_start,reset_episode,observe
from .training_metrics import reward_components,SustainedAttack,should_record,EpisodeMotionStats
from . import training_checkpoints as cp


def average(values):return float(np.mean(values)) if len(values) else 0.

def conditional_percent(value,count):
    return '--' if not count else f'{value:.0%}'

def history_metric(rows,key,predicate=None):
    values=[]
    for row in rows:
        if key not in row or row[key] is None:continue
        if predicate is not None and not predicate(row):continue
        values.append(float(row[key]))
    return average(values) if values else None



def backfill_mean_cumulative_rewards(run,state):
    """Recover mean cumulative reward for old completed episodes from step logs.

    Older checkpoints only stored the final episode return. The per-step JSONL
    logs already contain `episode_reward`, so this reconstructs the average
    height of each episode's cumulative-reward curve without changing learning.
    """
    missing={int(r.get('episode',-1)) for r in state.get('history',[])
             if not r.get('censored',False) and 'mean_cumulative_reward' not in r}
    missing.discard(-1)
    if not missing:return 0
    totals={e:[0.0,0] for e in missing}
    for path in sorted(Path(run).glob('metrics-*.jsonl')):
        try:
            with path.open('r',encoding='utf-8') as f:
                for line in f:
                    try:row=json.loads(line)
                    except json.JSONDecodeError:continue
                    e=int(row.get('episode',-1))
                    if e not in totals or 'episode_reward' not in row:continue
                    totals[e][0]+=float(row['episode_reward']);totals[e][1]+=1
        except OSError:
            continue
    filled=0
    for r in state.get('history',[]):
        e=int(r.get('episode',-1));pair=totals.get(e)
        if pair and pair[1]:
            r['mean_cumulative_reward']=pair[0]/pair[1];filled+=1
    return filled

def normalized_progress_from_history(row):
    if 'normalized_target_progress' in row:return float(row['normalized_target_progress'])
    start=float(row.get('start_target_distance',0.0) or 0.0)
    return float(row.get('target_progress',0.0))/start if start>1e-9 else 0.0


def motor_panel(control,action,attack,game_config):
    rates={}
    spikes=0
    for r in control['readouts']:
        rates[(r['type'],r['side'])]=r['rate_hz']
        if r['type']=='DNpe017':spikes+=r['spikes']
    return {'right_hz':rates.get(('DNp20','R'),0), 'left_hz':rates.get(('DNp20','L'),0),
        'forward_hz':sum(v for (typ,side),v in rates.items() if typ=='DNpe017'),
        'turn':control['turn'],'forward':control['forward'],'raw_attack':control['attack'],
        'spikes':spikes,'yaw':action['camera_yaw'],'walking':action['forward'],'attacking':action['attack'],
        'yaw_gain':game_config['yaw_gain'],'forward_threshold':game_config['forward_threshold'],
        'accumulator':attack.accumulator,'threshold':attack.config['threshold'],'decay':attack.config['decay'],
        'hold_left':attack.remaining,'hold_steps':attack.config['hold_steps']}


def panel(state,limit,episode_reward,metrics,parts,config):
    completed=[r for r in state['history'] if not r.get('censored',False)]
    rewards=[r['reward'] for r in completed]
    mean_cumulative_rows=[r for r in completed if 'mean_cumulative_reward' in r]
    mean_cumulative_rewards=[float(r['mean_cumulative_reward']) for r in mean_cumulative_rows]
    mean_cumulative_episodes=[int(r.get('episode',i+1)) for i,r in enumerate(mean_cumulative_rows)]
    window=config['dashboard']['reward_rolling_average_episodes']
    recent=completed[-window:]
    progress_values=[normalized_progress_from_history(r) for r in recent]
    behavior_rolling={
        'normalized_target_progress':average(progress_values),
        'aim_within_30_fraction':history_metric(recent,'aim_within_30_fraction'),
        'forward_when_ahead_fraction':history_metric(recent,'forward_when_ahead_fraction',lambda r:int(r.get('ahead_ticks',0))>0),
        'forward_when_not_ahead_fraction':history_metric(recent,'forward_when_not_ahead_fraction',lambda r:int(r.get('not_ahead_ticks',0))>0),
        'attack_when_ahead_fraction':history_metric(recent,'attack_when_ahead_fraction',lambda r:int(r.get('ahead_ticks',0))>0),
        'attack_when_not_ahead_fraction':history_metric(recent,'attack_when_not_ahead_fraction',lambda r:int(r.get('not_ahead_ticks',0))>0),
        'target_hit_rate':history_metric(recent,'target_hit_rate'),
        'max_break_progress':history_metric(recent,'max_break_progress'),
        'success_rate':average([float(bool(r.get('success',False))) for r in recent]),
        'conditioned_episodes':sum(1 for r in recent if 'aim_within_30_fraction' in r),
    }
    return {'step':state['step'],'total':limit,'episode':state['episode']+1,'reward':episode_reward,
        'latest':rewards[-1] if rewards else 0.,'rolling':average(rewards[-window:]),
        'mean':average(rewards),'best_eval':state['best_eval'],'distance':metrics['distance'],
        'angle':metrics['angle'],'progress':metrics['progress'],'parts':parts,'history':rewards,
        'mean_cumulative_history':mean_cumulative_rewards,
        'mean_cumulative_episodes':mean_cumulative_episodes,
        'mean_cumulative_latest':mean_cumulative_rewards[-1] if mean_cumulative_rewards else 0.,
        'mean_cumulative_rolling':average(mean_cumulative_rewards[-window:]),
        'mean_cumulative_mean':average(mean_cumulative_rewards),
        'behavior_rolling':behavior_rolling,'rolling_window':window}


def evaluation(env,checkpoint,config,game_config,count,record=False,output=None,preview=True,activity_mode='spikes',voltage_smoothing_ms=80.0):
    # A separate instance restores every neural trace. The training instance and
    # checkpoint are never stepped or written by evaluation.
    fly=LearningFly(config['plasticity']);data=cp.load(checkpoint,fly);fly.freeze()
    backfill_mean_cumulative_rewards(Path(data['run']),data['state'])
    original=fly.brain.weight.copy();trace_names=('eligibility','eligibility_last','modulation','modulation_last')
    traces={k:getattr(fly.brain,k).copy() for k in trace_names}
    results=[];rng=np.random.default_rng(config['training']['seed']+1000000000)
    for e in range(count):
        # Identical learned initial neural state for each independent held-out start.
        cp.load(checkpoint,fly);fly.freeze()
        start=randomized_start(rng,config['environment']);obs=reset_episode(env,start)
        attack=SustainedAttack(config['attack']);motion=EpisodeMotionStats();reward=0.;cumulative_reward_sum=0.;hits=0;distances=[];rec=None
        if record:
            from .training_recording import EpisodeRecording
            rec=EpisodeRecording(Path(output)/f'episode-{e+1:06d}',fly,config['recording'],game_config,observe(obs,start['target']),data['state']['history'],preview=preview,activity_mode=activity_mode,voltage_smoothing_ms=voltage_smoothing_ms)
        try:
            for tick in range(config['training']['max_steps_per_episode']):
                before=observe(obs,start['target']);preview_rgb=preview_pixels(obs);rgb=pixels(obs,game_config)
                control,neural=fly.step(rgb,record=bool(rec));action=map_action(control,game_config)
                action['attack']=attack.step(control['attack'])
                obs=env.step(action)[0];after=observe(obs,start['target'])
                behavior=motion.update(before,after,action)
                value,parts=reward_components(before,after,config['reward']);reward+=value;cumulative_reward_sum+=reward;hits+=after['on_target'];distances.append(after['distance'])
                if rec:
                    p=panel(data['state'],data['state']['step'],reward,after,parts,config);p.update(episode_step=tick+1, motor=motor_panel(control,action,attack,game_config), behavior=behavior);fly.training=p
                    rec.write(fly,rgb,preview_rgb,control,action,before,after,tick,neural,p)
                if not after['target_present']:break
            result={'episode':e+1,'start':start,'reward':reward,'success':not after['target_present'],
                'mean_distance':average(distances),'final_distance':after['distance'],'angular_error':after['angle'],
                'target_hit_rate':hits/(tick+1),'target_break_rate':float(not after['target_present']),'steps':tick+1,'seconds':(tick+1)/20,
                'mean_cumulative_reward':cumulative_reward_sum/max(1,tick+1),
                **motion.summary()}
            results.append(result)
            print(f'Eval {e+1}/{count} | Prog {result["normalized_target_progress"]:+.0%} | Aim<30 {result["aim_within_30_fraction"]:.0%} | W<30 {conditional_percent(result["forward_when_ahead_fraction"],result["ahead_ticks"])} / other {conditional_percent(result["forward_when_not_ahead_fraction"],result["not_ahead_ticks"])} | Atk<30 {conditional_percent(result["attack_when_ahead_fraction"],result["ahead_ticks"])} / other {conditional_percent(result["attack_when_not_ahead_fraction"],result["not_ahead_ticks"])} | Hit {result["target_hit_rate"]:.0%} | Break {result["max_break_progress"]:.0%} | {"SUCCESS" if result["success"] else "timeout"}',flush=True)
        finally:
            if rec:rec.close(results[-1] if len(results)>e else {'interrupted':True})
    assert np.array_equal(original,fly.brain.weight),'Evaluation changed frozen weights'
    for k,v in traces.items():assert np.array_equal(v,getattr(fly.brain,k)),k
    return {'checkpoint':str(checkpoint),'episodes':results,'success_rate':average([r['success'] for r in results]),
        **{k:average([r[k] for r in results]) for k in ('reward','mean_cumulative_reward','mean_distance','final_distance','angular_error','target_hit_rate','target_break_rate','seconds','path_efficiency','path_length','target_progress','normalized_target_progress','closest_target_distance','turn_bias_deg_per_tick','total_abs_turn_deg','mean_abs_turn_deg_per_tick','forward_fraction','attack_fraction','mean_aim_error_deg','aim_within_30_fraction','forward_when_ahead_fraction','forward_when_not_ahead_fraction','attack_when_ahead_fraction','attack_when_not_ahead_fraction','on_target_fraction','max_break_progress')},
        'frozen_verified':True}


def validate(c):
    t=c['training'];e=c['environment'];p=c['plasticity']
    if p['model']!='gamma1-eligibility-ltd-v1':raise ValueError('Only upstream aversive LTD is supported')
    for k in ('total_steps','max_steps_per_episode','evaluation_episodes'):
        if type(t[k]) is not int or t[k]<=0:raise ValueError(f'{k} must be a positive integer')
    for k in ('checkpoint_every_steps','evaluate_every_steps'):
        if type(t[k]) is not int or t[k]<0:raise ValueError(f'{k} must be nonnegative')
    if not 0<e['min_tree_distance']<e['max_tree_distance'] or e['player_spawn_radius']+e['max_tree_distance']>15:raise ValueError('Spawn range must fit the existing arena (radius + max distance <= 15)')
    if p['pulse_ticks']<=0 or type(p['pulse_ticks']) is not int or p['eta']<0 or p['current']<=0 or p['negative_threshold']<0:raise ValueError('Invalid plasticity parameters')
    if c['dashboard']['reward_rolling_average_episodes']<1:raise ValueError('Rolling window must be positive')
    SustainedAttack(c['attack'])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['fresh','resume','branch','evaluate','replay'],default='fresh')
    parser.add_argument('--checkpoint',default='latest');parser.add_argument('--config',type=Path)
    parser.add_argument('--steps',type=int);parser.add_argument('--checkpoint-every',type=int)
    parser.add_argument('--episodes',type=int);parser.add_argument('--no-preview',action='store_true')
    parser.add_argument('--activity-mode',choices=('spikes','voltage','combined'),default='spikes')
    parser.add_argument('--voltage-smoothing-ms',type=float,default=80.0)
    args=parser.parse_args();root=ROOT/'artifacts/training';root.mkdir(exist_ok=True)
    selected=None;old=None
    if args.mode!='fresh':
        selected=cp.resolve(root/'latest.json' if args.checkpoint=='latest' else args.checkpoint)
        old=json.loads((selected/'state.json').read_text())
    c=json.loads(args.config.read_text(encoding='utf-8-sig')) if args.config else copy.deepcopy(old['config']) if old else json.loads((ROOT/'training.json').read_text())
    if args.steps is not None:c['training']['total_steps']=args.steps
    if args.checkpoint_every is not None:c['training']['checkpoint_every_steps']=args.checkpoint_every
    if args.episodes is not None:c['training']['evaluation_episodes']=args.episodes
    validate(c);game_config=json.loads((ROOT/'config/baseline.json').read_text(encoding='utf-8-sig'))
    game_config['seed']=c['training']['seed']
    if args.mode in ('evaluate','replay'):
        out=root/(args.mode+'-'+datetime.now().strftime('%Y%m%d-%H%M%S-%f'));out.mkdir()
        env=make_game(game_config)
        try:
            initialize_game(env,game_config)
            report=evaluation(env,selected,c,game_config,c['training']['evaluation_episodes'],record=args.mode=='replay',output=out,preview=not args.no_preview,activity_mode=args.activity_mode,voltage_smoothing_ms=args.voltage_smoothing_ms)
            cp.write_json(out/'evaluation.json',report);print(f'Saved frozen {args.mode}: {out}',flush=True)
        finally:env.close()
        return
    if args.mode=='resume':
        run=Path(old['run'])
        if cp.resolve(run)!=selected:raise ValueError('Use --mode branch for an older checkpoint; resume only the latest generation to preserve history')
    else:
        run=root/datetime.now().strftime('run-%Y%m%d-%H%M%S-%f');run.mkdir()
    fly=LearningFly(c['plasticity']);rng=np.random.default_rng(c['training']['seed'])
    if not (run/'model.json').exists():
        cp.write_json(run/'model.json',{'model':c['plasticity']['model'],'validated':False,
            'kernel':fly.brain.build,'native_runtime':getattr(fly.brain,'runtime_build',fly.brain.build),'circuit':fly.brain.circuit['report'],
            'initial_weight_sha256':fly.initial_weights,
            'upstream_weight_sha256_before_visual_adapter':fly.brain.pre_visual_weight_sha256,
            'positive_reward':'metrics only',
            'sensory_input':'Minecraft RGB -> DOOMFLY v6 visual adapter: R1-R6 luminance + inferred R8p blue/R8y green; existing R8->aMe12 sign correction',
            'visual_model':fly.brain.visual_report})
    state={'step':0,'episode':0,'history':[],'best_eval':None,'next_eval':c['training']['evaluate_every_steps'],'in_episode':False,'rng':rng.bit_generator.state}
    if old:
        cp.load(selected,fly);state=copy.deepcopy(old['state']);rng.bit_generator.state=state['rng']
        backfilled=backfill_mean_cumulative_rewards(Path(old['run']),state)
        if backfilled:print(f'Reconstructed mean cumulative reward for {backfilled} completed episodes from existing metrics logs.',flush=True)
        if state['in_episode']:
            # Full learning state is recovered; the unsaved Minecraft world is
            # deliberately restarted. Keep censored reward instead of erasing it.
            state['history'].append({**state['partial_episode'],'censored':True,'success':False})
            state['episode']+=1;state['in_episode']=False
            fly.pending_ticks=0
    target=state['step']+c['training']['total_steps']
    stamp=datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    cp.write_json(run/f'config-{stamp}.json',c);cp.write_json(run/'config.json',c)
    if args.mode!='resume':
        if old:cp.write_json(run/'branch.json',{'parent_checkpoint':str(selected)})
        selected=cp.save(run,fly,state,c,'baseline')
    cp.write_json(root/'latest.json',{'checkpoint':str(selected)})
    cp.write_json(run/'reward-history.json',state['history'])
    print(f'EXPERIMENTAL AVERSIVE LTD | positive rewards: metrics only | {run}',flush=True)
    env=None;rec=None
    metrics=open(run/f'metrics-{stamp}.jsonl','w',encoding='utf-8')
    try:
        env=make_game(game_config);initialize_game(env,game_config)
        while state['step']<target:
            start=randomized_start(rng,c['environment']);state['rng']=rng.bit_generator.state
            obs=reset_episode(env,start);fly.pending_ticks=0
            attack=SustainedAttack(c['attack']);motion=EpisodeMotionStats();episode=state['episode']+1;begin=state['step'];reward=0.;cumulative_reward_sum=0.;hits=0
            state['in_episode']=True
            # The final budget window may contain several early successes: record
            # all candidates so the actual final episode is guaranteed footage.
            record=should_record(episode,c['recording'],final=target-begin<=c['training']['max_steps_per_episode'])
            if record:
                from .training_recording import EpisodeRecording
                rec=EpisodeRecording(run/f'episode-{episode:06d}-step-{begin:09d}',fly,c['recording'],game_config,observe(obs,start['target']),state['history'],preview=not args.no_preview,activity_mode=args.activity_mode,voltage_smoothing_ms=args.voltage_smoothing_ms)
            for tick in range(min(c['training']['max_steps_per_episode'],target-begin)):
                before=observe(obs,start['target']);preview_rgb=preview_pixels(obs);rgb=pixels(obs,game_config)
                control,neural=fly.step(rgb,record=bool(rec));action=map_action(control,game_config)
                action['attack']=attack.step(control['attack'])
                obs=env.step(action)[0];after=observe(obs,start['target'])
                behavior=motion.update(before,after,action)
                value,parts=reward_components(before,after,c['reward']);reward+=value;cumulative_reward_sum+=reward;hits+=after['on_target']
                fly.reinforce(value);state['step']+=1
                p=panel(state,target,reward,after,parts,c);p.update(episode_step=tick+1, motor=motor_panel(control,action,attack,game_config), behavior=behavior);fly.training=p
                mean_cumulative_reward=cumulative_reward_sum/max(1,tick+1)
                state['partial_episode']={'episode':episode,'start_step':begin,'end_step':state['step'],'reward':reward,'mean_cumulative_reward':mean_cumulative_reward,'start':start,'steps':tick+1,'run':str(run),'checkpoint':str(selected),'rolling_reward':p['rolling'],**behavior}
                state['attack']={'accumulator':attack.accumulator,'remaining':attack.remaining}
                row={'step':state['step'],'episode':episode,'reward':value,'episode_reward':reward,'mean_cumulative_reward':mean_cumulative_reward,'components':parts,'state':after,'action':action,'reinforcement_pending_ticks':fly.pending_ticks,'behavior':behavior,**neural}
                if state['step']%20==0:row['plasticity']=fly.brain.memory()
                metrics.write(json.dumps(row)+'\n')
                if rec:rec.write(fly,rgb,preview_rgb,control,action,before,after,tick,neural,p)
                if state['step']%20==0:
                    metrics.flush();best='n/a' if state['best_eval'] is None else f'{state["best_eval"]:.0%}'
                    print(f'Step {state["step"]:,} / {target:,} | Ep {episode} | MeanCumR {mean_cumulative_reward:+.3f} | Prog {behavior["normalized_target_progress"]:+.0%} | Aim<30 {behavior["aim_within_30_fraction"]:.0%} | W aim/other {conditional_percent(behavior["forward_when_ahead_fraction"],behavior["ahead_ticks"])}/{conditional_percent(behavior["forward_when_not_ahead_fraction"],behavior["not_ahead_ticks"])} | Atk aim/other {conditional_percent(behavior["attack_when_ahead_fraction"],behavior["ahead_ticks"])}/{conditional_percent(behavior["attack_when_not_ahead_fraction"],behavior["not_ahead_ticks"])} | RollMean {p["mean_cumulative_rolling"]:+.3f} | Best {best}',flush=True)
                every=c['training']['checkpoint_every_steps']
                if every and state['step']%every==0:
                    selected=cp.save(run,fly,state,c,f'step-{state["step"]:09d}')
                    cp.write_json(root/'latest.json',{'checkpoint':str(selected)})
                if not after['target_present']:break
            result={**state['partial_episode'],'success':not after['target_present'],'target_hit_rate':hits/(tick+1),'final_distance':after['distance'],'angular_error':after['angle'],'plasticity':fly.brain.memory(),'censored':False,**motion.summary()}
            state['history'].append(result);state['episode']=episode;state['in_episode']=False
            completed=[r['reward'] for r in state['history'] if not r.get('censored',False)]
            window=c['dashboard']['reward_rolling_average_episodes']
            result['rolling_reward']=average(completed[-window:])
            mean_cumulative_completed=[float(r['mean_cumulative_reward']) for r in state['history'] if not r.get('censored',False) and 'mean_cumulative_reward' in r]
            result['rolling_mean_cumulative_reward']=average(mean_cumulative_completed[-window:])
            print(f'Episode {episode} done | MeanCumR {result["mean_cumulative_reward"]:+.3f} | Prog {result["normalized_target_progress"]:+.0%} | Aim<30 {result["aim_within_30_fraction"]:.0%} | W aim/other {conditional_percent(result["forward_when_ahead_fraction"],result["ahead_ticks"])}/{conditional_percent(result["forward_when_not_ahead_fraction"],result["not_ahead_ticks"])} | Atk aim/other {conditional_percent(result["attack_when_ahead_fraction"],result["ahead_ticks"])}/{conditional_percent(result["attack_when_not_ahead_fraction"],result["not_ahead_ticks"])} | Break {result["max_break_progress"]:.0%} | {"SUCCESS" if result["success"] else "timeout"}',flush=True)
            if rec:rec.close(result);rec=None
            cp.write_json(run/'reward-history.json',state['history'])
            every=c['training']['evaluate_every_steps']
            if every and state['step']>=state['next_eval']:
                selected=cp.save(run,fly,state,c,f'evaluation-step-{state["step"]:09d}')
                report=evaluation(env,selected,c,game_config,c['training']['evaluation_episodes'],activity_mode=args.activity_mode,voltage_smoothing_ms=args.voltage_smoothing_ms)
                cp.write_json(run/f'evaluation-{state["step"]:09d}.json',report)
                score=(report['success_rate'],report['reward'])
                if state['best_eval'] is None or score>tuple(state.get('best_score',(-1,-1e99))):
                    state['best_eval']=score[0];state['best_score']=list(score)
                    cp.write_json(run/'checkpoints/best.json',{'checkpoint':selected.name})
                state['next_eval']=(state['step']//every+1)*every
    except KeyboardInterrupt:
        print('Stopping and saving full learning state...',flush=True)
    finally:
        if rec:rec.close({**state.get('partial_episode',{}),'interrupted':True})
        metrics.close()
        selected=cp.save(run,fly,state,c,f'latest-step-{state["step"]:09d}-{stamp}')
        cp.write_json(root/'latest.json',{'checkpoint':str(selected)})
        if env:env.close()
        print(f'Saved {run}',flush=True)

if __name__=='__main__':main()
