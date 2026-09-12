"""Short real-Minecraft infrastructure smoke; shares one game process to save startup."""
import json,sys
from pathlib import Path
import numpy as np
from flycraft.brain import ROOT
from flycraft.game import make_game,initialize_game
from flycraft.training_game import reset_episode,observe
from flycraft.training_metrics import SustainedAttack
from craftground.environment.action_space import no_op_v2
from flycraft import train,training_checkpoints as cp

out=ROOT/'artifacts/training-proof';out.mkdir(exist_ok=True)
c=json.loads((ROOT/'training.json').read_text());c['training'].update(total_steps=12,max_steps_per_episode=4,checkpoint_every_steps=6,evaluate_every_steps=8,evaluation_episodes=1)
c['recording']['record_every_episodes']=3
config=out/'smoke-config.json';cp.write_json(config,c)
g=json.loads((ROOT/'config/baseline.json').read_text())
env=make_game(g);report={}
try:
    initialize_game(env,g)
    start={'x':.5,'z':.5,'yaw':0.,'target':[0,-58,3]}
    obs=reset_episode(env,start);attack=SustainedAttack(c['attack']);progress=[]
    assert observe(obs,start['target'])['on_target']
    for i in range(100):
        action=no_op_v2();action['attack']=attack.step(i%4==0)
        obs=env.step(action)[0];s=observe(obs,start['target']);progress.append(s['progress'])
        if not s['target_present']:break
    assert any(0<p<1 for p in progress),progress
    assert not s['target_present'],'Sustained attack did not break actual target log'
    # Independently verify genuine partial progress clears when attack releases.
    obs=reset_episode(env,start)
    for _ in range(8):
        action=no_op_v2();action['attack']=True;obs=env.step(action)[0]
    partial=observe(obs,start['target'])['progress'];assert partial>0
    for _ in range(3):obs=env.step(no_op_v2())[0]
    assert observe(obs,start['target'])['progress']==0
    report['genuine_breaking']={'progress':progress,'released_progress':partial,'break_ticks':i+1,'target_removed':True}
    class SharedGame:
        def __getattr__(self,name):return getattr(env,name)
        def close(self):pass
    train.make_game=lambda c:SharedGame()
    train.initialize_game=lambda env,c:None
    def invoke(*args):
        sys.argv=['train','--no-preview',*args];train.main()
    invoke('--config',str(config))
    checkpoint=cp.resolve(ROOT/'artifacts/training/latest.json');run=Path(json.loads((checkpoint/'state.json').read_text())['run'])
    original_run=run
    history=json.loads((run/'reward-history.json').read_text());assert len(history)==3
    dirs=list(run.glob('episode-*'));assert len(dirs)==2,[p.name for p in dirs]
    first=next(p for p in dirs if p.name.startswith('episode-000001'))
    original_history=history.copy()
    invoke('--mode','resume','--steps','4')
    latest=cp.resolve(run);data=json.loads((latest/'state.json').read_text())
    assert data['state']['step']==16 and data['state']['episode']==4
    assert data['state']['history'][:3]==original_history
    invoke('--mode','branch','--checkpoint',str(run/'checkpoints/step-000000006'),'--config',str(config),'--steps','4')
    branch=cp.resolve(ROOT/'artifacts/training/latest.json');branchdata=json.loads((branch/'state.json').read_text())
    assert branchdata['state']['step']==10 and branchdata['run']!=str(run)
    assert branchdata['state']['history'][1]['censored']
    hash_before=cp.digest(latest/'brain.npz')
    invoke('--mode','evaluate','--checkpoint',str(latest),'--episodes','1')
    assert cp.digest(latest/'brain.npz')==hash_before
    invoke('--mode','replay','--checkpoint',str(latest),'--episodes','1')
    from flycraft.visuals import Visuals
    proof=Visuals.playback(first,preview=False,limit=12)
    assert json.loads((proof/'report.json').read_text())['frames_loaded']==12
    import imageio_ffmpeg
    videos={}
    for name in ('baseline','dashboard','neural-activity','retina'):
        stream=imageio_ffmpeg.read_frames(str(first/(name+'.mp4')));meta=next(stream);stream.close()
        assert tuple(meta['size'])==(1920,1080) and meta['fps']==60
        frames,_=imageio_ffmpeg.count_frames_and_secs(str(first/(name+'.mp4')));assert frames==12
        videos[name]=frames
    report.update(passed=True,run=str(original_run),branch=branchdata['run'],videos=videos,checks=['genuine progress and reset','sustained attack breaks log','3 randomized training episodes','selective recording','resume step/history','branch from mid-episode checkpoint','frozen evaluation immutable','checkpoint replay','existing dashboard offline playback','1080p60 video'])
finally:
    cp.write_json(out/'live-report.json',report);env.close()
print('LIVE TRAINING INFRASTRUCTURE PASSED',flush=True)
