"""Short controlled checks; no game training or long run."""
import json,math,tempfile,importlib.util
from pathlib import Path
import numpy as np
from flycraft.learning import prepare_kernel,LearningFly,ROOT
from flycraft.training_metrics import reward_components,SustainedAttack,should_record
from flycraft.training_game import randomized_start
from flycraft import training_checkpoints as cp

prepare_kernel()
spec=importlib.util.spec_from_file_location('upstream_learning_tests',ROOT/'vendor/doomfly/tests/test_doom_learning.py')
tests=importlib.util.module_from_spec(spec);spec.loader.exec_module(tests)
out=ROOT/'artifacts/training-proof';out.mkdir(exist_ok=True)
with tempfile.TemporaryDirectory(dir=out) as temp:
    for name in ('test_no_dopamine_no_plasticity_and_wrong_compartment_unchanged','test_pairing_order_and_frozen_control','test_memory_checkpoint_reproduces_continuation','test_unchanged_integration_matches_original_kernel','test_ltd_matches_independent_per_tick_event_rule'):
        getattr(tests,name)(Path(temp));print('PASS',name,flush=True)
c=json.loads((ROOT/'training.json').read_text());rng=np.random.default_rng(41027)
starts=[randomized_start(rng,c['environment']) for _ in range(30)]
assert len({(s['x'],s['z'],s['yaw'],*s['target']) for s in starts})==30
for s in starts:
    d=math.hypot(s['target'][0]+.5-s['x'],s['target'][2]+.5-s['z']);assert 2<=d<=7
a=dict(distance=4.,angle=.5,progress=.4,on_target=False,target_present=True)
b=dict(distance=3.,angle=.3,progress=.6,on_target=True,target_present=True)
assert math.isclose(reward_components(a,b,c['reward'])[0],.529)
assert reward_components(b,b,c['reward'])[0]==-.001
assert reward_components(b,{**b,'progress':0},c['reward'])[1]['breaking_progress']==-.6
assert reward_components(b,{**b,'target_present':False,'progress':0},c['reward'])[1]['success']==20
attack=SustainedAttack(c['attack']);assert [attack.step(x) for x in [True]+[False]*7]==[True]*6+[False]*2
assert [i for i in range(1,22) if should_record(i,c['recording'])]==[1,10,20]
fly=LearningFly(c['plasticity']);brain=fly.brain
# Controlled physiological assay: KC excitation is test-only, never gameplay input.
brain.step(np.zeros(len(brain.retina)),50,learning=True,stimulation=(brain.circuit['kc'][:5],30.))
before=brain.weight[brain.circuit['edges']].copy()
fly.reinforce(20);assert fly.pending_ticks==0
fly.reinforce(-1);assert fly.pending_ticks==4
rgb=np.zeros((480,640,3),np.uint8)
applied=[]
for _ in range(4):
    _,log=fly.step(rgb,record=True);applied.append(log)
assert all(r['aversive_applied'] for r in applied)
assert sum(sum(r['DAN_spikes']) for r in applied)>0
assert np.any(before!=brain.weight[brain.circuit['edges']])
state={'step':5,'episode':2,'history':[{'reward':1.},{'reward':3.}],'rng':rng.bit_generator.state}
run=out/'checkpoint-smoke';run.mkdir(exist_ok=True)
label='test-'+str(__import__('time').time_ns());path=cp.save(run,fly,state,c,label)
fly.step(rgb);expected={k:getattr(brain,k).copy() for k in ['weight',*brain.fields]}
cp.load(path,fly);fly.step(rgb)
for k,v in expected.items():assert np.array_equal(v,getattr(brain,k)),k
del expected
cp.load(path,fly);fly.freeze()
frozen={k:getattr(brain,k).copy() for k in ('weight','eligibility','eligibility_last','modulation','modulation_last')}
fly.reinforce(-1);fly.step(rgb)
for k,v in frozen.items():assert np.array_equal(v,getattr(brain,k)),k
assert cp.load(path,fly)['state']['history']==state['history']
report={'passed':True,'model':c['plasticity']['model'],'controlled_aversion':applied,'plasticity':brain.memory(),'random_starts':starts,'checkpoint':str(path),'checks':['upstream numerical LTD','original integration','positive reward no pulse','PPL101 pulse','full-graph plastic change','frozen weights and traces','exact checkpoint continuation','reward deltas','attack hold','selective recording','random starts','history roundtrip']}
cp.write_json(out/'report.json',report);print('TRAINING CONTROLLED TESTS PASSED',flush=True)
