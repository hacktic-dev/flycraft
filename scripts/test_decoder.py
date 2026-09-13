"""Fast controlled readout tests, including checkpoint and action integration."""
import copy,json,tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from flycraft.decoder import make_decoder,signature,validated_config
from flycraft.game import map_action
from flycraft import training_checkpoints as cp
from doom.engine import NeuralControls

ROOT=Path(__file__).resolve().parents[1]
c=json.loads((ROOT/'config/training-bilateral.json').read_text())['decoder']
g=json.loads((ROOT/'config/baseline.json').read_text())
readouts=[{'index':i,'id':str(i),'type':t,'side':s} for i,(t,s) in enumerate([
    ('DNp20','L'),('DNp20','R'),('DNpe017','L'),('DNpe017','R')])]

def command(left,right):
    d=make_decoder(readouts,c)
    # Fractional counts deliberately isolate rate normalization in this unit test.
    return d.decode(np.array([left*c['left_reference_hz'],right*c['right_reference_hz'],0,0])*.05,.05)

assert command(0,0)['turn']==0
assert command(1,1)['turn']==0
assert np.isclose(command(2,1)['turn'],-command(1,2)['turn'])
assert command(1,2)['turn']>0 and command(2,1)['turn']<0
assert abs(command(0,100000)['turn'])<=6
assert map_action(command(1,2),g)['camera_yaw']>0 # legacy gain is negative
for bad in (float('nan'),0,-1):
    try:make_decoder(readouts,c).decode(np.zeros(4),bad)
    except ValueError:pass
    else:raise AssertionError('Invalid interval accepted')
bad=copy.deepcopy(c);bad['left_reference_hz']=0
try:validated_config(bad)
except ValueError:pass
else:raise AssertionError('Silent reference accepted')
try:make_decoder(readouts[:1],c)
except ValueError:pass
else:raise AssertionError('Missing bilateral cell accepted')

# Default legacy mode is unchanged; new steering retains forward/attack/filter.
a=make_decoder(readouts);b=NeuralControls(readouts,mode='bci');d=make_decoder(readouts,c)
rng=np.random.default_rng(42)
for _ in range(100):
    spikes=rng.integers(0,4,4);old=a.decode(spikes,.05);ref=b.decode(spikes,.05);new=d.decode(spikes,.05)
    assert old==ref
    assert old['forward']==new['forward'] and old['attack']==new['attack']
    assert old['readouts']==new['readouts']
    assert np.array_equal(a.rates,d.rates)

class Brain:
    def checkpoint(self,path):np.savez(path,v=np.zeros(2))
    def restore(self,path):self.restored=True

def fly(config):
    return SimpleNamespace(brain=Brain(),decoder=make_decoder(readouts,config),state=lambda:{},restore_state=lambda s:None)

with tempfile.TemporaryDirectory() as tmp:
    a=fly(c);a.decoder.rates[:]=[1,2,3,4]
    path=cp.save(tmp,a,{}, {'decoder':c},'test')
    b=fly(c);cp.load(path,b);assert np.array_equal(a.decoder.rates,b.decoder.rates)
    old=fly(None)
    try:cp.load(path,old)
    except ValueError:pass
    else:raise AssertionError('Decoder change silently accepted')
    assert not hasattr(old.brain,'restored')
    cp.load(path,old,allow_decoder_change=True);assert not old.decoder.rates.any()
    # Pre-decoder-schema legacy checkpoints retain legacy behaviour.
    path=cp.save(tmp,old,{}, {},'legacy')
    data=json.loads((path/'state.json').read_text());data.pop('decoder');cp.write_json(path/'state.json',data)
    cp.load(path,old)
print('PASS: bilateral symmetry, silence, bounds, yaw polarity, legacy equivalence, action isolation and checkpoint compatibility')
