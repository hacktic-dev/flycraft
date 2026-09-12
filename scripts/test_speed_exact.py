"""Bit-for-bit full-network reference/optimized regression, including plasticity."""
import json,os,sys,tempfile
from pathlib import Path
import numpy as np
from flycraft.learning import ROOT


def load_reference():
    # Frozen source captured before optimisation, not a duplicate reimplementation.
    for short,file in (('_speed_reference_r8','r8_visual.py'),('_speed_reference_learning','learning.py')):
        name='flycraft.'+short
        path=Path(__file__).parent/'fixtures/speed_reference'/file
        source=path.read_text()
        if file=='learning.py':source=source.replace('from .r8_visual import','from ._speed_reference_r8 import')
        module=type(sys)(name);module.__file__=str(ROOT/'src/flycraft'/file);module.__package__='flycraft';sys.modules[name]=module
        exec(compile(source,str(path),'exec'),module.__dict__)
    return sys.modules['flycraft._speed_reference_learning'].LearningFly


def main():
    from flycraft.learning import LearningFly
    config=json.loads((ROOT/'training.json').read_text())['plasticity']
    ref=load_reference()(config);os.environ.pop('FLYCRAFT_REFERENCE_KERNEL',None);fast=LearningFly(config)
    output=ROOT/'artifacts/speed';output.mkdir(parents=True,exist_ok=True)
    trace=output/'input-trace.npz'
    frames=dict(np.load(trace))['rgb'] if trace.exists() else np.random.default_rng(1701).integers(0,256,(32,480,640,3),dtype=np.uint8)
    def compare(label):
        for name in ['weight',*ref.brain.fields]:
            a=getattr(ref.brain,name);b=getattr(fast.brain,name)
            if not np.array_equal(a.view(np.uint8),b.view(np.uint8)):raise AssertionError((label,name,float(np.max(np.abs(a.astype(float)-b.astype(float))))))
        assert ref.brain.cursor==fast.brain.cursor
        assert ref.brain.total_spikes==fast.brain.total_spikes
        assert np.array_equal(ref.decoder.rates,fast.decoder.rates)
        assert ref.state()==fast.state()
    for tick in range(32):
        if tick==8:
            # Exercise eligibility and depression even when natural KC activity
            # is sparse. This stimulation is restricted to the regression assay.
            for fly in (ref,fast):fly.brain.step(np.zeros(len(fly.brain.retina)),50.,learning=True,stimulation=(fly.brain.circuit['kc'][:20],30.))
        for fly in (ref,fast):fly.reinforce(-.1 if tick%7<4 else .1)
        rgb=frames[tick] if tick<24 else np.full_like(frames[0],255 if tick%2 else 0)
        ca,la=ref.step(rgb,record=True);cb,lb=fast.step(rgb,record=True)
        assert ca==cb and la['spikes_sha256']==lb['spikes_sha256']
        assert np.array_equal(ref.activity_bins,fast.activity_bins)
        compare(tick)
    assert ref.brain.memory()['changed_edges']>0
    for fly in (ref,fast):fly.freeze()
    for i in range(4):ref.step(frames[i]);fast.step(frames[i]);compare('frozen')
    with tempfile.TemporaryDirectory(dir=ROOT/'artifacts/speed') as tmp:
        path=Path(tmp)/'old.npz';ref.brain.checkpoint(path);fast.brain.restore(path);compare('old checkpoint -> optimized')
        path=Path(tmp)/'new.npz';fast.brain.checkpoint(path);ref.brain.restore(path);compare('optimized checkpoint -> reference')
    report={'passed':True,'ticks':36,'all_neurons':ref.brain.n,'all_edges':len(ref.brain.weight),'bitwise_equal_fields':['weight',*ref.brain.fields],'controller_and_60hz_spikes_equal':True,'plastic_edges_changed':ref.brain.memory()['changed_edges'],'checkpoint_both_directions':True,'frozen_mode':True}
    (ROOT/'artifacts/speed/exactness.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)

if __name__=='__main__':main()
