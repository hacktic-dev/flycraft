"""Adapter for unchanged upstream gamma1 eligibility LTD; RGB-only sensory input."""
import ctypes, hashlib, json, subprocess
from pathlib import Path
import numpy as np
import doom_learning.brain as upstream
from doom.engine import NeuralControls
from doom.game import retinal_samples
ROOT=Path(__file__).resolve().parents[2]


def prepare_kernel():
    """Windows build plus the existing read-only 60 Hz spike observation hook."""
    folder=ROOT/'.tools/learning-kernel';folder.mkdir(parents=True,exist_ok=True)
    source=upstream.SOURCE.read_text()
    assert source.count('counts[i]++;')==1
    source=source.replace('extern "C" void memory_advance(',
        'static int32_t* observed_bins=nullptr;\nextern "C" void set_observed_bins(int32_t* p){observed_bins=p;}\nextern "C" void memory_advance(')
    source=source.replace('counts[i]++;','counts[i]++;if(observed_bins)observed_bins[(((*clock)*60/10000)%3)*n+i]++;')
    digest=hashlib.sha256(source.encode()).hexdigest()
    dll=folder/'memory.dll';meta=folder/'build.json'
    record=json.loads(meta.read_text()) if meta.exists() else {}
    if not dll.exists() or record.get('observer_sha256')!=digest or record.get('binary_sha256')!=hashlib.sha256(dll.read_bytes()).hexdigest():
        cpp=folder/'kernel.cpp';cpp.write_text(source)
        compiler=next((ROOT/'.tools').glob('llvm-mingw-*/bin/clang++.exe'))
        subprocess.run([str(compiler),'-O3','-std=c++17','-shared','-static','-Wl,--export-all-symbols',str(cpp),'-o',str(dll)],check=True)
        record={'model':upstream.MODEL,'source_sha256':hashlib.sha256(upstream.SOURCE.read_bytes()).hexdigest(),'observer_sha256':digest,'binary_sha256':hashlib.sha256(dll.read_bytes()).hexdigest()}
        meta.write_text(json.dumps(record,indent=2))
    upstream.LIBRARY=dll;upstream.build=lambda:record
    return record


class LearningFly:
    def __init__(self,config):
        prepare_kernel()
        self.brain=upstream.MemoryBrain(eta=config['eta'])
        self.config=config;self.learning=True;self.pending_ticks=0;self.delivered_ticks=0
        self.manifest=json.loads((upstream.GRAPH.parent/'manifest.json').read_text())
        self.decoder=NeuralControls(self.manifest['readouts'],mode='bci')
        self.initial_weights=hashlib.sha256(self.brain.weight.tobytes()).hexdigest()
        lock=json.loads((ROOT/'model-lock.json').read_text(encoding='utf-8-sig'))
        assert self.initial_weights==lock['weight_sha256']
        self.samples=None;self.last_voltage=self.brain.v.copy();self.activity_bins=None
        self.bins=np.zeros((3,self.brain.n),np.int32)
        self.set_bins=self.brain.library.set_observed_bins
        self.set_bins.argtypes=[ctypes.c_void_p];self.set_bins.restype=None

    def step(self,rgb,record=False):
        if rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3:raise ValueError('Expected RGB only')
        self.samples=retinal_samples(rgb,self.brain.uv)
        active=self.learning and self.pending_ticks>0
        self.bins.fill(0);self.set_bins(self.bins.ctypes.data if record else None)
        traces={k:getattr(self.brain,k).copy() for k in ('eligibility','eligibility_last','modulation','modulation_last')} if not self.learning else {}
        try:
            counts,wall=self.brain.step(self.samples,50.,learning=self.learning,
                stimulation=(self.brain.circuit['dan'],self.config['current']) if active else None)
        finally:self.set_bins(None)
        for k,value in traces.items():getattr(self.brain,k)[:]=value
        self.pending_ticks=max(0,self.pending_ticks-1)
        self.delivered_ticks+=int(active)
        self.last_counts=counts;self.last_voltage=self.brain.v.copy()
        self.activity_bins=self.bins if record else None
        if record:assert np.array_equal(self.bins.sum(axis=0),counts)
        return self.decoder.decode(counts,.05), {'neural_ms':self.brain.sim_ms,'neural_compute_seconds':wall,
            'spikes_sha256':hashlib.sha256(counts.tobytes()).hexdigest(),
            'aversive_applied':bool(active),'aversive_current':self.config['current'] if active else 0.,
            'DAN_spikes':counts[self.brain.circuit['dan']].tolist(),
            'KC_spikes':int(counts[self.brain.circuit['kc']].sum())}

    def reinforce(self,reward):
        # Preserve upstream pulse scheduling: negative events extend a fixed
        # PPL101 pulse. Positive reward never writes weights or injects current.
        if self.learning and reward < -self.config['negative_threshold']:
            self.pending_ticks=max(self.pending_ticks,self.config['pulse_ticks'])

    def state(self):
        return {'pending_ticks':self.pending_ticks,'delivered_ticks':self.delivered_ticks}

    def restore_state(self,state):
        self.pending_ticks=state['pending_ticks'];self.delivered_ticks=state['delivered_ticks']

    def freeze(self):
        self.learning=False;self.pending_ticks=0

if __name__=='__main__':print(prepare_kernel())
