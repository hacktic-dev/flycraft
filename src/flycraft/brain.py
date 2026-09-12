"""Frozen DOOMFLY bridge. Only pixels cross this boundary."""
import hashlib,json
from pathlib import Path
import numpy as np
from doom.native import NativeBrain
from doom.engine import NeuralControls
from doom.game import retinal_samples
ROOT=Path(__file__).resolve().parents[2]
class FrozenFly:
    def __init__(self,observe_activity=False):
        folder=ROOT/'vendor/doomfly/outputs/doom/malecns_v1'
        self.manifest=json.loads((folder/'manifest.json').read_text())
        self.brain=NativeBrain(folder/'graph.npz')
        assert self.brain.n==166700, 'Full MaleCNS graph required'
        assert len(self.brain.weight)==25582938, 'All retained edges required'
        self.brain.weight.flags.writeable=False
        self.initial_weights=self.weight_hash()
        locked=json.loads((ROOT/'model-lock.json').read_text(encoding='utf-8-sig'))
        if self.initial_weights!=locked['weight_sha256']:
            raise RuntimeError('Weights differ from the verified original baseline. Rebuild the original graph.')
        self.decoder=NeuralControls(self.manifest['readouts'],mode='bci')
        self.samples=None
        self.last_voltage=self.brain.v.copy()
        self.observer=None
        if observe_activity:
            from .activity import SpikeObserver
            self.observer=SpikeObserver(self.brain.n)
    def weight_hash(self):
        return hashlib.sha256(memoryview(self.brain.weight)).hexdigest()
    def verify_frozen(self):
        if self.weight_hash()!=self.initial_weights: raise RuntimeError('Frozen weights changed')
    def step(self,rgb,duration_ms=50.0):
        if rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3: raise ValueError('Expected HWC uint8 RGB')
        self.samples=retinal_samples(rgb,self.brain.uv)
        if self.observer:
            counts,elapsed=self.observer.step(self.brain,self.samples,duration_ms,sugar=False)
            self.activity_bins=self.observer.bins.copy()
        else:
            counts,elapsed=self.brain.step(self.samples,duration_ms,sugar=False)
            self.activity_bins=None
        self.last_counts=counts
        # NativeBrain materializes all membrane voltages at each observation boundary.
        # Copy them for read-only visualization; this does not affect simulation state.
        self.last_voltage=self.brain.v.copy()
        return self.decoder.decode(counts,duration_ms/1000),{'neural_ms':self.brain.sim_ms,'neural_compute_seconds':elapsed,'total_spikes_this_step':int(counts.sum()),'retinal_sha256':hashlib.sha256(self.samples.tobytes()).hexdigest(),'spikes_sha256':hashlib.sha256(counts.tobytes()).hexdigest()}

