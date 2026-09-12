"""Read-only neural activity capture helpers.

Spikes are observed at 60 Hz without changing the native simulation dynamics.
Membrane voltages are sampled at the existing 20 Hz/50 ms game-control boundary
and quantized only for recording/playback.  Quantization matches the display's
useful voltage range, so it does not alter the simulation or its controller.
"""
import ctypes
import gzip
import hashlib
import json
import subprocess
from collections import OrderedDict
from pathlib import Path

import numpy as np
import doom.native as native

ROOT=Path(__file__).resolve().parents[2]
BUILD=ROOT/'.tools/activity-kernel'

# The existing visualizer treats -52 mV as rest and maps roughly +/-7 mV around
# rest to colour.  Values outside this range are already visually clipped, so a
# uint8 snapshot over this range reproduces the display very closely while using
# 1/8 the space of float64 voltages.
VOLTAGE_MIN_MV=-59.0
VOLTAGE_MAX_MV=-45.0
VOLTAGE_HZ=20
VOLTAGE_CHUNK_FRAMES=100  # five seconds per compressed chunk


def build_observer():
    source=ROOT/'vendor/doomfly/doom/kernel.cpp'
    original=source.read_text()
    # Add one integer counter at the already-defined spike event. No floating-
    # point operation, neural step, input update or controller call is changed.
    signature='int32_t* active,uint8_t* flags,int32_t* nactive,int64_t* last)'
    assert original.count(signature)==1 and original.count('counts[i]++;')==1
    instrumented=original.replace('void neural_advance(', 'void neural_advance_observed(')
    instrumented=instrumented.replace(signature,signature[:-1]+',int32_t* bins)')
    instrumented=instrumented.replace('counts[i]++;','counts[i]++;bins[(((*clock)*60/10000)%3)*n+i]++;')
    BUILD.mkdir(parents=True,exist_ok=True)
    cpp=BUILD/'kernel_observed.cpp';dll=BUILD/'activity.dll';manifest=BUILD/'build.json'
    digest=hashlib.sha256(instrumented.encode()).hexdigest()
    if dll.exists() and manifest.exists():
        old=json.loads(manifest.read_text())
        if old['instrumented_sha256']==digest and old['binary_sha256']==hashlib.sha256(dll.read_bytes()).hexdigest():return dll
    cpp.write_text(instrumented)
    compiler=next((ROOT/'.tools').glob('llvm-mingw-*/bin/clang++.exe'))
    temp=dll.with_suffix('.partial.dll')
    command=[str(compiler),'-O3','-std=c++17','-shared','-static','-Wl,--export-all-symbols',str(cpp),'-o',str(temp)]
    subprocess.run(command,check=True);temp.replace(dll)
    manifest.write_text(json.dumps({'original_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'instrumented_sha256':digest,'binary_sha256':hashlib.sha256(dll.read_bytes()).hexdigest(),'observation':'integer spike counters only; bin=floor(native_tick*60/10000)','compile_command':command},indent=2))
    return dll


class SpikeObserver:
    def __init__(self,n):
        self.dll=ctypes.CDLL(str(build_observer()))
        self.function=self.dll.neural_advance_observed
        self.function.argtypes=native._f.argtypes+[ctypes.c_void_p]
        self.function.restype=None
        self.bins=np.zeros((3,n),dtype=np.int32)

    def step(self,brain,*args,**kwargs):
        if brain.cursor%500:raise ValueError('Observer must start on a 50 ms control boundary')
        self.bins.fill(0)
        original=native._f
        # The project runs one simulation on one thread. Retain NativeBrain.step
        # itself, including its exact 50 ms retinal low-pass and state updates.
        native._f=lambda *a:self.function(*a,self.bins.ctypes.data)
        try:result=brain.step(*args,**kwargs)
        finally:native._f=original
        if brain.cursor%500:raise ValueError('Observer requires 50 ms control updates')
        if not np.array_equal(self.bins.sum(axis=0),result[0]):raise RuntimeError('Spike observer count mismatch')
        return result


class ActivityWriter:
    """Sparse 60 Hz spike recorder: [neuron_id, spike_count] per active neuron."""
    def __init__(self,folder,ids):
        self.ids=ids
        self.path=Path(folder)/'activity-60hz.jsonl.gz'
        self.stream=gzip.open(self.path,'wt',encoding='utf-8',compresslevel=3)
        self.frame=0

    def write(self,bins):
        for counts in bins:
            indices=np.flatnonzero(counts)
            row={'frame':self.frame,'t':self.frame/60,'active':np.column_stack((self.ids[indices],counts[indices])).tolist()}
            self.stream.write(json.dumps(row,separators=(',',':'))+'\n');self.frame+=1
        if self.frame%60==0:self.stream.flush()

    def close(self):self.stream.close()


def encode_voltage(voltage):
    """Quantize a voltage vector to the exact range used by the visualizer."""
    v=np.asarray(voltage,dtype=np.float32)
    q=np.rint((np.clip(v,VOLTAGE_MIN_MV,VOLTAGE_MAX_MV)-VOLTAGE_MIN_MV)
               * (255.0/(VOLTAGE_MAX_MV-VOLTAGE_MIN_MV)))
    return q.astype(np.uint8)


def decode_voltage(encoded):
    q=np.asarray(encoded,dtype=np.float32)
    return q*((VOLTAGE_MAX_MV-VOLTAGE_MIN_MV)/255.0)+VOLTAGE_MIN_MV


class VoltageWriter:
    """Chunked 20 Hz membrane-voltage recorder for seekable offline playback.

    One uint8 value is stored per neuron per 50 ms control tick.  This is a
    visualization recording only; the native brain continues to use its full
    precision state.  Chunks avoid holding an entire long run in RAM.
    """
    def __init__(self,folder,neuron_count,chunk_frames=VOLTAGE_CHUNK_FRAMES):
        self.folder=Path(folder)/'voltage-20hz'
        self.folder.mkdir(parents=True,exist_ok=True)
        self.neuron_count=int(neuron_count)
        self.chunk_frames=int(chunk_frames)
        self.buf=np.empty((self.chunk_frames,self.neuron_count),dtype=np.uint8)
        self.ticks=np.empty(self.chunk_frames,dtype=np.int64)
        self.sim_ms=np.empty(self.chunk_frames,dtype=np.float64)
        self.used=0;self.total=0;self.chunk_index=0
        self.meta_path=self.folder/'metadata.json'
        self._write_meta(complete=False)

    def _write_meta(self,complete):
        meta={
            'hz':VOLTAGE_HZ,
            'neuron_count':self.neuron_count,
            'dtype':'uint8',
            'voltage_min_mv':VOLTAGE_MIN_MV,
            'voltage_max_mv':VOLTAGE_MAX_MV,
            'chunk_frames':self.chunk_frames,
            'total_frames':self.total,
            'complete':bool(complete),
            'sampling':'one snapshot after each 50 ms NativeBrain.step; simulation itself remains at 0.1 ms',
            'quantization':'linear uint8 over the visualizer range [-59,-45] mV; clipped outside this range',
        }
        self.meta_path.write_text(json.dumps(meta,indent=2))

    def write(self,voltage,tick,sim_ms):
        if len(voltage)!=self.neuron_count:raise ValueError('Voltage vector length changed')
        self.buf[self.used]=encode_voltage(voltage)
        self.ticks[self.used]=int(tick)
        self.sim_ms[self.used]=float(sim_ms)
        self.used+=1;self.total+=1
        if self.used==self.chunk_frames:self.flush()

    def flush(self):
        if not self.used:return
        path=self.folder/f'chunk-{self.chunk_index:05d}.npz'
        np.savez_compressed(path,
            voltage=self.buf[:self.used],
            tick=self.ticks[:self.used],
            sim_ms=self.sim_ms[:self.used])
        self.chunk_index+=1;self.used=0
        self._write_meta(complete=False)

    def close(self):
        self.flush();self._write_meta(complete=True)


class VoltageReader:
    """Lazy, small-cache reader for VoltageWriter chunks."""
    def __init__(self,folder,max_cached_chunks=3):
        folder=Path(folder)
        if folder.name!='voltage-20hz':folder=folder/'voltage-20hz'
        self.folder=folder
        self.meta=json.loads((folder/'metadata.json').read_text())
        self.hz=int(self.meta['hz'])
        self.neuron_count=int(self.meta['neuron_count'])
        self.chunk_frames=int(self.meta['chunk_frames'])
        self.total_frames=int(self.meta['total_frames'])
        self.max_cached_chunks=max(1,int(max_cached_chunks))
        self.cache=OrderedDict()

    def _chunk(self,index):
        index=int(index)
        if index in self.cache:
            value=self.cache.pop(index);self.cache[index]=value;return value
        path=self.folder/f'chunk-{index:05d}.npz'
        if not path.exists():raise IndexError(f'Missing voltage chunk {path.name}')
        with np.load(path) as a:
            value=(a['voltage'].copy(),a['tick'].copy(),a['sim_ms'].copy())
        self.cache[index]=value
        while len(self.cache)>self.max_cached_chunks:self.cache.popitem(last=False)
        return value

    def get_encoded(self,tick):
        tick=int(np.clip(tick,0,max(0,self.total_frames-1)))
        chunk=tick//self.chunk_frames;local=tick%self.chunk_frames
        voltage,ticks,_=self._chunk(chunk)
        if local>=len(voltage):raise IndexError(tick)
        # Tick numbers are recorded explicitly so corrupted/missing chunks fail
        # loudly rather than silently showing the wrong brain state.
        if int(ticks[local])!=tick:raise RuntimeError(f'Voltage timeline mismatch: wanted tick {tick}, got {int(ticks[local])}')
        return voltage[local]

    def get(self,tick):
        return decode_voltage(self.get_encoded(tick))


if __name__=='__main__':print(build_observer())
