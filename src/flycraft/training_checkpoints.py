"""Immutable generations with complete neural/temporal state and atomic pointers."""
import hashlib,json,uuid
from pathlib import Path
import numpy as np
from .decoder import signature,validated_config


def write_json(path,data):
    path=Path(path);temp=path.with_name(path.name+'.partial')
    temp.write_text(json.dumps(data,indent=2,allow_nan=False));temp.replace(path)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def resolve(path):
    path=Path(path).resolve()
    if path.is_dir() and not (path/'state.json').exists():path=path/'checkpoints/latest.json' if (path/'checkpoints').is_dir() else path/'latest.json'
    if path.suffix=='.json':path=path.parent/json.loads(path.read_text())['checkpoint']
    if not (path/'state.json').exists():raise ValueError(f'Not a checkpoint: {path}')
    return path


def save(run,fly,state,config,label,latest=True):
    directory=Path(run)/'checkpoints';directory.mkdir(exist_ok=True)
    final=directory/label
    if final.exists():raise FileExistsError(f'Checkpoint is immutable: {final}')
    stage=directory/(uuid.uuid4().hex+'.partial');stage.mkdir()
    fly.brain.checkpoint(stage/'brain.npz')
    np.save(stage/'decoder.npy',fly.decoder.rates)
    write_json(stage/'state.json',{'schema':1,'run':str(Path(run).resolve()),'config':config,'state':state,
        'reinforcement':fly.state(),'decoder':signature(fly.decoder),'sha256':{n:digest(stage/n) for n in ('brain.npz','decoder.npy')}})
    stage.rename(final)
    if latest:write_json(directory/'latest.json',{'checkpoint':label})
    return final


def load(path,fly,*,allow_decoder_change=False):
    path=resolve(path);data=json.loads((path/'state.json').read_text())
    saved=data.get('decoder',validated_config(data['config'].get('decoder')))
    changed=saved!=signature(fly.decoder)
    if changed and not allow_decoder_change:
        raise ValueError('Decoder differs from checkpoint; use an explicit Branch or Evaluate configuration, not Resume')
    for name in ('brain.npz','decoder.npy'):
        if digest(path/name)!=data['sha256'][name]:raise ValueError('Checkpoint checksum mismatch')
    fly.brain.restore(path/'brain.npz')
    rates=np.load(path/'decoder.npy',allow_pickle=False)
    if rates.shape!=fly.decoder.rates.shape:raise ValueError('Decoder shape mismatch')
    fly.decoder.rates[:]=0 if changed else rates;fly.restore_state(data['reinforcement'])
    return data
