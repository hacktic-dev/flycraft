"""Explicit neural-only bilateral readout; no game state or reward inputs."""
import copy,math
import numpy as np
from doom.engine import NeuralControls

LEGACY={'mode':'legacy-bci'}


def validated_config(config=None):
    c=copy.deepcopy(LEGACY if config is None else config)
    if c.get('mode')=='legacy-bci':
        if c!=LEGACY:raise ValueError('Unexpected legacy decoder options')
        return c
    if c.get('mode')!='balanced-bilateral-v1':raise ValueError('Unknown decoder mode')
    if c.get('neuron_type') not in ('DNp20','DNa02','MBON11'):raise ValueError('Unsupported steering readout')
    for k in ('left_reference_hz','right_reference_hz','max_yaw_deg_per_second','regularizer'):
        if not math.isfinite(c[k]) or c[k]<=0:raise ValueError(f'Invalid decoder {k}')
    if not math.isfinite(c['deadband']) or not 0<=c['deadband']<1:raise ValueError('Invalid decoder deadband')
    if c.get('polarity') not in (-1,1):raise ValueError('Decoder polarity must be -1 or +1')
    return c


class BilateralControls(NeuralControls):
    def __init__(self,readouts,config):
        super().__init__(readouts,mode='bci')
        self.config=validated_config(config)
        self.side_indices={side:[i for i,r in enumerate(readouts) if r['type']==self.config['neuron_type'] and r['side']==side] for side in ('L','R')}
        if any(len(ix)!=1 for ix in self.side_indices.values()):
            raise ValueError('Balanced decoder requires exactly one annotated cell per side')

    def decode(self,counts,seconds):
        if not math.isfinite(seconds) or seconds<=0:raise ValueError('Positive finite interval required')
        # Preserve the existing 100 ms filter, readout serialization, forward and
        # attack mappings. Only bilateral steering changes in this experiment.
        out=super().decode(counts,seconds)
        c=self.config
        left=float(self.rates[self.side_indices['L'][0]])
        right=float(self.rates[self.side_indices['R'][0]])
        l=left/c['left_reference_hz'];r=right/c['right_reference_hz']
        opponent=(r-l)/(r+l+c['regularizer'])
        command=math.copysign(max(0.,abs(opponent)-c['deadband'])/(1.-c['deadband']),opponent)
        out['turn']=c['polarity']*command*c['max_yaw_deg_per_second']*seconds
        out['decoder']={'mode':c['mode'],'neuron_type':c['neuron_type'],
            'turn_units':'degrees_per_tick','left_hz':left,'right_hz':right,
            'left_normalized':l,'right_normalized':r,'opponent':opponent,
            'deadband':c['deadband'],'max_yaw_deg_per_second':c['max_yaw_deg_per_second']}
        return out


def make_decoder(readouts,config=None):
    c=validated_config(config)
    if c['mode']=='legacy-bci':return NeuralControls(readouts,mode='bci')
    return BilateralControls(readouts,c)


def signature(decoder):
    return copy.deepcopy(getattr(decoder,'config',LEGACY))
