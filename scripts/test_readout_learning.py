"""Dependency-light smoke tests for frozen-connectome supervised readout math."""
import importlib.util
from pathlib import Path
import numpy as np

MODULE = Path(__file__).resolve().parents[1] / 'src' / 'flycraft' / 'readout_learning.py'
spec = importlib.util.spec_from_file_location('readout_learning', MODULE)
r = importlib.util.module_from_spec(spec); spec.loader.exec_module(r)

class Brain:
    def __init__(self):
        self.n = 200
        self.retina = np.arange(60, dtype=np.int32)
        x,y=np.meshgrid(np.linspace(-1,1,10),np.linspace(-1,1,6))
        self.uv=np.c_[x.ravel(),y.ravel()]
        self.r8=np.arange(60,100,dtype=np.int32)
        x,y=np.meshgrid(np.linspace(-1,1,8),np.linspace(-1,1,5))
        self.r8_uv=np.c_[x.ravel(),y.ravel()]
        self.r8_channel=np.where(np.arange(40)%2,1,2).astype(np.int32)

def test_teacher():
    c={'max_yaw_deg_per_tick':6.0,'teacher_attack_distance':2.8}
    s={'x':0.,'z':0.,'yaw':0.,'distance':5.,'on_target':False,'target_present':True}
    right=r.expert_action(s,[-3,-58,4],c)
    left=r.expert_action(s,[3,-58,4],c)
    assert right['yaw']>0 and left['yaw']<0
    assert right['forward'] is False and left['forward'] is False

    # Regression: seeing the tree from outside conservative break range must not
    # make the teacher stop and attack forever. It should keep walking forward.
    aligned_far={**s,'yaw':0.,'distance':4.0,'on_target':True}
    far=r.expert_action(aligned_far,[0,-58,4],c)
    assert far['forward'] is True and far['attack'] is False, far

    aligned_close={**s,'yaw':0.,'distance':2.0,'on_target':True}
    close=r.expert_action(aligned_close,[0,-58,4],c)
    assert close['forward'] is False and close['attack'] is True, close

def test_encoder_and_learning():
    b=Brain(); cfg={'hash_dim':64,'retina_grid_width':8,'retina_grid_height':6,'hidden_units':32,
                   'replay_capacity':1200,'validation_capacity':300,'batch_size':32,
                   'warmup_train_samples':64,'train_every_samples':16,'gradient_steps':8}
    learner=r.ReadoutLearner(b,cfg); rng=np.random.default_rng(4)
    for i in range(900):
        counts=rng.poisson(.15,size=b.n).astype(np.int32)
        # Embed an easy left/right supervised signal in neural activity.
        side=1 if i%2 else -1
        counts[120 if side>0 else 121]+=8
        x=learner.encoder.encode(counts,.05)
        teacher={'yaw':side*4.0,'forward':bool(i%3),'attack':bool(i%7==0)}
        learner.observe(x,teacher)
    learner.fit(80); m=learner.evaluate_validation()
    assert learner.train.size>0 and learner.validation.size>0
    assert m is not None and m['imitation_score']>.65, m

def main():
    test_teacher(); print('PASS teacher geometry')
    test_encoder_and_learning(); print('PASS neural encoder + supervised MLP')
    print('Frozen-connectome readout smoke tests passed')

if __name__=='__main__': main()
