"""Independent full-connectome sanity test before Minecraft integration."""
import json
import numpy as np
from flycraft.brain import FrozenFly
fly=FrozenFly()
rows=[]
for i in range(40):
    rgb=np.full((480,640,3),180 if i<20 else 50,dtype=np.uint8)
    rgb[:,320:]=240
    control,stats=fly.step(rgb)
    rows.append({'step':i,'control':control,**stats})
fly.verify_frozen()
assert sum(r['total_spikes_this_step'] for r in rows)>0
assert any(r['control']['forward']>0 for r in rows)
from pathlib import Path
Path('artifacts/neural-proof.json').write_text(json.dumps({'learning':False,'neurons':fly.brain.n,'edges':len(fly.brain.weight),'weight_sha256':fly.initial_weights,'steps':rows},indent=2))
print('FULL NATIVE BRAIN PASS; LEARNING: OFF', rows[-1],flush=True)
