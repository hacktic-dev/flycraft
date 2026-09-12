"""Quick structural check for the DOOMFLY v6 RGB/R8 input adapter."""
import json
import numpy as np
from flycraft.learning import LearningFly,ROOT

c=json.loads((ROOT/'training.json').read_text(encoding='utf-8-sig'))
fly=LearningFly(c['plasticity']);b=fly.brain
print('visual model:',b.visual_report['model'])
print('mapped R8 total:',len(b.r8))
print('R8p / blue:',int((b.r8_channel==2).sum()))
print('R8y / green:',int((b.r8_channel==1).sum()))
print('unmapped known R8p/R8y:',b.visual_report['known_unmapped'])
print('corrected existing R8->aMe12 edges:',len(b.corrected_edges))
print('projection confidence median:',b.visual_report['projection_confidence_median'])
print('upstream graph hash:',b.pre_visual_weight_sha256)
print('visual-model initial hash:',fly.initial_weights)
assert len(b.r8)==811
assert np.isfinite(b.r8_uv).all() and ((b.r8_uv>=0)&(b.r8_uv<=1)).all()
assert len(b.corrected_edges)>0
print('R8 VISUAL ADAPTER STRUCTURE OK')

# Smoke-test the exact vector-valued current shape used by the v6 R8 adapter.
rgb=np.zeros((480,640,3),dtype=np.uint8)
counts,wall=b.rgb_step(rgb,1.0,learning=False)
assert counts.shape==(b.n,) and np.isfinite(b.r8_light).all()
print('R8 VECTOR STIMULATION SMOKE TEST OK')
