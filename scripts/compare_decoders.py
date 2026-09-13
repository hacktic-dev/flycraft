"""Compare mappings on identical recorded spikes; not a closed-loop score test."""
import json
import numpy as np
from flycraft.learning import ROOT
from flycraft.decoder import make_decoder
from flycraft.game import map_action

c=json.loads((ROOT/'config/training-bilateral.json').read_text())
g=json.loads((ROOT/'config/baseline.json').read_text())
traces=json.loads((ROOT/'artifacts/weight-effect-2d/tick-results.json').read_text())
neutral=ROOT/'artifacts/decoder-calibration/neutral-readouts.json'
if neutral.exists():traces['neutral']=[{'readouts':r} for r in json.loads(neutral.read_text())]
result={}
for key,rows in traces.items():
    readouts=[{k:r[k] for k in ('index','id','type','side')} for r in rows[0]['readouts']]
    old=make_decoder(readouts);new=make_decoder(readouts,c['decoder'])
    counts=np.zeros(max(r['index'] for r in readouts)+1,np.int32)
    yaw={'legacy':[],'bilateral':[]}
    for tick,row in enumerate(rows):
        counts.fill(0)
        for r in row['readouts']:counts[r['index']]=r['spikes']
        a=old.decode(counts,.05);b=new.decode(counts,.05)
        assert a['readouts']==b['readouts']
        assert a['forward']==b['forward'] and a['attack']==b['attack']
        if key=='neutral' and tick<20:continue
        yaw['legacy'].append(map_action(a,g)['camera_yaw'])
        yaw['bilateral'].append(map_action(b,g)['camera_yaw'])
    result[key]={mode:{'mean_yaw_deg_per_tick':float(np.mean(v)),
        'mean_abs_yaw_deg_per_tick':float(np.mean(np.abs(v))),
        'left_fraction':float(np.mean(np.asarray(v)<0)),
        'right_fraction':float(np.mean(np.asarray(v)>0)),
        'still_fraction':float(np.mean(np.asarray(v)==0))} for mode,v in yaw.items()}
out=ROOT/'artifacts/decoder-calibration';out.mkdir(exist_ok=True)
(out/'comparison.json').write_text(json.dumps({'protocol':'Identical recorded spikes, fresh decoder filters, unchanged forward/attack. No claim about closed-loop reward.','results':result},indent=2))
print(json.dumps(result,indent=2))
