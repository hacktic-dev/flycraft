"""Measure neutral reference rates only; never trains or rewrites configuration."""
import hashlib,json
from pathlib import Path
import numpy as np
from flycraft.learning import LearningFly,ROOT

c=json.loads((ROOT/'config/training-bilateral.json').read_text())
fly=LearningFly(c['plasticity']);fly.freeze();fly.reset_episode()
before=hashlib.sha256(fly.brain.weight.tobytes()).hexdigest()
rows=[]
for tick in range(80):
    control,_=fly.step(np.full((480,640,3),128,np.uint8));rows.append(control['readouts'])
report={}
for r in rows[0]:
    values=[next(q['rate_hz'] for q in row if q['index']==r['index']) for row in rows[20:]]
    report[f'{r["type"]}/{r["side"]}']={'index':r['index'],'mean_hz':float(np.mean(values)),'std_hz':float(np.std(values))}
assert hashlib.sha256(fly.brain.weight.tobytes()).hexdigest()==before
out=ROOT/'artifacts/decoder-calibration';out.mkdir(exist_ok=True)
(out/'neutral.json').write_text(json.dumps(report,indent=2))
(out/'neutral-readouts.json').write_text(json.dumps(rows))
print(json.dumps(report,indent=2))
