"""Audit closed-loop evidence, including neural-to-action mapping and movement."""
import gzip,hashlib,json,math,sys
from pathlib import Path
import numpy as np
from PIL import Image
from doom.game import retinal_samples
from flycraft.game import map_action
ROOT=Path(__file__).resolve().parents[1]
folder=Path(sys.argv[1]) if len(sys.argv)>1 else sorted(p for p in (ROOT/'artifacts').glob('baseline-*') if p.is_dir())[-1]
c=json.loads((folder/'config.json').read_text())
summary=json.loads((folder/'summary.json').read_text())
assert summary['complete'] and summary['steps']>=100
assert summary['learning'] is False
assert summary['weight_sha256_before']==summary['weight_sha256_after']
rows=[]
for p in sorted(folder.glob('steps-*.jsonl.gz')):
    with gzip.open(p,'rt') as f: rows.extend(json.loads(line) for line in f)
assert len(rows)==summary['steps']
for i,r in enumerate(rows):
    assert r['frame']==i and r['learning'] is False
    assert r['neural_ms']==(i+1)*50
    assert r['action']==map_action(r['controller'],c)
    assert r['controller']['attack']==any(x['spikes']>0 for x in r['controller']['readouts'] if x['type']=='DNpe017')
path=sum(math.hypot(r['after']['x']-r['before']['x'],r['after']['z']-r['before']['z']) for r in rows)
yaw=sum(abs((r['after']['yaw']-r['before']['yaw']+180)%360-180) for r in rows)
attacks=sum(r['action']['attack'] for r in rows)
assert path>1 and yaw>10 and attacks>0,(path,yaw,attacks)
assert len({r['input_rgb_sha256'] for r in rows})>10
uv=np.load(ROOT/'vendor/doomfly/outputs/doom/malecns_v1/graph.npz')['uv']
for p in folder.glob('rgb-*.png'):
    i=int(p.stem.split('-')[1]);rgb=np.asarray(Image.open(p))
    assert hashlib.sha256(rgb.tobytes()).hexdigest()==rows[i]['input_rgb_sha256']
    samples=retinal_samples(rgb,uv)
    assert hashlib.sha256(samples.tobytes()).hexdigest()==rows[i]['retinal_sha256']
report={'passed':True,'steps':len(rows),'distance_blocks':path,'total_yaw_degrees':yaw,'attack_ticks':attacks,'unchanged_weights':True,'rgb_to_retina_verified':True,'learning':False}
(folder/'verification.json').write_text(json.dumps(report,indent=2))
print('CLOSED LOOP VERIFIED:',json.dumps(report))

