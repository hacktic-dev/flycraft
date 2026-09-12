"""Validate synchronized videos and lossless per-neuron exported spike counts."""
import gzip, hashlib, json, sys
from pathlib import Path
import numpy as np
import imageio_ffmpeg

folder=Path(sys.argv[1])
rows=[json.loads(line) for f in sorted(folder.glob('steps-*.jsonl.gz')) for line in gzip.open(f,'rt')]
layout=np.load(folder/'neuron-layout.npz')
chunks=[np.load(f) for f in sorted(folder.glob('spikes-*.npz'))]
assert len(layout['ids'])==166700
assert len(np.unique(layout['display_order']))==166700
for i,row in enumerate(rows):
    counts=np.zeros(166700,dtype=np.int32)
    for a in chunks:
        select=a['frame']==i
        counts[a['neuron_index'][select]]=a['count'][select]
    assert hashlib.sha256(counts.tobytes()).hexdigest()==row['spikes_sha256'],i
videos={}
for name in ('baseline.mp4','retina.mp4','neural-activity.mp4','dashboard.mp4'):
    frames,seconds=imageio_ffmpeg.count_frames_and_secs(str(folder/name))
    assert frames==len(rows),(name,frames,len(rows))
    assert abs(seconds-len(rows)/20)<.05,(name,seconds)
    videos[name]={'frames':frames,'seconds':seconds}
summary=json.loads((folder/'summary.json').read_text())
assert summary['weight_sha256_before']==summary['weight_sha256_after']
report={'passed':True,'all_per_neuron_spike_hashes_match':True,'learning':False,'videos':videos}
(folder/'visuals-verification.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
