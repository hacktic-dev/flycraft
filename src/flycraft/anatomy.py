"""Published MaleCNS anatomical anchors. Never infer locations from graph layout."""
import hashlib
import http.client
import json
import struct
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pyarrow.feather as feather

ROOT=Path(__file__).resolve().parents[2]
CACHE=ROOT/'.tools/anatomy'
PREFIX='/flyem-male-cns/v1.0/segmentation/skeletons-malecns/skeletons-precomputed/'
LOCAL=threading.local()

def skeleton(body,whole=False):
    for attempt in range(4):
        try:
            if not getattr(LOCAL,'connection',None):
                LOCAL.connection=http.client.HTTPSConnection('storage.googleapis.com',timeout=30)
            conn=LOCAL.connection
            conn.request('GET',PREFIX+str(body),headers={} if whole else {'Range':'bytes=0-19'})
            res=conn.getresponse();data=res.read()
            if res.status==404:return None
            if res.status not in (200,206):raise RuntimeError(f'HTTP {res.status}')
            n,e=struct.unpack_from('<II',data)
            if n<1:return None
            if not whole:return list(struct.unpack_from('<fff',data,8))
            xyz=np.frombuffer(data,dtype='<f4',count=n*3,offset=8).reshape(-1,3).copy()
            edges=np.frombuffer(data,dtype='<u4',count=e*2,offset=8+n*12).reshape(-1,2).copy()
            return xyz,edges,hashlib.sha256(data).hexdigest()
        except Exception:
            if getattr(LOCAL,'connection',None):LOCAL.connection.close()
            LOCAL.connection=None
            if attempt==3:raise
            time.sleep(.5*(attempt+1))

def prepare():
    CACHE.mkdir(parents=True,exist_ok=True)
    graph=ROOT/'vendor/doomfly/outputs/doom/malecns_v1/graph.npz'
    ids=np.load(graph)['ids']
    path=ROOT/'vendor/doomfly/connectome_data/malecns_v1/annotations.feather'
    table=feather.read_table(path).to_pandas().set_index('bodyId').loc[ids]
    xyz=np.full((len(ids),3),np.nan,dtype=np.float64)
    kinds=np.full(len(ids),'missing',dtype='U24')
    for column in ('somaLocation','tosomaLocation'):
        for i,pos in enumerate(table[column]):
            if kinds[i]=='missing' and pos is not None and len(pos)==3:
                xyz[i]=np.asarray(pos)*8.0;kinds[i]=column
    cachefile=CACHE/'skeleton-anchors.jsonl'
    saved={}
    if cachefile.exists():
        for line in cachefile.read_text().splitlines():
            item=json.loads(line);saved[item['id']]=item['xyz_nm']
    missing=np.flatnonzero(kinds=='missing')
    wanted=[int(ids[i]) for i in missing if int(ids[i]) not in saved]
    print(f'Anatomy: {len(ids)-len(missing):,} annotation anchors; fetching {len(wanted):,} skeleton anchors',flush=True)
    with cachefile.open('a',encoding='utf-8') as output,ThreadPoolExecutor(32) as pool:
        futures={pool.submit(skeleton,body):body for body in wanted}
        for k,job in enumerate(as_completed(futures)):
            body=futures[job];pos=job.result();saved[body]=pos
            output.write(json.dumps({'id':body,'xyz_nm':pos})+'\n')
            if (k+1)%1000==0:output.flush();print(f'Anatomy downloads: {k+1:,}/{len(wanted):,}',flush=True)
    for i in missing:
        pos=saved.get(int(ids[i]))
        if pos is not None:xyz[i]=pos;kinds[i]='skeleton_vertex'
    np.savez_compressed(CACHE/'positions.npz',ids=ids,xyz_nm=xyz,source=kinds,superclass=table.superclass.fillna('unassigned').to_numpy(dtype='U64'))
    readouts=json.loads((graph.parent/'manifest.json').read_text())['readouts']
    skeleton_manifest={}
    for r in readouts:
        if r['type'] not in ('DNp20','DNpe017'):continue
        result=skeleton(int(r['id']),whole=True)
        if result is not None:
            points,edges,digest=result
            np.savez_compressed(CACHE/f'skeleton-{r["id"]}.npz',xyz_nm=points,edges=edges)
            skeleton_manifest[r['id']]={'sha256':digest,'type':r['type'],'side':r['side']}
    report={'source':'https://male-cns.janelia.org/download/','coordinate_space':'MaleCNS EM','units':'nm','annotations_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'annotation_units':'8 nm voxel coordinates converted to nm','skeleton_units':'precomputed format, nm','anchor_policy':'somaLocation, then tosomaLocation, then first actual skeleton vertex; no fabricated positions','counts':{str(k):int(np.sum(kinds==k)) for k in np.unique(kinds)},'skeletons':skeleton_manifest,'missing_ids':ids[kinds=='missing'].tolist(),'license':'CC-BY; FlyEM/Janelia, Cambridge, MRC LMB, Google Research'}
    (CACHE/'anatomy.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report['counts']),flush=True)

def load():
    path=CACHE/'positions.npz'
    if not path.exists():
        raise RuntimeError('Anatomical positions missing. Run .\\setup.ps1 to prepare official geometry.')
    return np.load(path)

if __name__=='__main__':prepare()
