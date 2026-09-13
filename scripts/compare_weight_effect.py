"""Two paired open-loop weight interventions; never writes training checkpoints."""
import hashlib,json,time
from pathlib import Path
import numpy as np
from flycraft.learning import LearningFly,ROOT
from flycraft.game import make_game,initialize_game,pixels,map_action
from flycraft.training_game import reset_episode,observe
from flycraft.training_metrics import SustainedAttack,reward_components
from doom_learning.common import digest


def main():
    run=next((ROOT/'artifacts/training').glob('*_2d'))
    out=ROOT/'artifacts/weight-effect-2d';out.mkdir(exist_ok=True)
    c=json.loads((run/'config.json').read_text())
    g=json.loads((ROOT/'config/baseline.json').read_text());g['port']=8032
    checkpoint=run/'checkpoints/evaluation-step-000060000/brain.npz'
    starts=[x['start'] for x in json.loads((run/'evaluation-000060000.json').read_text())['episodes'][:2]]
    fly=LearningFly(c['plasticity']);fly.freeze();b=fly.brain
    baseline=b.weight.copy();edges=b.circuit['edges']
    with np.load(checkpoint,allow_pickle=False) as data:
        meta=json.loads(str(data['metadata']));learned=data['weight'].copy()
        for k,a in [('graph_ids_sha256',b.ids),('graph_ptr_sha256',b.ptr),('graph_post_sha256',b.post),('plastic_edges_sha256',edges)]:
            assert meta[k]==digest(a),(k,'checkpoint graph mismatch')
        assert learned.shape==baseline.shape and learned.dtype==baseline.dtype
    changed=np.flatnonzero(learned!=baseline)
    assert np.isin(changed,edges).all(),'Checkpoint altered nonplastic edges'
    weights={'baseline':baseline[edges].copy(),'learned':learned[edges].copy(),
             'raised_50pct':(baseline[edges]*np.float32(1.5)).astype(np.float32)}
    del learned
    n=c['training']['max_steps_per_episode'];assert n==400
    mb=b.circuit['mbon'] if 'mbon' in b.circuit else None
    if mb is None:
        from doom_learning.common import annotations
        mb=np.flatnonzero(annotations(b.ids).type.eq('MBON11'))
    all_rows={};initial_hashes={};pixel_hashes={};env=None
    def reset(condition):
        b.weight[:]=baseline;b.weight[edges]=weights[condition]
        fly.reset_episode()
        h=hashlib.sha256()
        for name in b.fields:h.update(getattr(b,name).tobytes())
        h.update(fly.decoder.rates.tobytes());h.update(str(b.cursor).encode())
        return h.hexdigest()
    def step(rgb,attack):
        control,neural=fly.step(rgb)
        action=map_action(control,g);action['attack']=attack.step(control['attack'])
        row={'yaw':action['camera_yaw'],'forward':action['forward'],'attack':action['attack'],
             'mbon_spikes':b.counts[mb].tolist(),
             'readouts':control['readouts'],'spikes_sha256':neural['spikes_sha256']}
        return action,row
    try:
        env=make_game(g);initialize_game(env,g)
        for ep,start in enumerate(starts):
            initial_hashes[f'baseline/{ep}']=reset('baseline')
            obs=reset_episode(env,start);attack=SustainedAttack(c['attack']);rows=[];reward=0.
            frames=np.lib.format.open_memmap(out/f'input-{ep}.npy',mode='w+',dtype=np.uint8,shape=(n,g['height'],g['width'],3))
            h=hashlib.sha256()
            for tick in range(n):
                rgb=pixels(obs,g);frames[tick]=rgb;h.update(rgb.tobytes())
                before=observe(obs,start['target']);action,row=step(rgb,attack)
                obs=env.step(action)[0];after=observe(obs,start['target'])
                value,_=reward_components(before,after,c['reward']);reward+=value
                row.update(tick=tick,reward=value,angle=after['angle'],progress=after['progress'],target_present=after['target_present'])
                rows.append(row)
                if (tick+1)%100==0:print(f'baseline episode {ep+1}: {tick+1}/{n}',flush=True)
            frames.flush();del frames
            pixel_hashes[str(ep)]=h.hexdigest();all_rows[f'baseline/{ep}']=rows
            print(f'baseline episode {ep+1}: reward={reward:.4f}',flush=True)
        env.close();env=None
        for condition in ('learned','raised_50pct'):
            for ep in range(2):
                initial_hashes[f'{condition}/{ep}']=reset(condition)
                assert initial_hashes[f'{condition}/{ep}']==initial_hashes[f'baseline/{ep}']
                frames=np.load(out/f'input-{ep}.npy',mmap_mode='r');h=hashlib.sha256()
                attack=SustainedAttack(c['attack']);rows=[]
                for tick,rgb in enumerate(frames):
                    h.update(rgb.tobytes());_,row=step(rgb,attack);row['tick']=tick;rows.append(row)
                    if (tick+1)%100==0:print(f'{condition} episode {ep+1}: {tick+1}/{n}',flush=True)
                assert h.hexdigest()==pixel_hashes[str(ep)]
                assert np.array_equal(b.weight[edges],weights[condition]),'Frozen weights changed'
                all_rows[f'{condition}/{ep}']=rows
                (out/'partial-results.json').write_text(json.dumps(all_rows))
        report={'protocol':'Two 400-tick baseline Minecraft episodes; exact uncompressed RGB replay to frozen learned and +50% baseline KC->MBON11 weights. Replay actions do not drive Minecraft; no replay reward claims.',
                'checkpoint':str(checkpoint),'checkpoint_step':60000,
                'starts':starts,'initial_state_hashes':initial_hashes,'input_sha256':pixel_hashes,
                'neurons':b.n,'edges':len(b.weight),'plastic_edges':len(edges),'conditions':{}}
        for condition in weights:
            comparisons=[]
            for ep in range(2):
                rows=all_rows[f'{condition}/{ep}'];ref=all_rows[f'baseline/{ep}']
                delta=np.asarray([r['yaw']-a['yaw'] for r,a in zip(rows,ref)])
                rate_deltas={typ:float(np.mean([abs(sum(x['rate_hz'] for x in r['readouts'] if x['type']==typ)-sum(x['rate_hz'] for x in a['readouts'] if x['type']==typ)) for r,a in zip(rows,ref)])) for typ in ('DNp20','DNpe017')}
                comparisons.append({'episode':ep+1,'mean_yaw':float(np.mean([r['yaw'] for r in rows])),
                    'mean_abs_yaw_delta':float(np.mean(abs(delta))),'max_abs_yaw_delta':float(np.max(abs(delta))),
                    'forward_changed_ticks':sum(r['forward']!=a['forward'] for r,a in zip(rows,ref)),
                    'attack_changed_ticks':sum(r['attack']!=a['attack'] for r,a in zip(rows,ref)),
                    'forward_ticks':sum(r['forward'] for r in rows),'attack_ticks':sum(r['attack'] for r in rows),
                    'mbon_spikes_total':np.asarray([r['mbon_spikes'] for r in rows]).sum(axis=0).tolist(),
                    'readout_sum_rate_abs_delta_hz':rate_deltas,
                    'different_whole_brain_spike_ticks':sum(r['spikes_sha256']!=a['spikes_sha256'] for r,a in zip(rows,ref))})
            report['conditions'][condition]={'weight_fraction_mean':float(np.mean(weights[condition]/weights['baseline'])),'episodes':comparisons}
        (out/'results.json').write_text(json.dumps(report,indent=2))
        (out/'tick-results.json').write_text(json.dumps(all_rows))
        print(json.dumps(report['conditions'],indent=2),flush=True)
    finally:
        if env:env.close()


if __name__=='__main__':main()
