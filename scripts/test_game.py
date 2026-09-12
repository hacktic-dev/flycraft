"""Verify real CraftGround frames, turning, walking, attack, and reset."""
import json,math
from pathlib import Path
from PIL import Image
from flycraft.game import make_game,pixels,telemetry,arena_commands
from craftground.environment.action_space import no_op_v2
ROOT=Path(__file__).resolve().parents[1]
c=json.loads((ROOT/'config/baseline.json').read_text(encoding='utf-8-sig'))
out=ROOT/'artifacts/craftground-proof';out.mkdir(parents=True,exist_ok=True)
report={}
env=make_game(c)
def step(**keys):
    action=no_op_v2();action.update(keys)
    return env.step(action)[0]
def settle(n=3):
    for _ in range(n): obs=step()
    return obs
try:
    obs,_=env.reset(seed=c['seed'])
    obs=settle()
    rgb=pixels(obs,c)
    assert rgb.shape==(c['height'],c['width'],3) and rgb.std()>3
    Image.fromarray(rgb).save(out/'minecraft-rgb.png')
    report['rgb']={'shape':list(rgb.shape),'std':float(rgb.std())}
    before=telemetry(obs);obs=step(camera_yaw=30.0);obs=settle()
    after=telemetry(obs);delta=(after['yaw']-before['yaw']+180)%360-180
    assert abs(delta)>10, (before,after)
    report['turn']={'before':before,'after':after,'delta_degrees':delta}
    before=after
    for _ in range(20): obs=step(forward=True)
    after=telemetry(obs);distance=math.hypot(after['x']-before['x'],after['z']-before['z'])
    assert distance>1,(before,after)
    report['walk']={'before':before,'after':after,'distance':distance}
    settle(12) # Release movement and let momentum decay before reset.
    obs,_=env.reset(seed=c['seed'],options={'extra_commands':arena_commands(c)})
    obs=settle(5)
    assert abs(obs['full'].x-.5)<.2 and abs(obs['full'].z-.5)<.2,telemetry(obs)
    report['reset']=telemetry(obs)
    env.add_commands(['setblock 0 -58 2 minecraft:gold_block','tp @p 0.5 -59 0.5 0 0'])
    obs=settle(4)
    hit=obs['full'].raycast_result
    assert hit.target_block.translation_key=='block.minecraft.gold_block',str(hit)
    Image.fromarray(pixels(obs,c)).save(out/'before-attack.png')
    obs=step(attack=True)
    obs=settle(3)
    assert obs['full'].raycast_result.target_block.translation_key!='block.minecraft.gold_block',str(obs['full'].raycast_result)
    report['attack']={'gold_block_destroyed':True}
    Image.fromarray(pixels(obs,c)).save(out/'after-attack.png')
    report['passed']=True
    print('CRAFTGROUND INDEPENDENT TEST: PASS',json.dumps(report),flush=True)
finally:
    (out/'report.json').write_text(json.dumps(report,indent=2))
    env.close()

