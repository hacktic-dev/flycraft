"""Episode setup and reward-only telemetry; never consulted by the controller."""
import math, struct
from .game import telemetry
from .training_metrics import target_metrics
from craftground.environment.action_space import no_op_v2


def randomized_start(rng,config):
    radius=config['player_spawn_radius']
    px,pz=rng.uniform(-radius,radius,2) if config['randomize_player_position'] else (0.5,.5)
    # Sample a real integer log cell and reject discretization outside the range.
    for _ in range(1000):
        angle=rng.uniform(-math.pi,math.pi) if config['randomize_tree_position'] else 0.
        distance=rng.uniform(config['min_tree_distance'],config['max_tree_distance']) if config['randomize_tree_position'] else (config['min_tree_distance']+config['max_tree_distance'])/2
        tx,tz=math.floor(px+math.sin(angle)*distance),math.floor(pz+math.cos(angle)*distance)
        d=math.hypot(tx+.5-px,tz+.5-pz)
        if config['min_tree_distance']<=d<=config['max_tree_distance']:break
    else:raise ValueError('Cannot place integer tree within configured distance range')
    return {'x':float(px),'z':float(pz),'yaw':float(rng.uniform(-180,180)) if config['randomize_yaw'] else 0.,'target':[tx,-58,tz]}


def reset_episode(env,start):
    x,y,z=start['target']
    env.add_commands(['gamemode creative @p',
        'fill -18 -59 -18 18 -52 18 minecraft:air',
        f'fill {x-2} -56 {z-2} {x+2} -54 {z+2} minecraft:oak_leaves[persistent=true]',
        f'fill {x} -59 {z} {x} -55 {z} minecraft:oak_log',
        f'tp @p {start["x"]} -59 {start["z"]} {start["yaw"]} 0',
        'clear @p','gamemode survival @p',f'flycraft_target {x} {y} {z}'])
    # No neural time/reward during world construction and command settlement.
    for _ in range(12):obs=env.step(no_op_v2())[0]
    state=observe(obs,start['target'])
    if not state['target_present'] or abs(state['x']-start['x'])>.08 or abs(state['z']-start['z'])>.08:
        raise RuntimeError(f'Training arena did not settle: {state}')
    return obs


def observe(obs,target):
    p=telemetry(obs);s=obs['full'].misc_statistics
    if 'flycraft.progress_bits' not in s:raise RuntimeError('Missing genuine breaking telemetry; run scripts/prepare_training_runtime.py and restart Minecraft')
    if tuple(s['flycraft.target_'+k] for k in 'xyz')!=tuple(target):raise RuntimeError('Target telemetry mismatch')
    block=obs['full'].raycast_result.target_block
    on_target=(block.x,block.y,block.z)==tuple(target) and block.translation_key=='block.minecraft.oak_log'
    progress=struct.unpack('<f',struct.pack('<I',s['flycraft.progress_bits']))[0]
    if not math.isfinite(progress) or not 0<=progress<=1:raise RuntimeError('Invalid Minecraft breaking progress')
    return {**p,**target_metrics(p,target),'progress':progress,'on_target':bool(on_target),'target_present':bool(s['flycraft.target_present']),'target':list(target)}
