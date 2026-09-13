"""CraftGround boundary: setup and telemetry stay outside the neural controller."""
import numpy as np
import craftground
from craftground.environment.action_space import ActionSpaceVersion,no_op_v2
from craftground.initial_environment_config import InitialEnvironmentConfig,GameMode,Difficulty,WorldType

def arena_commands(c):
    x,z=c['target_x'],c['target_z']
    return ['gamemode creative @p','gamerule doMobSpawning false','gamerule doDaylightCycle false','gamerule doWeatherCycle false','gamerule randomTickSpeed 0','gamerule sendCommandFeedback false','time set noon','weather clear','kill @e[type=!minecraft:player]',
      'fill -20 -60 -20 20 -60 20 minecraft:smooth_stone','fill -20 -59 -20 20 -53 20 minecraft:air',
      'fill -20 -59 -20 20 -56 -20 minecraft:blue_concrete','fill -20 -59 20 20 -56 20 minecraft:yellow_concrete',
      'fill -20 -59 -20 -20 -56 20 minecraft:white_concrete','fill 20 -59 -20 20 -56 20 minecraft:black_concrete',
      f'fill {x-1} -59 {z} {x+1} -55 {z} {c["target_block"]}',
      'clear @p','tp @p 0.5 -59.0 0.5 0 0','effect give @p minecraft:resistance infinite 255 true']

def _capture_size(c):
    """Use CraftGround's native 16:9 render aspect, then crop to our 4:3 neural view.

    CraftGround/Minecraft 1.21 renders the scene/HUD against a 16:9 client
    framebuffer on Windows. Asking it directly for 640x480 produced a 16:9
    image horizontally compressed into 4:3 pixels. Capture at the matching
    16:9 width instead, then take an undistorted centered 4:3 crop ourselves.
    """
    h=int(c['height'])
    return max(int(c['width']),int(np.ceil(h*16/9))),h

def make_game(c):
    capture_w,capture_h=_capture_size(c)
    initial=InitialEnvironmentConfig(image_width=capture_w,image_height=capture_h,gamemode=GameMode.CREATIVE,difficulty=Difficulty.PEACEFUL,world_type=WorldType.SUPERFLAT,seed=str(c['seed']),generate_structures=False,initial_extra_commands=arena_commands(c),hud_hidden=False,render_distance=4,simulation_distance=5,no_fov_effect=True,request_raycast=True)
    return craftground.make(initial_env_config=initial,port=c['port'],action_space_version=ActionSpaceVersion.V2_MINERL_HUMAN,verbose_gradle=False,verbose_jvm=False)

def pixels(obs,c):
    """Return the exact undistorted RGB frame used by FlyCraft (normally 640x480)."""
    rgb=np.asarray(obs['rgb'])
    if rgb.ndim!=3 or rgb.shape[2]!=3 or rgb.dtype!=np.uint8: raise RuntimeError(f'Invalid Minecraft RGB: {rgb.shape} {rgb.dtype}')
    target_w,target_h=int(c['width']),int(c['height'])
    if rgb.shape[0]!=target_h:
        raise RuntimeError(f'Unexpected Minecraft capture height: got {rgb.shape}, expected height {target_h}')
    if rgb.shape[1]<target_w:
        raise RuntimeError(f'Minecraft capture is narrower than requested FlyCraft view: got {rgb.shape}, target {target_w}x{target_h}')
    # Center-crop the correctly-proportioned 16:9 capture to the configured 4:3
    # neural/dashboard view. No stretching is performed here.
    x=(rgb.shape[1]-target_w)//2
    rgb=rgb[:,x:x+target_w]
    if rgb.shape!=(target_h,target_w,3):
        raise RuntimeError(f'Invalid cropped Minecraft RGB: {rgb.shape}, expected {(target_h,target_w,3)}')
    return np.ascontiguousarray(rgb)

def preview_pixels(obs):
    """Return Minecraft's uncropped render for human-facing preview/recording.

    This is intentionally separate from :func:`pixels`, which returns the
    centered 4:3 crop used as the fly's neural sensory input.
    """
    rgb=np.asarray(obs['rgb'])
    if rgb.ndim!=3 or rgb.shape[2]!=3 or rgb.dtype!=np.uint8:
        raise RuntimeError(f'Invalid Minecraft preview RGB: {rgb.shape} {rgb.dtype}')
    return np.ascontiguousarray(rgb)


def telemetry(obs):
    full=obs['full']
    names=['x','y','z','yaw','pitch','world_time','game_time','health']
    return {name:getattr(full,name) for name in names if hasattr(full,name)}

def map_action(control,c):
    action=no_op_v2()
    # Bilateral steering already produces Minecraft degrees per control tick.
    # Applying the legacy negative gain here would invert and amplify it twice.
    direct=control.get('decoder',{}).get('turn_units')=='degrees_per_tick'
    yaw=control['turn'] if direct else control['turn']*c['yaw_gain']
    action['camera_yaw']=float(np.clip(yaw,-c['max_yaw_degrees'],c['max_yaw_degrees']))
    action['forward']=bool(control['forward']>c['forward_threshold'])
    action['attack']=bool(control['attack'] and c['attack_enabled'])
    return action



def initialize_game(env,c):
    """Finish environment-only startup before the neural clock begins."""
    obs,_=env.reset(seed=c['seed'])
    # CraftGround returns an early frame while its initialization commands and
    # client/server time packets are still pending. Drain them with no input.
    for _ in range(40): obs=env.step(no_op_v2())[0]
    env.add_commands(arena_commands(c))
    for _ in range(20): obs=env.step(no_op_v2())[0]
    t=telemetry(obs)
    if abs(t['x']-.5)>.05 or abs(t['z']-.5)>.05 or abs(t['y']+59)>.05:
        raise RuntimeError(f'Arena failed to initialize at fixed pose: {t}')
    return obs
