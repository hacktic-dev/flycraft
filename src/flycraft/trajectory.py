"""Compact Minecraft pose/action recording for later cinematic replay.

This is deliberately separate from neural activity recording.  One JSONL row is
written per Minecraft tick (20 Hz) plus an initial pose at t=0.  The stream is
gzip-compressed so it stays small while remaining human/tool friendly.
"""
import gzip
import json
from pathlib import Path

POSE_KEYS=("x","y","z","yaw","pitch")
OPTIONAL_KEYS=("world_time","game_time","health")


def _pose(state):
    """Copy only stable scalar state needed for replay/camera matching."""
    out={k:float(state[k]) for k in POSE_KEYS if k in state}
    for k in OPTIONAL_KEYS:
        if k in state:
            value=state[k]
            # Preserve integer clocks where possible, floats otherwise.
            out[k]=int(value) if k in ("world_time","game_time") else float(value)
    missing=[k for k in POSE_KEYS if k not in out]
    if missing:
        raise RuntimeError(f"Minecraft telemetry missing replay pose fields: {missing}")
    return out


class TrajectoryWriter:
    """Write an exact 20 Hz player pose trace and the action that led to it."""
    def __init__(self,folder,config,initial_state):
        self.folder=Path(folder)
        self.path=self.folder/'minecraft-trajectory.jsonl.gz'
        self.stream=gzip.open(self.path,'wt',encoding='utf-8',compresslevel=5)
        self.rows=0
        meta={
            'format':'flycraft-minecraft-trajectory-v1',
            'tick_hz':20,
            'tick_seconds':0.05,
            'coordinate_system':'Minecraft world coordinates; yaw/pitch are CraftGround/Minecraft degrees',
            'pose_fields':list(POSE_KEYS),
            'config':{
                'seed':config.get('seed'),
                'target_x':config.get('target_x'),
                'target_z':config.get('target_z'),
                'target_block':config.get('target_block'),
            },
        }
        (self.folder/'minecraft-trajectory-meta.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
        # Initial sample means a replay can be positioned correctly before frame 0.
        self.stream.write(json.dumps({'frame':-1,'t':0.0,'pose':_pose(initial_state)},separators=(',',':'))+'\n')

    def write(self,frame,before,after,action):
        # `after` is the authoritative resulting pose after this tick's action.
        row={
            'frame':int(frame),
            't':(int(frame)+1)/20.0,
            'pose':_pose(after),
            'action':{
                'camera_yaw':float(action.get('camera_yaw',0.0)),
                'camera_pitch':float(action.get('camera_pitch',0.0)),
                'forward':bool(action.get('forward',False)),
                'back':bool(action.get('back',False)),
                'left':bool(action.get('left',False)),
                'right':bool(action.get('right',False)),
                'jump':bool(action.get('jump',False)),
                'sneak':bool(action.get('sneak',False)),
                'sprint':bool(action.get('sprint',False)),
                'attack':bool(action.get('attack',False)),
                'use':bool(action.get('use',False)),
            },
        }
        # Keep pre-step pose too.  It costs little at 20 Hz and makes auditing,
        # collision debugging and exact action reconstruction much easier.
        row['before']=_pose(before)
        self.stream.write(json.dumps(row,separators=(',',':'))+'\n')
        self.rows+=1
        if self.rows%20==0:self.stream.flush()

    def close(self):
        if self.stream:
            self.stream.flush();self.stream.close();self.stream=None
