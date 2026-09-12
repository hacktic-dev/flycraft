"""Interactive offline player for recorded anatomical neural activity.

Loads activity-60hz.jsonl.gz plus neuron-layout.npz and, when present,
voltage-20hz/.  Playback never runs Minecraft or the neural simulation.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .activity import VoltageReader
from .visuals import BG, FG, CYAN, HD, Visuals

FPS=60.0
TRAIL_SECONDS=2.0
VOLTAGE_REBUILD_SECONDS=1.5
SPEEDS=(0.25,0.5,1.0,2.0,4.0)


class ActivityTimeline:
    """Disk-backed sparse cache for fast random access to gzipped JSONL."""
    def __init__(self,source:Path,ids:np.ndarray):
        self.source=Path(source)
        self.ids=np.asarray(ids)
        self.cache=self.source.parent/'.activity-player-cache'
        self.cache.mkdir(exist_ok=True)
        self.meta_path=self.cache/'meta.json'
        self.offsets_path=self.cache/'offsets.npy'
        self.times_path=self.cache/'times.npy'
        self.frames_path=self.cache/'frames.npy'
        self.indices_path=self.cache/'indices.bin'
        self.counts_path=self.cache/'counts.bin'
        signature={
            'source_name':self.source.name,
            'source_size':self.source.stat().st_size,
            'source_mtime_ns':self.source.stat().st_mtime_ns,
            'ids_sha256':hashlib.sha256(self.ids.tobytes()).hexdigest(),
        }
        valid=False
        if self.meta_path.exists():
            try:
                old=json.loads(self.meta_path.read_text())
                valid=all(old.get(k)==v for k,v in signature.items())
                valid=valid and all(p.exists() for p in (self.offsets_path,self.times_path,self.frames_path,self.indices_path,self.counts_path))
            except Exception:valid=False
        if not valid:self._build(signature)
        self.meta=json.loads(self.meta_path.read_text())
        self.offsets=np.load(self.offsets_path,mmap_mode='r')
        self.times=np.load(self.times_path,mmap_mode='r')
        self.frames=np.load(self.frames_path,mmap_mode='r')
        self.indices=np.memmap(self.indices_path,dtype=np.int32,mode='r')
        self.counts=np.memmap(self.counts_path,dtype=np.uint16,mode='r')
        self.frame_count=len(self.times)
        self.n=len(self.ids)
        self._dense=np.zeros(self.n,dtype=np.int32)

    def _build(self,signature):
        print('Building seek cache for activity JSONL (first run only)...',flush=True)
        id_to_index={int(body):i for i,body in enumerate(self.ids)}
        offsets=[0];times=[];frames=[];pairs=0;unknown=set()
        with self.indices_path.open('wb') as fi,self.counts_path.open('wb') as fc,gzip.open(self.source,'rt',encoding='utf-8') as src:
            for line_no,line in enumerate(src,1):
                row=json.loads(line);active=row.get('active',[])
                if active:
                    ix=np.empty(len(active),dtype=np.int32);ct=np.empty(len(active),dtype=np.uint16);used=0
                    for body,count in active:
                        mapped=id_to_index.get(int(body))
                        if mapped is None:
                            unknown.add(int(body));continue
                        ix[used]=mapped;ct[used]=min(65535,max(0,int(count)));used+=1
                    ix[:used].tofile(fi);ct[:used].tofile(fc);pairs+=used
                offsets.append(pairs)
                frames.append(int(row.get('frame',len(frames))))
                times.append(float(row.get('t',len(times)/FPS)))
                if line_no%600==0:print(f'  cached {line_no:,} frames',flush=True)
        if unknown:raise RuntimeError(f'Activity recording contains {len(unknown)} neuron IDs absent from neuron-layout.npz')
        np.save(self.offsets_path,np.asarray(offsets,dtype=np.int64))
        np.save(self.times_path,np.asarray(times,dtype=np.float64))
        np.save(self.frames_path,np.asarray(frames,dtype=np.int64))
        meta={**signature,'frame_count':len(times),'pair_count':pairs,'fps':FPS}
        self.meta_path.write_text(json.dumps(meta,indent=2))

    def dense(self,frame:int):
        frame=int(np.clip(frame,0,self.frame_count-1))
        a=int(self.offsets[frame]);b=int(self.offsets[frame+1])
        self._dense.fill(0)
        self._dense[self.indices[a:b]]=self.counts[a:b]
        return self._dense


class ControlTimeline:
    def __init__(self,folder:Path):
        self.controls={}
        for p in sorted(Path(folder).glob('steps-*.jsonl.gz')):
            with gzip.open(p,'rt',encoding='utf-8') as f:
                for line in f:
                    row=json.loads(line)
                    if 'controller' in row:self.controls[int(row['frame'])]=row['controller']
    def get(self,activity_frame:int):
        return self.controls.get(activity_frame//3,{'readouts':[]})


class OfflinePlayer:
    def __init__(self,activity_file:Path,mode='combined',smoothing_ms=80.0):
        self.activity_file=Path(activity_file)
        self.folder=self.activity_file.parent
        layout_path=self.folder/'neuron-layout.npz'
        if not layout_path.exists():raise FileNotFoundError(f'Missing {layout_path}')
        self.layout=np.load(layout_path)
        brain=SimpleNamespace(
            ids=self.layout['ids'],n=len(self.layout['ids']),superclass=self.layout['superclass'],
            retina=self.layout['retina_indices'],uv=self.layout['retina_uv'])
        fly=SimpleNamespace(brain=brain,samples=np.zeros(len(brain.retina)),last_voltage=None)
        self.view=Visuals(fly,self.folder/'.activity-player',preview=False,layout=self.layout,playback=True,activity_mode=mode,voltage_smoothing_ms=smoothing_ms)
        self.view.skeletons={}
        for p in self.folder.glob('skeleton-*.npz'):
            with np.load(p) as a:self.view.skeletons[int(p.stem.split('-')[1])]=(a['xyz_nm'].copy(),a['edges'].copy())
        self.activity=ActivityTimeline(self.activity_file,brain.ids)
        self.controls=ControlTimeline(self.folder)
        voltage_folder=self.folder/'voltage-20hz'
        self.voltage=VoltageReader(voltage_folder) if (voltage_folder/'metadata.json').exists() else None
        if mode in ('voltage','combined') and self.voltage is None:
            print('No voltage-20hz recording found; using Spikes mode for this older recording.')
            self.view.activity_mode='spikes'
        if self.voltage and self.voltage.neuron_count!=brain.n:
            raise RuntimeError('Voltage recording neuron count does not match layout')
        self.frame=0;self.last_state_frame=None;self.current_counts=np.zeros(brain.n,dtype=np.int32)
        self._rebuild_state(0)

    @property
    def duration(self):return self.activity.frame_count/FPS

    def reset_camera(self):
        self.view.yaw=.25;self.view.pitch=-.7;self.view.roll=np.pi;self.view.zoom=1.0;self.view.scene_cache.clear()

    def _voltage_target(self,frame):
        if not self.voltage:return None
        return self.voltage.get(frame//3)

    def _apply_voltage_step(self,frame):
        if self.view.activity_mode not in ('voltage','combined') or not self.voltage:return
        target=self._voltage_target(frame)
        if self.view.display_voltage is None:
            self.view.display_voltage=target.copy();return
        if self.view.voltage_smoothing_ms<=0:
            self.view.display_voltage=target.copy();return
        tau=self.view.voltage_smoothing_ms/1000.0
        alpha=1.0-math.exp(-(1.0/FPS)/tau)
        self.view.display_voltage += np.float32(alpha)*(target-self.view.display_voltage)

    def _rebuild_state(self,frame):
        frame=int(np.clip(frame,0,self.activity.frame_count-1))
        self.view.trail.fill(0)
        trail_start=max(0,frame-int(TRAIL_SECONDS*FPS))
        decay=math.exp(-(1.0/FPS)/.25)
        for f in range(trail_start,frame+1):
            c=self.activity.dense(f)
            self.view.trail=np.maximum(c,self.view.trail*decay)
        if self.view.activity_mode in ('voltage','combined') and self.voltage:
            voltage_start=max(0,frame-int(VOLTAGE_REBUILD_SECONDS*FPS))
            self.view.display_voltage=self._voltage_target(voltage_start).copy()
            for f in range(voltage_start+1,frame+1):self._apply_voltage_step(f)
        self.current_counts=self.activity.dense(frame).copy()
        self.frame=frame;self.last_state_frame=frame

    def set_frame(self,frame):
        frame=int(np.clip(frame,0,self.activity.frame_count-1))
        if self.last_state_frame is not None and frame==self.last_state_frame+1:
            self.current_counts=self.activity.dense(frame).copy()
            self.view.trail=np.maximum(self.current_counts,self.view.trail*math.exp(-(1.0/FPS)/.25))
            self._apply_voltage_step(frame)
            self.frame=frame;self.last_state_frame=frame
        elif frame!=self.last_state_frame:self._rebuild_state(frame)

    def set_mode(self,mode):
        if mode in ('voltage','combined') and not self.voltage:return
        self.view.activity_mode=mode
        self._rebuild_state(self.frame)

    def image(self):
        t=float(self.activity.times[self.frame]) if self.frame<len(self.activity.times) else self.frame/FPS
        return self.view.activity_image(self.current_counts,self.controls.get(self.frame),t,1.0/FPS)


def format_time(seconds):
    seconds=max(0,float(seconds));m=int(seconds//60);s=seconds-m*60
    return f'{m:02d}:{s:05.2f}'


def run_ui(player:OfflinePlayer):
    import pygame
    pygame.init();pygame.display.set_caption('DOOMFLY Activity Player')
    screen=pygame.display.set_mode((1600,900),pygame.RESIZABLE)
    clock=pygame.time.Clock();small=pygame.font.SysFont('consolas',17);medium=pygame.font.SysFont('consolas',20)
    playing=False;speed_index=2;playhead=float(player.frame);dirty=True;drag_rotate=False;drag_roll=False;scrubbing=False;last_mouse=(0,0)
    running=True
    while running:
        dt=clock.tick(60)/1000.0
        w,h=screen.get_size();timeline_y=h-62
        for e in pygame.event.get():
            if e.type==pygame.QUIT:running=False;continue
            if e.type==pygame.KEYDOWN:
                shift=bool(e.mod & pygame.KMOD_SHIFT)
                if e.key==pygame.K_ESCAPE:running=False
                elif e.key==pygame.K_SPACE:playing=not playing;playhead=float(player.frame)
                elif e.key==pygame.K_LEFT:
                    playing=False;player.set_frame(player.frame-(300 if shift else 60));playhead=float(player.frame);dirty=True
                elif e.key==pygame.K_RIGHT:
                    playing=False;player.set_frame(player.frame+(300 if shift else 60));playhead=float(player.frame);dirty=True
                elif e.key==pygame.K_COMMA:
                    playing=False;player.set_frame(player.frame-1);playhead=float(player.frame);dirty=True
                elif e.key==pygame.K_PERIOD:
                    playing=False;player.set_frame(player.frame+1);playhead=float(player.frame);dirty=True
                elif e.key==pygame.K_HOME:
                    playing=False;player.set_frame(0);playhead=0;dirty=True
                elif e.key==pygame.K_END:
                    playing=False;player.set_frame(player.activity.frame_count-1);playhead=float(player.frame);dirty=True
                elif e.key==pygame.K_LEFTBRACKET:speed_index=max(0,speed_index-1);dirty=True
                elif e.key==pygame.K_RIGHTBRACKET:speed_index=min(len(SPEEDS)-1,speed_index+1);dirty=True
                elif e.key==pygame.K_a:player.view.yaw-=.12;player.view.scene_cache.clear();dirty=True
                elif e.key==pygame.K_d:player.view.yaw+=.12;player.view.scene_cache.clear();dirty=True
                elif e.key==pygame.K_w:player.view.pitch+=.12;player.view.scene_cache.clear();dirty=True
                elif e.key==pygame.K_s:player.view.pitch-=.12;player.view.scene_cache.clear();dirty=True
                elif e.key==pygame.K_q:player.view.roll-=.12;player.view.scene_cache.clear();dirty=True
                elif e.key==pygame.K_e:player.view.roll+=.12;player.view.scene_cache.clear();dirty=True
                elif e.key in (pygame.K_EQUALS,pygame.K_PLUS):player.view.zoom=min(8,player.view.zoom*1.15);player.view.scene_cache.clear();dirty=True
                elif e.key==pygame.K_MINUS:player.view.zoom=max(.2,player.view.zoom/1.15);player.view.scene_cache.clear();dirty=True
                elif e.key==pygame.K_r:player.reset_camera();dirty=True
                elif e.key==pygame.K_1:player.set_mode('spikes');dirty=True
                elif e.key==pygame.K_2:player.set_mode('voltage');dirty=True
                elif e.key==pygame.K_3:player.set_mode('combined');dirty=True
            elif e.type==pygame.MOUSEBUTTONDOWN:
                if e.button==1 and e.pos[1]>=timeline_y:
                    scrubbing=True;playing=False
                    target=round(np.clip(e.pos[0]/max(1,w-1),0,1)*(player.activity.frame_count-1));player.set_frame(target);playhead=float(target);dirty=True
                elif e.button==1:drag_rotate=True;last_mouse=e.pos
                elif e.button==3:drag_roll=True;last_mouse=e.pos
                elif e.button==4:player.view.zoom=min(8,player.view.zoom*1.1);player.view.scene_cache.clear();dirty=True
                elif e.button==5:player.view.zoom=max(.2,player.view.zoom/1.1);player.view.scene_cache.clear();dirty=True
            elif e.type==pygame.MOUSEBUTTONUP:
                if e.button==1:drag_rotate=False;scrubbing=False
                if e.button==3:drag_roll=False
            elif e.type==pygame.MOUSEMOTION:
                if scrubbing:
                    target=round(np.clip(e.pos[0]/max(1,w-1),0,1)*(player.activity.frame_count-1));player.set_frame(target);playhead=float(target);dirty=True
                elif drag_rotate:
                    dx=e.pos[0]-last_mouse[0];dy=e.pos[1]-last_mouse[1];last_mouse=e.pos
                    player.view.yaw+=dx*.006;player.view.pitch-=dy*.006;player.view.scene_cache.clear();dirty=True
                elif drag_roll:
                    dx=e.pos[0]-last_mouse[0];last_mouse=e.pos
                    player.view.roll+=dx*.006;player.view.scene_cache.clear();dirty=True
        if playing:
            playhead+=dt*FPS*SPEEDS[speed_index]
            target=min(player.activity.frame_count-1,int(playhead))
            if target!=player.frame:player.set_frame(target);dirty=True
            if target>=player.activity.frame_count-1:playing=False
        if dirty:
            im=player.image();surf=pygame.image.frombuffer(im.tobytes(),im.size,'RGB')
            # Reserve the bottom transport strip and letterbox the 16:9 render.
            avail_h=max(1,h-64);scale=min(w/im.width,avail_h/im.height);size=(max(1,int(im.width*scale)),max(1,int(im.height*scale)))
            screen.fill(BG);scaled=pygame.transform.smoothscale(surf,size);screen.blit(scaled,((w-size[0])//2,(avail_h-size[1])//2))
            # Video-like transport/timeline UI.
            pygame.draw.rect(screen,(7,10,17),(0,timeline_y,w,h-timeline_y))
            frac=player.frame/max(1,player.activity.frame_count-1)
            pygame.draw.rect(screen,(42,55,70),(18,timeline_y+15,max(1,w-36),6),border_radius=3)
            pygame.draw.rect(screen,CYAN,(18,timeline_y+15,int((w-36)*frac),6),border_radius=3)
            x=18+int((w-36)*frac);pygame.draw.circle(screen,(240,244,248),(x,timeline_y+18),7)
            status='PLAY' if playing else 'PAUSE'
            now=player.frame/FPS
            text=f'{status}   {format_time(now)} / {format_time(player.duration)}   {SPEEDS[speed_index]:g}x   {player.view.activity_mode.upper()}'
            screen.blit(medium.render(text,True,FG),(18,timeline_y+30))
            help_text='Space play/pause | click/drag timeline | ←/→ 1s (Shift 5s) | ,/. frame | [/] speed | mouse drag rotate | right-drag roll | wheel zoom | 1/2/3 mode | R reset'
            help_surface=small.render(help_text,True,(175,190,205));screen.blit(help_surface,(max(18,w-help_surface.get_width()-18),timeline_y+34))
            pygame.display.flip();dirty=False
    pygame.quit()


def resolve_source(path:Path):
    path=Path(path)
    if path.is_dir():path=path/'activity-60hz.jsonl.gz'
    if not path.exists():raise FileNotFoundError(path)
    return path


def main():
    parser=argparse.ArgumentParser(description='Interactive 3D playback for recorded DOOMFLY neural activity')
    parser.add_argument('activity',type=Path,help='activity-60hz.jsonl.gz or its recording folder')
    parser.add_argument('--mode',choices=('spikes','voltage','combined'),default='combined')
    parser.add_argument('--voltage-smoothing-ms',type=float,default=80.0)
    args=parser.parse_args()
    player=OfflinePlayer(resolve_source(args.activity),mode=args.mode,smoothing_ms=args.voltage_smoothing_ms)
    run_ui(player)


if __name__=='__main__':main()
