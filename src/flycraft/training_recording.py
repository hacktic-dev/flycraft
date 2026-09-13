"""Selective episode recording using the existing 1080p dashboard and writers."""
import gzip,json
from pathlib import Path
import numpy as np
from PIL import Image
from .visuals import Visuals,fit
from .activity import ActivityWriter,VoltageWriter
from .trajectory import TrajectoryWriter


class EpisodeRecording:
    def __init__(self,folder,fly,config,game_config,initial,history,preview=False,activity_mode='spikes',voltage_smoothing_ms=80.0):
        self.folder=Path(folder);self.folder.mkdir(parents=True)
        self.view=Visuals(fly,self.folder,record=config['record_dashboard'],preview=preview,activity_mode=activity_mode,voltage_smoothing_ms=voltage_smoothing_ms)
        self.view.record_names={'dashboard.mp4','retina.mp4','neural-activity.mp4'}
        self.render=config['record_dashboard'] or preview
        self.activity=ActivityWriter(self.folder,fly.brain.ids) if config['record_activity'] else None
        self.voltage=VoltageWriter(self.folder,fly.brain.n,simulation_dt_ms=fly.brain.dt) if config['record_activity'] else None
        self.trajectory=TrajectoryWriter(self.folder,game_config,initial)
        self.log=gzip.open(self.folder/'steps-00000.jsonl.gz','wt',encoding='utf-8')
        self.video=None
        if config['record_minecraft']:
            import imageio_ffmpeg
            self.video=imageio_ffmpeg.write_frames(str(self.folder/'baseline.mp4'),(1920,1080),fps=60,macro_block_size=1,codec='libx264',pix_fmt_in='rgb24',pix_fmt_out='yuv420p',output_params=['-crf','18','-preset','veryfast'])
            self.video.send(None)
        (self.folder/'reward-history.json').write_text(json.dumps(history))
        meta=json.loads((self.folder/'visuals.json').read_text());meta.update(learning=fly.learning,model=fly.config['model'],visual_model=getattr(fly.brain,'visual_report',None))
        (self.folder/'visuals.json').write_text(json.dumps(meta,indent=2))

    def write(self,fly,rgb,preview_rgb,control,action,before,after,tick,neural,panel):
        if self.activity:self.activity.write(fly.activity_bins)
        if self.voltage:self.voltage.write(fly.last_voltage,tick,fly.brain.sim_ms)
        self.trajectory.write(tick,before,after,action)
        self.log.write(json.dumps({'frame':tick,'controller':control,'action':action,'before':before,'after':after,'training':{k:v for k,v in panel.items() if k!='history'},**neural})+'\n')
        if self.video:
            frame=np.asarray(fit(Image.fromarray(preview_rgb)))
            for _ in range(3):self.video.send(frame)
        if self.render:self.view.update(rgb,control,tick,preview_rgb=preview_rgb)

    def close(self,summary):
        for item in (self.activity,self.voltage,self.trajectory,self.log,self.video,self.view):
            if item:item.close()
        (self.folder/'episode.json').write_text(json.dumps(summary,indent=2))
