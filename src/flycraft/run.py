"""Run the original frozen connectome against live CraftGround RGB."""
import argparse,gzip,hashlib,json,time
from datetime import datetime
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from .brain import FrozenFly
from .game import make_game,pixels,preview_pixels,telemetry,map_action,initialize_game
ROOT=Path(__file__).resolve().parents[2]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--steps',type=int,default=0,help='0 runs until Ctrl+C')
    parser.add_argument('--record',action='store_true')
    parser.add_argument('--debug',action='store_true')
    parser.add_argument('--record-visuals',action='store_true')
    parser.add_argument('--record-activity',action='store_true')
    parser.add_argument('--record-trajectory',action='store_true',help='record exact Minecraft pose/action trace at 20 Hz')
    parser.add_argument('--activity-mode',choices=('spikes','voltage','combined'),default='spikes')
    parser.add_argument('--voltage-smoothing-ms',type=float,default=80.0,help='Display-only membrane-voltage smoothing; 0 disables')
    parser.add_argument('--playback',type=Path)
    parser.add_argument('--no-preview',action='store_true')
    parser.add_argument('--config',type=Path,default=ROOT/'config/baseline.json')
    args=parser.parse_args()
    if args.playback:
        from .visuals import Visuals
        print(Visuals.playback(args.playback,preview=not args.no_preview,limit=args.steps,activity_mode=args.activity_mode,voltage_smoothing_ms=args.voltage_smoothing_ms))
        return
    if args.record_visuals: args.record=True;args.record_activity=True
    # Any footage/data recording automatically preserves the Minecraft trajectory.
    # --record-trajectory also allows pose-only capture with no video/neural dump.
    record_trajectory=args.record_trajectory or args.record or args.record_visuals or args.record_activity
    c=json.loads(args.config.read_text(encoding='utf-8-sig'))
    if c.get('learning') is not False: raise ValueError('Only LEARNING: OFF is supported')
    if c['neural_ms_per_tick']!=50.0: raise ValueError('Minecraft 20 Hz requires exactly 50 ms neural time per tick')
    out=ROOT/'artifacts'/datetime.now().strftime('baseline-%Y%m%d-%H%M%S')
    out.mkdir(parents=True)
    (out/'config.json').write_text(json.dumps(c,indent=2))
    print(f'LEARNING: OFF | ORIGINAL FROZEN BRAIN | logs: {out}',flush=True)
    # Voltage/combined preview benefits from the observer's three 60 Hz bins per
    # 50 ms game tick, even when the sparse activity file is not being recorded.
    observe_activity=args.record_activity or (not args.no_preview and args.activity_mode in ('voltage','combined'))
    fly=FrozenFly(observe_activity=observe_activity)
    env=None;writer=None;log=None;visuals=None;activity_writer=None;trajectory_writer=None;steps=0;start=time.perf_counter()
    summary={'learning':False,'neurons':fly.brain.n,'edges':len(fly.brain.weight),'weight_sha256_before':fly.initial_weights,'complete':False}
    try:
        env=make_game(c)
        obs=initialize_game(env,c)
        rgb=pixels(obs,c)
        preview_rgb=preview_pixels(obs)
        Image.fromarray(rgb).save(out/'first-frame.png')
        Image.fromarray(preview_rgb).save(out/'first-frame-preview.png')
        summary['initial_game']=telemetry(obs)
        if record_trajectory:
            from .trajectory import TrajectoryWriter
            trajectory_writer=TrajectoryWriter(out,c,summary['initial_game'])
        if args.record_visuals or args.record_activity or not args.no_preview:
            from .visuals import Visuals
            visuals=Visuals(fly,out,record=args.record_visuals,preview=not args.no_preview,activity_mode=args.activity_mode,voltage_smoothing_ms=args.voltage_smoothing_ms)
        if args.record_activity:
            from .activity import ActivityWriter
            activity_writer=ActivityWriter(out,fly.brain.ids)
        if args.record:
            import imageio_ffmpeg
            writer=imageio_ffmpeg.write_frames(str(out/'baseline.mp4'),(1920,1080),fps=60,macro_block_size=1,codec='libx264',pix_fmt_in='rgb24',pix_fmt_out='yuv420p',output_params=['-crf','18','-preset','veryfast'])
            writer.send(None)
        while not args.steps or steps<args.steps:
            tick_start=time.perf_counter()
            rgb=pixels(obs,c)
            preview_rgb=preview_pixels(obs)
            control,neural=fly.step(rgb,c['neural_ms_per_tick'])
            if activity_writer: activity_writer.write(fly.activity_bins)
            action=map_action(control,c)
            before=telemetry(obs)
            next_obs,_,_,_,_=env.step(action)
            after=telemetry(next_obs)
            if trajectory_writer: trajectory_writer.write(steps,before,after,action)
            row={'frame':steps,'game_elapsed_ms':(steps+1)*50,'learning':False,'input_rgb_sha256':hashlib.sha256(rgb.tobytes()).hexdigest(),'controller':control,'action':action,'before':before,'after':after,**neural}
            if steps%10000==0:
                if log: log.close()
                log=gzip.open(out/f'steps-{steps//10000:05d}.jsonl.gz','wt',encoding='utf-8')
            log.write(json.dumps(row)+'\n')
            if writer:
                from .visuals import fit
                output_frame=np.asarray(fit(Image.fromarray(preview_rgb)))
                for _ in range(3):writer.send(output_frame)
            if args.debug and steps%c['debug_every']==0:
                Image.fromarray(rgb).save(out/f'rgb-{steps:07d}.png')
                np.save(out/f'retina-{steps:07d}.npy',fly.samples)
                view=Image.new('RGB',(640,480));draw=ImageDraw.Draw(view)
                for uv,v in zip(fly.brain.uv,fly.samples):
                    x,y=uv*[639,479];b=int(v*255)
                    draw.ellipse((x-1,y-1,x+1,y+1),fill=(b,b,b))
                view.save(out/f'retina-{steps:07d}.png')
                (out/f'debug-{steps:07d}.json').write_text(json.dumps(row,indent=2))
            steps+=1;obs=next_obs
            if visuals and (args.record_visuals or not args.no_preview): visuals.update(rgb,control,steps-1,preview_rgb=preview_rgb)
            if steps%20==0:
                log.flush()
                print(f'LEARNING: OFF | tick={steps} brain={fly.brain.sim_ms/1000:.2f}s yaw={action["camera_yaw"]:.3f} forward={action["forward"]} attack={action["attack"]}',flush=True)
            if steps%1000==0: fly.verify_frozen()
            time.sleep(max(0,.05-(time.perf_counter()-tick_start)))
        summary['complete']=True
    except KeyboardInterrupt:
        summary['stopped_by_user']=True
        print('Stopping; checking frozen weights and finalizing recording...',flush=True)
    finally:
        if log: log.close()
        if activity_writer: activity_writer.close()
        if trajectory_writer: trajectory_writer.close()
        if writer: writer.close()
        if visuals: visuals.close()
        try:
            if env: env.close()
        finally:
            fly.verify_frozen()
            summary.update(steps=steps,neural_ms=fly.brain.sim_ms,wall_seconds=time.perf_counter()-start,weight_sha256_after=fly.weight_hash())
            if 'obs' in locals():
                summary['final_game']=telemetry(obs)
                Image.fromarray(pixels(obs,c)).save(out/'last-frame.png')
                Image.fromarray(preview_pixels(obs)).save(out/'last-frame-preview.png')
            (out/'summary.json').write_text(json.dumps(summary,indent=2))
            print(f'Saved {out}',flush=True)
if __name__=='__main__':main()

