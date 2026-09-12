"""Existing read-only dashboard: anatomical activity, retina, live view and playback."""
import gzip
import json
import shutil
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from . import anatomy

BG=(12,17,27)
FG=(229,237,247)
CYAN=(56,218,221)
HD=(1920,1080)

def font(size=18):
    for p in ('C:/Windows/Fonts/consola.ttf','C:/Windows/Fonts/arial.ttf'):
        if Path(p).exists(): return ImageFont.truetype(p,size)
    return ImageFont.load_default()

def fit(im,size=HD):
    result=Image.new('RGB',size,BG)
    scale=min(size[0]/im.width,size[1]/im.height)
    resized=im.resize((round(im.width*scale),round(im.height*scale)),Image.Resampling.LANCZOS)
    result.paste(resized,((size[0]-resized.width)//2,(size[1]-resized.height)//2))
    return result

class Visuals:
    def __init__(self,fly,out,record=False,preview=True,layout=None,playback=False,activity_mode='spikes',voltage_smoothing_ms=80.0):
        self.fly=fly;self.out=Path(out);self.record=record;self.preview=preview;self.playing_back=playback
        self.activity_mode=activity_mode
        self.voltage_smoothing_ms=max(0.0,float(voltage_smoothing_ms))
        self.display_voltage=None
        self.out.mkdir(parents=True,exist_ok=True)
        self.groups=np.unique(fly.brain.superclass)
        self.group_indices=[np.flatnonzero(fly.brain.superclass==g) for g in self.groups]
        self.trail=np.zeros(fly.brain.n,dtype=np.float32)
        self.writers={};self.mode=3 if playback else 4 if record else 1
        self.pg=None;self.screen=None;self.paused=False
        # Default view is rolled 180 degrees from the source orientation so the
        # central brain sits above the VNC and reads more naturally on screen.
        self.yaw=.25;self.pitch=-.7;self.roll=np.pi;self.zoom=1.0;self.scene_cache={}
        geometry=layout if layout is not None else anatomy.load()
        assert np.array_equal(geometry['ids'],fly.brain.ids),'Anatomy IDs must match graph IDs exactly'
        self.xyz=np.asarray(geometry['xyz_nm'],dtype=np.float64)
        self.located=np.isfinite(self.xyz).all(axis=1)
        self.position_source=np.asarray(geometry['position_source'] if 'position_source' in geometry else geometry['source'])
        finite=self.xyz[self.located]
        self.center=(finite.min(axis=0)+finite.max(axis=0))/2
        self.extent=np.ptp(finite,axis=0).max()
        self.skeletons={}
        skeleton_folder=Path(layout.filename).parent if layout is not None and hasattr(layout,'filename') else self.out if playback else anatomy.CACHE
        for p in skeleton_folder.glob('skeleton-*.npz'):
            with np.load(p) as a:self.skeletons[int(p.stem.split('-')[1])]=(a['xyz_nm'],a['edges'])
        if preview:
            import pygame
            self.pg=pygame;pygame.display.init()
            self.screen=pygame.display.set_mode(HD,pygame.RESIZABLE)
            pygame.display.set_caption('DOOMFLY Live | 1 Game 2 Retina 3 Neurons 4 All | arrows orbit Q/E roll +/- zoom | Esc Stop')
        if record or not playback:
            np.savez_compressed(self.out/'neuron-layout.npz',ids=fly.brain.ids,superclass=fly.brain.superclass,xyz_nm=self.xyz,position_source=self.position_source,retina_indices=fly.brain.retina,retina_uv=fly.brain.uv)
            if (anatomy.CACHE/'anatomy.json').exists():shutil.copy2(anatomy.CACHE/'anatomy.json',self.out/'anatomy.json')
            for p in anatomy.CACHE.glob('skeleton-*.npz'):shutil.copy2(p,self.out/p.name)
            (self.out/'visuals.json').write_text(json.dumps({'fps':60 if record else 20,'resolution':list(HD),'spike_bin_hz':60,'simulation_dt_ms':.1,'native_steps_per_bin_pattern':[167,167,166],'trail_decay_ms':250,'layout':'actual MaleCNS EM anatomical XYZ in nm; soma, tosoma or actual skeleton vertex','positioned_neurons':int(self.located.sum()),'missing_positions':int((~self.located).sum()),'retina':'actual upstream linear-luminance samples; inferred eye projection','spike_data':'activity-60hz.jsonl.gz; source MaleCNS body IDs; frame starts at t=frame/60; interval [t,t+1/60)','video_sampling':'game and retina remain 20 Hz; frames held for three 60 Hz output frames; no new visual input to brain','activity_mode':self.activity_mode,'voltage_display_smoothing_ms':self.voltage_smoothing_ms,'voltage_display_note':'display-only exponential smoothing of membrane voltage; simulation state is unchanged','voltage_recording':'when -RecordActivity is used, voltage-20hz/ stores quantized membrane snapshots at the existing 50 ms control boundary','learning':False},indent=2))

    def project(self,xyz,size):
        p=(np.asarray(xyz)-self.center)/self.extent
        cy,sy=np.cos(self.yaw),np.sin(self.yaw);cp,sp=np.cos(self.pitch),np.sin(self.pitch)
        rotation=np.array([[cy,0,sy],[sp*sy,cp,-sp*cy],[-cp*sy,sp,cp*cy]])
        p=p@rotation.T
        # Camera roll around the viewing axis. A default pi roll puts the VNC down.
        cr,sr=np.cos(self.roll),np.sin(self.roll)
        rolled=np.empty_like(p)
        rolled[:,0]=p[:,0]*cr-p[:,1]*sr
        rolled[:,1]=p[:,0]*sr+p[:,1]*cr
        rolled[:,2]=p[:,2]
        p=rolled
        scale=min(size)*.91*self.zoom
        return np.column_stack((size[0]/2+p[:,0]*scale,size[1]/2-p[:,1]*scale,p[:,2]))

    def video(self,name,im):
        assert im.size==HD
        if name not in self.writers:
            import imageio_ffmpeg
            w=imageio_ffmpeg.write_frames(str(self.out/name),HD,fps=60,macro_block_size=1,codec='libx264',pix_fmt_in='rgb24',pix_fmt_out='yuv420p',output_params=['-crf','18','-preset','veryfast'])
            w.send(None);self.writers[name]=w
        self.writers[name].send(np.asarray(im))

    def retina_image(self):
        retina=Image.new('RGB',(640,480),BG);d=ImageDraw.Draw(retina)
        for uv,v in zip(self.fly.brain.uv,self.fly.samples):
            x,y=uv*[639,407]+[0,48];b=int(np.clip(v,0,1)*255)
            d.ellipse((x-1.2,y-1.2,x+1.2,y+1.2),fill=(b,b,b))
        d.text((16,10),'RETINA | 3,335 luminance samples',font=font(20),fill=FG)
        d.text((16,458),'Actual input samples; inferred eye projection',font=font(14),fill=CYAN)
        return retina

    def scene(self,map_width,map_height):
        key=(map_width,map_height,self.yaw,self.pitch,self.roll,self.zoom)
        if key in self.scene_cache:return self.scene_cache[key]
        projected=self.project(self.xyz,(map_width,map_height))
        good=self.located.copy()
        good[good]&=(projected[good,0]>=0)&(projected[good,0]<map_width)&(projected[good,1]>=0)&(projected[good,1]<map_height)
        indices=np.flatnonzero(good)
        indices=indices[np.argsort(projected[indices,2])]
        points=projected[indices,:2].astype(int)
        canvas=np.zeros((map_height,map_width,3),dtype=np.uint8);canvas[:]=BG
        canvas[points[:,1],points[:,0]]=(35,51,68)
        base=Image.fromarray(canvas);draw=ImageDraw.Draw(base)
        for body,(vertices,edges) in self.skeletons.items():
            proj=self.project(vertices,(map_width,map_height))
            for a,b in edges:
                if a>=len(proj) or b>=len(proj):continue
                u,w=proj[a],proj[b]
                if min(u[0],w[0],u[1],w[1])<0 or max(u[0],w[0])>=map_width or max(u[1],w[1])>=map_height:continue
                draw.line((u[0],u[1],w[0],w[1]),fill=(29,90,95),width=1)
        if len(self.scene_cache)>4:self.scene_cache.clear()
        self.scene_cache[key]=(projected,indices,np.asarray(base).copy())
        return self.scene_cache[key]


    def update_display_voltage(self,duration):
        """Low-pass the displayed membrane voltage only; never alter brain state."""
        if self.activity_mode not in ('voltage','combined'):
            return
        voltage=getattr(self.fly,'last_voltage',None)
        if voltage is None:
            return
        target=np.asarray(voltage,dtype=np.float32)
        if self.display_voltage is None or self.display_voltage.shape!=target.shape:
            self.display_voltage=target.copy()
            return
        if self.voltage_smoothing_ms<=0:
            self.display_voltage[...] = target
            return
        tau=self.voltage_smoothing_ms/1000.0
        alpha=1.0-np.exp(-float(duration)/tau)
        self.display_voltage += np.float32(alpha)*(target-self.display_voltage)

    def activity_image(self,counts,control,t,duration,size=HD,clean=False):
        width,height=size;map_width=width if clean else width-370;map_height=height-90
        im=Image.new('RGB',size,BG)
        projected,indices,base=self.scene(map_width,map_height)
        canvas=base.copy()

        # Optional membrane-potential layer. DOOMFLY uses rest=-52 mV and
        # threshold=-45 mV. Cyan shows depolarization toward threshold; purple
        # shows hyperpolarization below rest. This is a read-only view of v.
        voltage=self.display_voltage
        if self.activity_mode in ('voltage','combined') and voltage is not None:
            vv=np.asarray(voltage)
            dep=np.clip((vv+52.0)/7.0,0,1)
            hyp=np.clip((-52.0-vv)/7.0,0,1)
            strength=np.maximum(dep,hyp)
            shown=indices[strength[indices]>0.025]
            shown=shown[np.argsort(strength[shown])]
            pp=projected[shown,:2].astype(int)
            # Mild gamma compression makes low-level subthreshold activity visible.
            dv=np.sqrt(dep[shown]);hv=np.sqrt(hyp[shown])
            colors=np.column_stack((35+hv*120,55+dv*180,85+dv*150+hv*80)).clip(0,255).astype(np.uint8)
            for dx,dy in ((0,0),(1,0),(0,1)):
                x=np.clip(pp[:,0]+dx,0,map_width-1);y=np.clip(pp[:,1]+dy,0,map_height-1)
                canvas[y,x]=colors

        if self.activity_mode in ('spikes','combined'):
            active=indices[self.trail[indices]>0.02]
            # Draw dim trails first and brighter measured activity last at overlaps.
            active=active[np.argsort(self.trail[active])]
            pp=projected[active,:2].astype(int)
            v=np.clip(np.log1p(self.trail[active])/np.log1p(6),0,1)
            colors=np.column_stack((v*255,90+v*150,20+v*50)).astype(np.uint8)
            for dx,dy in ((0,0),(1,0),(0,1)):
                x=np.clip(pp[:,0]+dx,0,map_width-1);y=np.clip(pp[:,1]+dy,0,map_height-1)
                canvas[y,x]=colors
        if clean and self.zoom==1.0:
            # Fit the anatomical silhouette, not a square around the EM volume.
            # Uniform scaling preserves anatomy and fills the available viewport
            # as far as its aspect ratio permits; no reserved text sidebar.
            occupied=np.any(canvas!=np.asarray(BG,dtype=np.uint8),axis=2)
            rows=np.flatnonzero(occupied.any(axis=1));cols=np.flatnonzero(occupied.any(axis=0))
            if len(rows) and len(cols):
                crop=Image.fromarray(canvas).crop((int(cols[0]),int(rows[0]),int(cols[-1])+1,int(rows[-1])+1))
                fitted=fit(crop,(map_width-24,map_height-24))
                canvas=np.asarray(Image.new('RGB',(map_width,map_height),BG)).copy()
                canvas[12:map_height-12,12:map_width-12]=np.asarray(fitted)
        im.paste(Image.fromarray(canvas),(0,55));d=ImageDraw.Draw(im)
        if clean:
            d.text((28,10),'NEURAL ACTIVITY',font=font(24),fill=FG)
            status='EXPERIMENTAL LEARNING' if getattr(self.fly,'learning',False) else 'FROZEN / REPLAY'
            d.text((28,39),status,font=font(14),fill=CYAN)
            d.text((width-300,15),f'{np.count_nonzero(counts):,} firing this frame',font=font(18),fill=CYAN)
            d.text((28,height-27),'MaleCNS anatomy   /   gold = spikes   /   250 ms visual trail',font=font(16),fill=FG)
            return im
        learning='AVERSIVE LTD (EXPERIMENTAL)' if getattr(self.fly,'learning',False) else 'OFF'
        d.text((18,12),f'MALE CNS ANATOMY | {t:.3f} s | LEARNING: {learning}',font=font(24),fill=FG)
        legend={
            'spikes':'gold: spikes, 250 ms trail',
            'voltage':f'cyan/purple: membrane voltage ({self.voltage_smoothing_ms:g} ms display smoothing)',
            'combined':f'cyan/purple: smoothed membrane voltage ({self.voltage_smoothing_ms:g} ms) | gold: spikes'
        }[self.activity_mode]
        d.text((14,height-27),f'Published anatomical XYZ (nm) | {legend} | arrows: orbit  Q/E: roll  +/-: zoom',font=font(18),fill=FG)
        x=width-350;y=60
        d.text((x,y),f'{np.count_nonzero(counts):,} active',font=font(24),fill=CYAN);y+=34
        d.text((x,y),f'{duration*1000:.3f} ms observation bin',font=font(18),fill=FG);y+=28
        unlocated=int(np.count_nonzero((counts>0)&~self.located))
        d.text((x,y),f'{unlocated} active without position',font=font(17),fill=FG);y+=37
        d.text((x,y),'Descending neurons',font=font(21),fill=FG);y+=30
        for r in control['readouts']:
            if r['type'] not in ('DNp20','DNpe017'):continue
            d.text((x,y),f'{r["type"]} {r["side"]}: {r["rate_hz"]:.1f} Hz',font=font(19),fill=FG)
            d.rectangle((x,y+25,x+min(310,r['rate_hz']*3),y+30),fill=CYAN);y+=45
        if height>700:
            d.text((x,y+10),'Population mean Hz',font=font(20),fill=FG);y+=45
            ranked=sorted([(float(counts[ix].mean()/duration),str(g)) for g,ix in zip(self.groups,self.group_indices)],reverse=True)[:8]
            for rate,g in ranked:
                d.text((x,y),g[:25],font=font(18),fill=FG)
                d.text((x,y+22),f'{rate:.2f} Hz / neuron',font=font(17),fill=CYAN);y+=48
        return im

    def events(self):
        if not self.pg:return
        for e in self.pg.event.get():
            if e.type==self.pg.QUIT or (e.type==self.pg.KEYDOWN and e.key==self.pg.K_ESCAPE):raise KeyboardInterrupt
            if e.type!=self.pg.KEYDOWN:continue
            if e.unicode in ('1','2','3','4'):self.mode=int(e.unicode)
            if e.key==self.pg.K_LEFT:self.yaw-=.12
            if e.key==self.pg.K_RIGHT:self.yaw+=.12
            if e.key==self.pg.K_UP:self.pitch+=.12
            if e.key==self.pg.K_DOWN:self.pitch-=.12
            if e.key==self.pg.K_q:self.roll-=.12
            if e.key==self.pg.K_e:self.roll+=.12
            if e.unicode in ('+','='):self.zoom=min(8,self.zoom*1.15)
            if e.unicode=='-':self.zoom=max(.2,self.zoom/1.15)
            if e.key==self.pg.K_SPACE:self.paused=not self.paused

    def show(self,rgb,retina,activity,dashboard):
        self.events()
        if not self.pg:return
        im={1:fit(Image.fromarray(rgb)),2:fit(retina),3:activity,4:dashboard}[self.mode]
        surf=self.pg.image.frombuffer(im.tobytes(),im.size,'RGB')
        w,h=self.screen.get_size();scale=min(w/im.width,h/im.height)
        size=(int(im.width*scale),int(im.height*scale));self.screen.fill(BG)
        self.screen.blit(self.pg.transform.smoothscale(surf,size),((w-size[0])//2,(h-size[1])//2));self.pg.display.flip()

    def render_frame(self,rgb,retina,counts,control,frame,t,duration,write=False):
        self.trail=np.maximum(counts,self.trail*np.exp(-duration/.25))
        self.update_display_voltage(duration)
        activity=self.activity_image(counts,control,t,duration)
        dashboard=Image.new('RGB',HD,BG)
        if getattr(self.fly,'training',None):
            # Keep the original 50/50 dashboard columns. Minecraft RGB is 4:3
            # (640x480), while its dashboard frame is 960x540 (16:9), so fit()
            # letterboxes/pillarboxes the image inside that unchanged frame.
            # This preserves the game's true aspect ratio without stretching or
            # cropping and without changing the retina/brain panel widths.
            dashboard.paste(fit(Image.fromarray(rgb),(960,540)),(0,0))
            dashboard.paste(fit(retina,(960,432)),(960,0))
            dashboard.paste(self.activity_image(counts,control,t,duration,(960,648),clean=True),(960,432))
            panel_canvas=Image.new('RGB',HD,BG)
            self.training_panel(panel_canvas,self.fly.training)
            dashboard.paste(fit(panel_canvas.crop((0,450,800,1080)),(960,540)),(0,540))
            borders=ImageDraw.Draw(dashboard)
            borders.line((960,0,960,1080),fill=(40,53,69),width=2)
            borders.line((0,540,960,540),fill=(40,53,69),width=2)
            borders.line((960,432,1920,432),fill=(40,53,69),width=2)
        else:
            dashboard.paste(fit(Image.fromarray(rgb),(960,540)),(0,0));dashboard.paste(fit(retina,(960,540)),(960,0))
            dashboard.paste(self.activity_image(counts,control,t,duration,(1920,540)),(0,540))
        if write:
            for name,im in [('retina.mp4',fit(retina)),('neural-activity.mp4',activity),('dashboard.mp4',dashboard)]:self.video(name,im)
        if frame==0 or frame%300==0:
            fit(retina).save(self.out/'retina-preview.png');activity.save(self.out/'activity-preview.png');dashboard.save(self.out/'dashboard-preview.png')
        self.show(rgb,retina,activity,dashboard)
        return activity

    def training_panel(self,im,p):
        d=ImageDraw.Draw(im);x=22;y=785
        d.rectangle((0,450,799,1079),fill=BG)
        best='--' if p['best_eval'] is None else f'{p["best_eval"]:.0%}'
        d.text((22,458),f'EP {p["episode"]}   /   {p["step"]:,} of {p["total"]:,} steps',font=font(20),fill=FG)
        d.text((490,458),f'REWARD {p["reward"]:+.2f}',font=font(24),fill=(245,195,78))
        d.text((22,492),'NEURONS',font=font(16),fill=FG)
        d.text((264,492),'FIXED DECODER',font=font(16),fill=FG)
        d.text((586,492),'GAME INPUT',font=font(16),fill=FG)
        m=p.get('motor')
        if m:
            def meter(xx,yy,value,color,width=145,maximum=100):
                d.rounded_rectangle((xx,yy,xx+width,yy+7),radius=3,fill=(37,49,65))
                fill=int(width*np.clip(value/maximum,0,1))
                if fill:d.rounded_rectangle((xx,yy,xx+fill,yy+7),radius=3,fill=color)
            def path(yy,enabled,color):
                col=color if enabled else (54,65,80)
                d.line((228,yy,555,yy),fill=col,width=3)
                d.polygon([(555,yy),(543,yy-6),(543,yy+6)],fill=col)
            cyan=CYAN;green=(91,225,165);gold=(245,195,78)
            # Node labels are measured neural channels; these lines represent
            # the engineered decoder, not reconstructed synaptic connections.
            for top,color in ((520,cyan),(596,green),(672,gold)):
                d.rounded_rectangle((18,top,780,top+70),radius=12,outline=(41,54,72),width=1)
            d.text((30,524),'DNp20 L / R  Hz',font=font(18),fill=cyan)
            meter(30,552,m['left_hz'],cyan,maximum=100)
            meter(30,573,m['right_hz'],cyan,maximum=100)
            d.text((181,544),f'{m["left_hz"]:.0f}',font=font(14),fill=FG)
            d.text((181,566),f'{m["right_hz"]:.0f}',font=font(14),fill=FG)
            path(559,abs(m['yaw'])>0,cyan)
            d.text((261,524),'rate difference -> camera',font=font(16),fill=FG)
            d.text((274,566),f'x0.12  then  x{m["yaw_gain"]:g}',font=font(14),fill=FG)
            direction='LEFT' if m['yaw']<0 else 'RIGHT' if m['yaw']>0 else 'STILL'
            d.text((576,526),direction,font=font(23),fill=cyan)
            d.text((576,558),f'{abs(m["yaw"]):.2f} deg/tick',font=font(18),fill=FG)
            d.text((30,602),'DNpe017 L+R Hz',font=font(18),fill=green)
            meter(30,636,m['forward_hz'],green)
            d.text((181,626),f'{m["forward_hz"]:.0f}',font=font(14),fill=FG)
            path(635,m['walking'],green)
            d.text((264,601),'rate -> forward threshold',font=font(16),fill=FG)
            d.text((272,643),f'{m["forward"]:.1f} > {m["forward_threshold"]:g}',font=font(14),fill=FG)
            d.rounded_rectangle((579,608,627,655),radius=7,fill=green if m['walking'] else (37,49,65))
            d.text((593,616),'W',font=font(26),fill=BG if m['walking'] else FG)
            d.text((639,620),'HELD' if m['walking'] else 'OFF',font=font(22),fill=green if m['walking'] else FG)
            d.text((30,678),'DNpe017 spikes',font=font(18),fill=gold)
            for j in range(min(12,m['spikes'])):
                d.line((33+j*14,730,33+j*14,710),fill=gold,width=3)
            d.text((181,711),str(m['spikes']),font=font(16),fill=FG)
            path(711,m['attacking'],gold)
            d.text((264,677),'pulse -> accumulate -> hold',font=font(16),fill=FG)
            meter(270,730,m['accumulator'],gold,width=130,maximum=m['threshold'])
            d.text((413,720),f'{m["hold_left"]} ticks',font=font(14),fill=FG)
            d.rounded_rectangle((582,681,620,734),radius=13,outline=gold,width=2)
            if m['attacking']:d.rectangle((587,685,600,706),fill=gold)
            d.line((601,683,601,708),fill=gold,width=1)
            d.text((638,697),'HELD' if m['attacking'] else 'OFF',font=font(22),fill=gold if m['attacking'] else FG)
        d.text((22,754),f'{p["distance"]:.1f}m to log  |  aim {np.degrees(p["angle"]):.0f} deg  |  break {p["progress"]:.0%}',font=font(17),fill=FG)
        d.text((22,1049),f'Rolling {p["rolling"]:+.2f}   Mean {p["mean"]:+.2f}   Best eval {best}',font=font(17),fill=CYAN)
        # Draw every control tick, including the first episode; 60 Hz video
        # repeats do not create additional reward samples.
        key=p['episode']
        if getattr(self,'reward_episode',None)!=key:
            self.reward_episode=key;self.reward_trace=[]
        tick=p.get('episode_step',p['step'])
        if not self.reward_trace:self.reward_trace.append((max(0,tick-1),0.))
        if self.reward_trace[-1][0]!=tick:self.reward_trace.append((tick,p['reward']))
        else:self.reward_trace[-1]=(tick,p['reward'])
        def graph(title,ys,xs,top,height=40,xlabel='step',style='line',bounds=None,empty='Waiting for data'):
            left=x+60;right=770;bottom=top+height
            d.text((x,top-20),title,font=font(15),fill=FG)
            if not len(ys):
                d.text((left,top+12),empty,font=font(14),fill=(160,174,190));return
            a=np.asarray(ys,dtype=float);xx=np.asarray(xs,dtype=float)
            if bounds is None:
                low=min(0.,float(a.min()));high=max(0.,float(a.max()))
                pad=max(.01,(high-low)*.1);low-=pad;high+=pad
            else:
                low,high=bounds
            if high<=low:high=low+.02
            for value in (low,0.,high):
                yy=bottom-(value-low)/(high-low)*height
                d.line((left,yy,right,yy),fill=(42,55,72))
                d.text((x,yy-7),f'{value:.2f}',font=font(12),fill=FG)
            d.line((left,top,left,bottom),fill=FG)
            span=max(1.,float(xx[-1]-xx[0]))
            # A 100k-step run only has hundreds of episodes, so keep every
            # completed-episode point. Long live traces are thinned to screen width.
            if style=='points':
                indices=np.arange(len(a))
            else:
                indices=np.unique(np.linspace(0,len(a)-1,min(len(a),700)).astype(int))
            pts=[(left+(xx[i]-xx[0])/span*(right-left),bottom-(a[i]-low)/(high-low)*height) for i in indices]
            if style=='line' and len(pts)>1:d.line(pts,fill=CYAN if 'ROLLING' in title else (245,195,78),width=2)
            if style=='points':
                # Completed-episode rewards are discrete samples, but joining
                # them makes the run's trajectory much easier to read at a glance.
                if len(pts)>1:d.line(pts,fill=(245,195,78),width=2)
                for px,py in pts:d.ellipse((px-2,py-2,px+2,py+2),fill=(245,195,78))
            elif pts:
                px,py=pts[-1];d.ellipse((px-3,py-3,px+3,py+3),fill=CYAN if 'ROLLING' in title else (245,195,78))
            d.text((left,bottom+2),str(int(xx[0])),font=font(12),fill=FG)
            d.text((right-145,bottom+2),f'{xlabel} {int(xx[-1])}',font=font(12),fill=FG)

        # Three distinct training views:
        # 1) cumulative reward evolving live within the current episode,
        # 2) one point per completed episode,
        # 3) the smoothed rolling average across completed episodes.
        graph('CURRENT EPISODE | CUMULATIVE REWARD',
            [v for t,v in self.reward_trace],[t for t,v in self.reward_trace],y+28,height=40)
        values=np.asarray(p['history'],dtype=float)
        win=p['rolling_window'];ix=np.arange(len(values));sums=np.r_[0,np.cumsum(values)]
        smooth=(sums[ix+1]-sums[np.maximum(0,ix+1-win)])/np.minimum(ix+1,win)
        episode_bounds=None
        if len(values):
            low=min(0.,float(values.min()),float(smooth.min()));high=max(0.,float(values.max()),float(smooth.max()))
            pad=max(.01,(high-low)*.1);episode_bounds=(low-pad,high+pad)
        graph('COMPLETED EPISODE REWARD',values,ix+1,y+105,height=40,xlabel='episode',style='points',bounds=episode_bounds,empty='No completed episodes yet')
        graph(f'ROLLING AVERAGE | LAST {win} EPISODES',smooth,ix+1,y+182,height=40,xlabel='episode',style='line',bounds=episode_bounds,empty='No completed episodes yet')

    def update(self,rgb,control,frame):
        retina=self.retina_image()
        bins=getattr(self.fly,'activity_bins',None)
        if bins is None:
            self.render_frame(rgb,retina,self.fly.last_counts,control,frame,(frame+1)/20,.05)
        else:
            for k,counts in enumerate(bins):
                self.render_frame(rgb,retina,counts,control,frame*3+k,(frame*3+k)/60,1/60,write=self.record)

    def close(self):
        for w in self.writers.values():w.close()
        if self.pg:self.pg.display.quit()

    @classmethod
    def playback(cls,folder,preview=True,limit=0,activity_mode='spikes',voltage_smoothing_ms=80.0):
        """Load sparse source IDs into the SAME renderer; no brain/game simulation."""
        folder=Path(folder);layout=np.load(folder/'neuron-layout.npz')
        brain=SimpleNamespace(ids=layout['ids'],n=len(layout['ids']),superclass=layout['superclass'],retina=layout['retina_indices'],uv=layout['retina_uv'])
        fly=SimpleNamespace(brain=brain,samples=np.zeros(len(brain.retina)))
        output=folder/'playback-proof';output.mkdir(exist_ok=True)
        voltage=None
        if (folder/'voltage-20hz'/'metadata.json').exists():
            from .activity import VoltageReader
            voltage=VoltageReader(folder/'voltage-20hz')
        elif activity_mode!='spikes':
            print('This older recording contains spikes only; falling back to spike view.')
            activity_mode='spikes'
        view=cls(fly,output,preview=preview,layout=layout,playback=True,activity_mode=activity_mode,voltage_smoothing_ms=voltage_smoothing_ms)
        # Anatomy and optional skeletons belong to the recording, not a new layout.
        view.skeletons={}
        for p in folder.glob('skeleton-*.npz'):
            with np.load(p) as a:view.skeletons[int(p.stem.split('-')[1])]=(a['xyz_nm'],a['edges'])
        index={int(body):i for i,body in enumerate(brain.ids)}
        controls={};training={}
        history=json.loads((folder/'reward-history.json').read_text()) if (folder/'reward-history.json').exists() else []
        for p in sorted(folder.glob('steps-*.jsonl.gz')):
            with gzip.open(p,'rt') as f:
                for line in f:
                    r=json.loads(line);controls[r['frame']]=r['controller']
                    if 'training' in r:training[r['frame']]=r['training']
        streams={}
        import imageio_ffmpeg
        for name in ('baseline','retina'):
            p=folder/(name+'.mp4')
            if p.exists():
                reader=imageio_ffmpeg.read_frames(str(p),pix_fmt='rgb24');meta=next(reader);streams[name]=(reader,meta)
        unknown=set();unlocated=set();frames=0;rgb=np.zeros((480,640,3),dtype=np.uint8);retina=Image.new('RGB',(640,480),BG)
        try:
            with gzip.open(folder/'activity-60hz.jsonl.gz','rt') as source:
                for line in source:
                    clock=time.perf_counter();row=json.loads(line);counts=np.zeros(brain.n,dtype=np.int32)
                    for body,count in row['active']:
                        if body not in index:unknown.add(body);continue
                        i=index[body];counts[i]=count
                        if not view.located[i]:unlocated.add(body)
                    for name,(reader,meta) in streams.items():
                        raw=next(reader);w,h=meta['size'];arr=np.frombuffer(raw,np.uint8).reshape(h,w,3)
                        if name=='baseline':rgb=arr
                        else:retina=Image.fromarray(arr)
                    control=controls.get(row['frame']//3,{'readouts':[]})
                    if row['frame']//3 in training:
                        fly.training={**training[row['frame']//3],'history':[r['reward'] for r in history if not r.get('censored',False)]}
                    if voltage is not None:
                        fly.last_voltage=voltage.get(row['frame']//3)
                    im=view.render_frame(rgb,retina,counts,control,row['frame'],row['t'],1/60)
                    if frames==0:
                        active=np.flatnonzero((counts>0)&view.located)[:20]
                        coords=view.project(view.xyz[active],(1550,990))
                        (output/'active-position-check.json').write_text(json.dumps([{'id':int(brain.ids[i]),'xyz_nm':view.xyz[i].tolist(),'screen_xy':[float(xy[0]),float(xy[1]+55)]} for i,xy in zip(active,coords)],indent=2))
                    frames+=1
                    while view.pg and view.paused:view.events();time.sleep(.03)
                    if limit and frames>=limit:break
                    if preview:time.sleep(max(0,1/60-(time.perf_counter()-clock)))
        except KeyboardInterrupt:pass
        finally:
            view.close()
            for reader,meta in streams.values():reader.close()
            (output/'report.json').write_text(json.dumps({'frames_loaded':frames,'unknown_ids':sorted(unknown),'active_ids_without_position':sorted(unlocated),'renderer':'existing flycraft.visuals.Visuals','resolution':list(HD)},indent=2))
        if unknown:raise RuntimeError('Recording has unknown neuron IDs')
        return output
