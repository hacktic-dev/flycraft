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
POS=(91,225,165)
NEG=(245,105,100)
AVERSIVE=(190,125,255)
MUTED=(118,132,150)
HD=(1920,1080)

def font(size=18):
    for p in ('C:/Windows/Fonts/consola.ttf','C:/Windows/Fonts/arial.ttf'):
        if Path(p).exists(): return ImageFont.truetype(p,size)
    return ImageFont.load_default()

def fit(im,size=HD,bg=BG):
    result=Image.new('RGB',size,bg)
    scale=min(size[0]/im.width,size[1]/im.height)
    resized=im.resize((round(im.width*scale),round(im.height*scale)),Image.Resampling.LANCZOS)
    result.paste(resized,((size[0]-resized.width)//2,(size[1]-resized.height)//2))
    return result

def fill_crop(im,size):
    """Fill *size* without distortion by center-cropping the source aspect ratio."""
    target_ratio=size[0]/size[1]
    source_ratio=im.width/im.height
    if source_ratio>target_ratio:
        crop_w=round(im.height*target_ratio)
        x=(im.width-crop_w)//2
        im=im.crop((x,0,x+crop_w,im.height))
    elif source_ratio<target_ratio:
        crop_h=round(im.width/target_ratio)
        y=(im.height-crop_h)//2
        im=im.crop((0,y,im.width,y+crop_h))
    return im.resize(size,Image.Resampling.LANCZOS)

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
            (self.out/'visuals.json').write_text(json.dumps({'fps':60 if record else 20,'resolution':list(HD),'spike_bin_hz':60,'simulation_dt_ms':fly.brain.dt,'native_steps_per_bin_pattern':[sum(int((tick*fly.brain.dt*60/1000))%3==b for tick in range(round(50/fly.brain.dt))) for b in range(3)],'trail_decay_ms':250,'layout':'actual MaleCNS EM anatomical XYZ in nm; soma, tosoma or actual skeleton vertex','positioned_neurons':int(self.located.sum()),'missing_positions':int((~self.located).sum()),'retina':'R1-R6 linear-luminance projection; learning mode also drives DOOMFLY-v6 inferred R8p blue/R8y green inputs','spike_data':'activity-60hz.jsonl.gz; source MaleCNS body IDs; frame starts at t=frame/60; interval [t,t+1/60)','video_sampling':'game and retina remain 20 Hz; frames held for three 60 Hz output frames; no new visual input to brain','activity_mode':self.activity_mode,'voltage_display_smoothing_ms':self.voltage_smoothing_ms,'voltage_display_note':'display-only exponential smoothing of membrane voltage; simulation state is unchanged','voltage_recording':'when -RecordActivity is used, voltage-20hz/ stores quantized membrane snapshots at the existing 50 ms control boundary','learning':False},indent=2))

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
        # R1-R6: the original luminance samples, unchanged.
        for uv,v in zip(self.fly.brain.uv,self.fly.samples):
            x,y=uv*[639,407]+[0,48];b=int(np.clip(v,0,1)*255)
            d.ellipse((x-1.2,y-1.2,x+1.2,y+1.2),fill=(b,b,b))

        if hasattr(self.fly.brain,'r8'):
            # R8: show the actual v6 low-pass color input at each inferred eye
            # position. R8p samples blue; R8y samples green. A small base
            # brightness keeps receptor identity visible even in dark pixels,
            # while the input value controls the rest of the intensity.
            lights=np.asarray(self.fly.brain.r8_light,dtype=float)
            channels=np.asarray(self.fly.brain.r8_channel)
            for uv,v,ch in zip(self.fly.brain.r8_uv,lights,channels):
                x,y=uv*[639,407]+[0,48]
                level=int(55+200*np.clip(v,0,1))
                color=(35,90,level) if ch==2 else (35,level,90)
                d.ellipse((x-1.8,y-1.8,x+1.8,y+1.8),fill=color)
            d.text((16,10),f'RETINA | 3,335 R1-R6 + {len(self.fly.brain.r8):,} R8 color inputs',font=font(20),fill=FG)
            d.text((16,458),'gray = R1-R6 luminance   blue = R8p   green = R8y',font=font(14),fill=CYAN)
        else:
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
        learning='SIGNED REVERSIBLE (EXPERIMENTAL)' if getattr(self.fly,'learning',False) else 'OFF'
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

    def show(self,rgb,retina,activity,dashboard,preview_rgb=None):
        self.events()
        if not self.pg:return
        game_rgb=rgb if preview_rgb is None else preview_rgb
        im={1:fit(Image.fromarray(game_rgb)),2:fit(retina),3:activity,4:dashboard}[self.mode]
        surf=self.pg.image.frombuffer(im.tobytes(),im.size,'RGB')
        w,h=self.screen.get_size();scale=min(w/im.width,h/im.height)
        size=(int(im.width*scale),int(im.height*scale));self.screen.fill(BG)
        self.screen.blit(self.pg.transform.smoothscale(surf,size),((w-size[0])//2,(h-size[1])//2));self.pg.display.flip()

    def render_frame(self,rgb,retina,counts,control,frame,t,duration,write=False,preview_rgb=None):
        self.trail=np.maximum(counts,self.trail*np.exp(-duration/.25))
        game_rgb=rgb if preview_rgb is None else preview_rgb
        self.update_display_voltage(duration)
        activity=self.activity_image(counts,control,t,duration)
        dashboard=Image.new('RGB',HD,BG)
        if getattr(self.fly,'training',None):
            # Keep the original 50/50 dashboard columns. The Minecraft pane is
            # 960x540 (16:9), but the full sensory frame is preserved: scale it
            # down to fit without cropping or stretching and letterbox/pillarbox
            # the unused area in black. This is display-only; the fly still gets
            # the original sensory frame unchanged.
            dashboard.paste(fit(Image.fromarray(game_rgb),(960,540),(0,0,0)),(0,0))
            dashboard.paste(fit(retina,(960,432)),(960,0))
            dashboard.paste(self.activity_image(counts,control,t,duration,(960,648),clean=True),(960,432))
            panel_canvas=Image.new('RGB',(960,540),BG)
            self.training_panel(panel_canvas,self.fly.training)
            dashboard.paste(panel_canvas,(0,540))
            borders=ImageDraw.Draw(dashboard)
            borders.line((960,0,960,1080),fill=(40,53,69),width=2)
            borders.line((0,540,960,540),fill=(40,53,69),width=2)
            borders.line((960,432,1920,432),fill=(40,53,69),width=2)
        else:
            dashboard.paste(fit(Image.fromarray(game_rgb),(960,540)),(0,0));dashboard.paste(fit(retina,(960,540)),(960,0))
            dashboard.paste(self.activity_image(counts,control,t,duration,(1920,540)),(0,540))
        if write:
            for name,im in [('retina.mp4',fit(retina)),('neural-activity.mp4',activity),('dashboard.mp4',dashboard)]:self.video(name,im)
        if frame==0 or frame%300==0:
            fit(retina).save(self.out/'retina-preview.png');activity.save(self.out/'activity-preview.png');dashboard.save(self.out/'dashboard-preview.png')
        self.show(rgb,retina,activity,dashboard,preview_rgb=game_rgb)
        return activity

    def training_panel(self,im,p):
        """Render actions plus the signals that actually reach plasticity.

        The behavioural reward score is deliberately not presented as a stimulus:
        the fly never receives that scalar. Instead this panel shows the signed
        teaching event applied to KC->MBON11 plasticity and the independent PPL101
        aversive neural pulse when it is actually active.
        """
        d=ImageDraw.Draw(im)
        width,height=im.size
        d.rectangle((0,0,width-1,height-1),fill=BG)

        teaching=p.get('teaching')
        signal=float(teaching.get('signal',0.0)) if teaching else 0.0
        if not getattr(self.fly,'learning',False):
            teach_label='TEACHING OFF / FROZEN';teach_color=MUTED
        elif signal>0:
            teach_label=f'TEACH +{signal:.2f}  POSITIVE';teach_color=POS
        elif signal<0:
            teach_label=f'TEACH {signal:.2f}  NEGATIVE';teach_color=NEG
        else:
            teach_label='TEACH +0.00  NEUTRAL';teach_color=MUTED

        # Header / column labels. No reward scalar is shown here because reward is
        # an external evaluation metric, not something delivered to the fly.
        d.text((22,7),f'EP {p["episode"]}   /   {p["step"]:,} of {p["total"]:,} steps',font=font(18),fill=FG)
        tf=font(18);box=d.textbbox((0,0),teach_label,font=tf)
        d.text((width-22-(box[2]-box[0]),8),teach_label,font=tf,fill=teach_color)
        d.text((22,38),'NEURONS',font=font(14),fill=FG)
        d.text((309,38),'FIXED DECODER',font=font(14),fill=FG)
        d.text((725,38),'GAME INPUT',font=font(14),fill=FG)

        m=p.get('motor')
        if m:
            def meter(xx,yy,value,color,width_px=180,maximum=100):
                d.rounded_rectangle((xx,yy,xx+width_px,yy+6),radius=3,fill=(37,49,65))
                fill_w=int(width_px*np.clip(value/maximum,0,1)) if maximum>0 else 0
                if fill_w:d.rounded_rectangle((xx,yy,xx+fill_w,yy+6),radius=3,fill=color)
            def path(yy,enabled,color):
                col=color if enabled else (54,65,80)
                d.line((302,yy,680,yy),fill=col,width=3)
                d.polygon([(680,yy),(668,yy-6),(668,yy+6)],fill=col)
            cyan=CYAN;green=POS;gold=(245,195,78)
            card_left,card_right=18,width-18
            for top in (58,124,190):
                d.rounded_rectangle((card_left,top,card_right,top+59),radius=11,outline=(41,54,72),width=1)

            d.text((30,62),f'{m.get("steering_type","DNp20")} L / R  Hz',font=font(16),fill=cyan)
            meter(30,87,m['left_hz'],cyan,maximum=100);meter(30,105,m['right_hz'],cyan,maximum=100)
            d.text((220,80),f'{m["left_hz"]:.0f}',font=font(13),fill=FG);d.text((220,99),f'{m["right_hz"]:.0f}',font=font(13),fill=FG)
            path(92,abs(m['yaw'])>0,cyan)
            d.text((326,62),f'L {m["left_normalized"]:.2f}x / R {m["right_normalized"]:.2f}x neutral' if m.get('steering_mode')=='balanced-bilateral-v1' else 'rate difference -> camera',font=font(15),fill=FG)
            d.text((342,99),f'opponent {m["opponent"]:+.2f} -> degrees' if m.get('steering_mode')=='balanced-bilateral-v1' else f'x0.12  then  x{m["yaw_gain"]:g}',font=font(13),fill=FG)
            direction='LEFT' if m['yaw']<0 else 'RIGHT' if m['yaw']>0 else 'STILL'
            d.text((704,64),direction,font=font(20),fill=cyan);d.text((704,94),f'{abs(m["yaw"]):.2f} deg/tick',font=font(16),fill=FG)

            d.text((30,128),'DNpe017 L+R Hz',font=font(16),fill=green)
            meter(30,157,m['forward_hz'],green);d.text((220,150),f'{m["forward_hz"]:.0f}',font=font(13),fill=FG)
            path(157,m['walking'],green)
            d.text((326,128),'rate -> forward threshold',font=font(15),fill=FG)
            d.text((342,165),f'{m["forward"]:.1f} > {m["forward_threshold"]:g}',font=font(13),fill=FG)
            d.rounded_rectangle((708,139,750,181),radius=7,fill=green if m['walking'] else (37,49,65))
            d.text((720,146),'W',font=font(23),fill=BG if m['walking'] else FG)
            d.text((766,149),'HELD' if m['walking'] else 'OFF',font=font(19),fill=green if m['walking'] else FG)

            d.text((30,194),'DNpe017 spikes',font=font(16),fill=gold)
            for j in range(min(12,m['spikes'])):d.line((33+j*14,238,33+j*14,220),fill=gold,width=3)
            d.text((220,219),str(m['spikes']),font=font(14),fill=FG);path(223,m['attacking'],gold)
            d.text((326,194),'pulse -> accumulate -> hold',font=font(15),fill=FG)
            meter(342,235,m['accumulator'],gold,width_px=160,maximum=m['threshold'])
            d.text((516,225),f'{m["hold_left"]} ticks',font=font(13),fill=FG)
            d.rounded_rectangle((713,202,748,245),radius=11,outline=gold,width=2)
            if m['attacking']:d.rectangle((718,206,730,223),fill=gold)
            d.line((731,204,731,225),fill=gold,width=1)
            d.text((766,213),'HELD' if m['attacking'] else 'OFF',font=font(19),fill=gold if m['attacking'] else FG)

        d.text((22,258),f'{p["distance"]:.1f}m to log  |  aim {np.degrees(p["angle"]):.0f} deg  |  break {p["progress"]:.0%}',font=font(15),fill=FG)
        behavior=p.get('behavior')
        if behavior:
            def cpct(value,count):return '--' if not count else f'{value:.0%}'
            d.text((22,278),
                f'PROG {behavior["normalized_target_progress"]:+.0%}   AIM<30 {behavior["aim_within_30_fraction"]:.0%}   '
                f'W aim/other {cpct(behavior["forward_when_ahead_fraction"],behavior["ahead_ticks"])}/{cpct(behavior["forward_when_not_ahead_fraction"],behavior["not_ahead_ticks"])}   '
                f'ATK aim/other {cpct(behavior["attack_when_ahead_fraction"],behavior["ahead_ticks"])}/{cpct(behavior["attack_when_not_ahead_fraction"],behavior["not_ahead_ticks"])}   '
                f'TURN {behavior["turn_bias_deg_per_tick"]:+.2f} deg/t',font=font(12),fill=(160,174,190))

        # Trace only control ticks; each tick is rendered three times into 60 Hz video.
        key=p['episode']
        if getattr(self,'teaching_episode',None)!=key:
            self.teaching_episode=key;self.teaching_trace=[]
        tick=int(p.get('episode_step',p['step']))
        trace_item={
            'tick':tick,'signal':signal,
            'aversive':bool(p.get('aversive_active',False)),
            'scheduled':bool((p.get('reinforcement') or {}).get('scheduled_aversive',False)),
        }
        if not self.teaching_trace or self.teaching_trace[-1]['tick']!=tick:self.teaching_trace.append(trace_item)
        else:self.teaching_trace[-1]=trace_item

        # Compact teaching card. Keep only the signals that matter on video:
        # what was taught now, how much plasticity was eligible, cumulative
        # signed teaching received this episode, and real PPL101 stimulation.
        top=303;teach_bottom=407
        d.rounded_rectangle((18,top,width-18,teach_bottom),radius=11,outline=(41,54,72),width=1)
        d.text((30,309),'PLASTIC TEACHING | THIS EPISODE',font=font(14),fill=FG)
        if not teaching:
            d.text((30,337),'Teaching is disabled during frozen evaluation / replay.',font=font(13),fill=MUTED)
        else:
            reinforcement=p.get('reinforcement') or {}
            plastic=reinforcement.get('plasticity') or {}
            counts=p.get('teaching_counts') or {'positive':0,'negative':0,'neutral':0}
            active_edges=int(plastic.get('active_edges',0))
            candidate_edges=int(plastic.get('candidate_edges',active_edges))
            total_edges=int(p.get('plastic_edges',0))
            if not total_edges:
                circuit=getattr(self.fly.brain,'circuit',None)
                total_edges=len(circuit['edges']) if isinstance(circuit,dict) and 'edges' in circuit else 0
            current_aversive=bool(p.get('aversive_active',False))
            scheduled=bool(reinforcement.get('scheduled_aversive',False))
            av_text=(f'PPL101 ACTIVE +{float(p.get("aversive_current",0.0)):.1f}' if current_aversive else
                     'PPL101 NEXT TICK' if scheduled else 'PPL101 off')
            av_col=AVERSIVE if current_aversive or scheduled else MUTED
            ab=d.textbbox((0,0),av_text,font=font(10));d.text((width-30-(ab[2]-ab[0]),311),av_text,font=font(10),fill=av_col)

            pos_sum=sum(max(0.0,float(item['signal'])) for item in self.teaching_trace)
            neg_sum=sum(min(0.0,float(item['signal'])) for item in self.teaching_trace)
            selected=f'{active_edges:,}/{candidate_edges:,}' if candidate_edges else f'{active_edges:,}/0'
            summary=(f'NOW {signal:+.3f}   selected {selected} candidate edges   |   '
                     f'sum +{pos_sum:.2f} / {neg_sum:.2f}   |   '
                     f'events +{counts.get("positive",0)} / -{counts.get("negative",0)} / 0 {counts.get("neutral",0)}')
            d.text((30,329),summary,font=font(10),fill=teach_color if signal else FG)

            # One symmetric autoscale for BOTH directions. Equal-magnitude
            # positive and negative teaching therefore always has equal visual
            # height. The range only expands when a stronger absolute event is
            # encountered, so weak signals still read clearly in compressed video.
            left,right=68,width-24;g_top,g_bottom=347,399;zero=(g_top+g_bottom)//2
            max_abs=max((abs(float(item['signal'])) for item in self.teaching_trace),default=0.0)
            signal_scale=max(0.03,max_abs)
            d.line((left,zero,right,zero),fill=(75,89,108),width=1)
            scale_lab=f'{signal_scale:.2f}' if signal_scale<1 else f'{signal_scale:.1f}'
            d.text((28,g_top-4),f'+{scale_lab}',font=font(8),fill=POS)
            d.text((42,zero-4),'0',font=font(8),fill=MUTED)
            d.text((28,g_bottom-7),f'-{scale_lab}',font=font(8),fill=NEG)
            if self.teaching_trace:
                last_tick=max(1,self.teaching_trace[-1]['tick'])
                half=(g_bottom-g_top)*0.46
                for item in self.teaching_trace[-400:]:
                    xx=left+(item['tick']-1)/max(1,last_tick-1)*(right-left)
                    v=float(item['signal'])
                    yy=zero-(v/signal_scale)*half
                    color=POS if v>0 else NEG if v<0 else (60,73,91)
                    if v!=0:d.line((xx,zero,xx,yy),fill=color,width=2)
                    if item['aversive']:d.rectangle((xx-1,g_top,xx+1,g_top+4),fill=AVERSIVE)
                    elif item['scheduled']:d.point((xx,g_top+6),fill=AVERSIVE)

        # Behavioral learning curve. This is deliberately separate from teaching:
        # it answers "is behavior improving?", not "what signal did we inject?".
        learn_top=413;learn_bottom=532
        d.rounded_rectangle((18,learn_top,width-18,learn_bottom),radius=11,outline=(41,54,72),width=1)
        curve=p.get('learning_curve') or {}
        episodes=list(curve.get('episodes') or [])
        vals=[float(v) for v in (curve.get('performance') or [])]
        rolling=[float(v) for v in (curve.get('rolling') or [])]
        successes=list(curve.get('success') or [])
        roll_window=int(curve.get('window',p.get('rolling_window',10)))
        d.text((30,419),'IS IT LEARNING? | EPISODE PERFORMANCE OVER TIME',font=font(13),fill=FG)
        d.text((30,436),f'combined task score   |   dots = episodes   cyan = rolling {roll_window}   gold ring = log broken   |   not input to fly',font=font(9),fill=CYAN)
        if not vals:
            d.text((30,474),'Waiting for the first completed episode...',font=font(13),fill=MUTED)
        else:
            plot_l,plot_r=69,width-25;plot_t,plot_b=452,520
            allv=vals+rolling
            yscale=max(0.25,max((abs(v) for v in allv),default=0.25))
            zero_y=(plot_t+plot_b)//2
            d.line((plot_l,zero_y,plot_r,zero_y),fill=(66,80,98),width=1)
            if yscale>=10:
                ylab=f'{yscale:.0f}'
            elif yscale>=1:
                ylab=f'{yscale:.1f}'
            else:
                ylab=f'{yscale:.2f}'
            d.text((27,plot_t-3),f'+{ylab}',font=font(8),fill=POS)
            d.text((43,zero_y-4),'0',font=font(8),fill=MUTED)
            d.text((27,plot_b-7),f'-{ylab}',font=font(8),fill=NEG)
            n=len(vals)
            def px(i):return plot_l if n<=1 else plot_l+i/(n-1)*(plot_r-plot_l)
            def py(v):return zero_y-(v/yscale)*(plot_b-plot_t)*0.46
            # Individual completed episodes: faint dots preserve the noisy data.
            for i,v in enumerate(vals):
                xx,yy=px(i),py(v)
                d.ellipse((xx-1.5,yy-1.5,xx+1.5,yy+1.5),fill=(119,133,151))
                if i<len(successes) and successes[i]:d.ellipse((xx-4,yy-4,xx+4,yy+4),outline=(245,195,78),width=2)
            if rolling:
                pts=[(px(i),py(v)) for i,v in enumerate(rolling)]
                if len(pts)>1:d.line(pts,fill=CYAN,width=3,joint='curve')
                else:
                    xx,yy=pts[0];d.ellipse((xx-2,yy-2,xx+2,yy+2),fill=CYAN)
            latest=rolling[-1] if rolling else vals[-1]
            success_count=sum(bool(x) for x in successes)
            status=f'rolling {latest:+.3f}   successes {success_count}/{len(vals)}'
            sb=d.textbbox((0,0),status,font=font(10));d.text((width-30-(sb[2]-sb[0]),419),status,font=font(10),fill=CYAN)

    def update(self,rgb,control,frame,preview_rgb=None):
        retina=self.retina_image()
        bins=getattr(self.fly,'activity_bins',None)
        if bins is None:
            self.render_frame(rgb,retina,self.fly.last_counts,control,frame,(frame+1)/20,.05,preview_rgb=preview_rgb)
        else:
            for k,counts in enumerate(bins):
                self.render_frame(rgb,retina,counts,control,frame*3+k,(frame*3+k)/60,1/60,write=self.record,preview_rgb=preview_rgb)

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
