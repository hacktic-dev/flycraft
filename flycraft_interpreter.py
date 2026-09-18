"""Deterministic FlyCraft narration animation. Run --help for playback/export options."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess

os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
import pygame

W, H = 1600, 900
BG = (11, 17, 27)
PANEL = (17, 26, 39)
FG = (231, 237, 244)
DIM = (146, 162, 180)
BORDER = (48, 65, 85)
BLUE = (115, 175, 255)
GREEN = (90, 215, 140)
RED = (242, 116, 123)
GOLD = (239, 206, 109)
PURPLE = (189, 148, 255)
V = pygame.Vector2

SCENES = [
    ('The learning hit a ceiling', '01 / THE PLATEAU', 7.0),
    ('Which connections actually helped?', '02 / THE CREDIT PROBLEM', 9.0),
    ('The reward system was guessing', '03 / THE WRONG CONNECTIONS', 7.0),
    ('Keep the brain. Learn around it.', '04 / A NEW APPROACH', 15.0),
    ('A little interpreter turns activity into action', '05 / THE ROBOT', 21.0),
    ('Learn to predict an expert', '06 / THE TEACHER', 12.0),
    ('The interpreter only sees the brain', '07 / THE INPUT BOUNDARY', 9.0),
]
TOTAL = sum(s[2] for s in SCENES)
FONTS = {}
TEXT_BOUNDS = []


def clamp(t):
    return max(0.0, min(1.0, t))


def ease(t):
    t = clamp(t)
    return t * t * (3 - 2 * t)


def mix(a, b, t):
    return tuple(round(x + (y - x) * clamp(t)) for x, y in zip(a, b))


def font(size, bold=False):
    key = size, bold
    if key not in FONTS:
        FONTS[key] = pygame.font.SysFont('segoeui', size, bold=bold)
    return FONTS[key]


def text(s, value, xy, size=28, color=FG, bold=False, center=False):
    img = font(size, bold).render(str(value), True, color)
    rect = img.get_rect()
    if center:
        rect.midtop = tuple(round(v) for v in xy)
    else:
        rect.topleft = tuple(round(v) for v in xy)
    TEXT_BOUNDS.append((str(value), rect))
    s.blit(img, rect)
    return rect


def line(s, a, b, color=BORDER, width=2):
    pygame.draw.line(s, color, a, b, width)


def arrow(s, a, b, color=BLUE, width=4, head=15):
    a, b = V(a), V(b)
    if a.distance_to(b) < 1:
        return
    direction = (b - a).normalize()
    base = b - direction * head
    side = V(-direction.y, direction.x) * head * .55
    line(s, a, base, color, width)
    pygame.draw.polygon(s, color, (b, base + side, base - side))


def card(s, rect, color=BORDER, fill=PANEL, width=2):
    rect = pygame.Rect(rect)
    pygame.draw.rect(s, fill, rect, border_radius=22)
    pygame.draw.rect(s, color, rect, width, border_radius=22)
    return rect


def render_scene(index, t):
    from flycraft_pictograms import render
    return render(index, t, SCENES[index][2])


def locate(t):
    t=max(0,min(t,TOTAL))
    start=0.0
    for i,(_,_,duration) in enumerate(SCENES):
        if t < start+duration or i==len(SCENES)-1:
            return i,t-start
        start+=duration


def render_timeline(t):
    index,local=locate(t)
    frame=render_scene(index,local)
    if index and local<.5:
        old=render_scene(index-1,SCENES[index-1][2])
        frame.set_alpha(round(255*ease(local/.5)))
        old.blit(frame,(0,0))
        return old
    return frame


def present(screen,frame):
    sw,sh=screen.get_size()
    scale=min(sw/W,sh/H)
    size=(max(1,round(W*scale)),max(1,round(H*scale)))
    screen.fill(BG)
    screen.blit(pygame.transform.smoothscale(frame,size),((sw-size[0])//2,(sh-size[1])//2))
    pygame.display.flip()


def export_video(path,fps=60,size=(1920,1080)):
    exe=shutil.which('ffmpeg')
    if not exe:
        try:
            import imageio_ffmpeg
            exe=imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            raise SystemExit('Video export requires ffmpeg on PATH or imageio-ffmpeg.')
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    command=[exe,'-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}',
             '-r',str(fps),'-i','-','-an','-vf',f'scale={size[0]}:{size[1]}:flags=lanczos',
             '-c:v','libx264','-preset','fast','-crf','17','-pix_fmt','yuv420p','-movflags','+faststart',str(path)]
    with subprocess.Popen(command,stdin=subprocess.PIPE) as process:
        try:
            for n in range(round(TOTAL*fps)):
                TEXT_BOUNDS.clear()
                frame=render_timeline(n/fps)
                process.stdin.write(pygame.image.tobytes(frame,'RGB'))
                if n%(fps*5)==0:
                    print(f'Export {n/fps:.0f}/{TOTAL:.0f}s',flush=True)
        finally:
            process.stdin.close()
        if process.wait()!=0:
            raise RuntimeError('Video encoding failed')
    print(f'Saved {path.resolve()}',flush=True)


def verify(out):
    out=Path(out)
    out.mkdir(parents=True,exist_ok=True)
    sheet=pygame.Surface((1200,7*225))
    # Three samples per scene, including the exact final pose.
    for index,(_,_,duration) in enumerate(SCENES):
        for j,t in enumerate((0.0,duration*.52,duration)):
            TEXT_BOUNDS.clear()
            frame=render_scene(index,t)
            assert not TEXT_BOUNDS, 'No text in the animation'
            for border in ((0,0,W,24),(0,H-24,W,24),(0,0,24,H),(W-24,0,24,H)):
                pixels=pygame.image.tobytes(frame.subsurface(border),'RGB')
                assert pixels==bytes(BG)*(border[2]*border[3]), (index,t,'Clipped artwork')
            for label,r in TEXT_BOUNDS:
                assert pygame.Rect(0,0,W,H).contains(r),(index,label,r)
            raw=pygame.image.tobytes(frame,'RGB')
            assert raw==pygame.image.tobytes(render_scene(index,t),'RGB'), 'Non-deterministic render'
            sheet.blit(pygame.transform.smoothscale(frame,(400,225)),(j*400,index*225))
            pygame.image.save(frame,out/f'scene-{index+1}-{j}.png')
    pygame.image.save(sheet,out/'contact-sheet.png')
    # Arbitrary frame history must not alter an export or paused frame.
    histories=[]
    for fps in (15,60):
        for n in range(fps):
            render_scene(5,n/fps)
        reference=pygame.image.tobytes(render_scene(5,7.25),'RGB')
        histories.append(reference)
    assert histories[0]==histories[1], 'Frame-rate dependent state'
    for i,(_,_,duration) in enumerate(SCENES):
        assert pygame.image.tobytes(render_scene(i,duration),'RGB')==pygame.image.tobytes(render_scene(i,duration+10),'RGB')
    from flycraft_pictograms import teaching_pose
    assert teaching_pose(3.4)[2] > 0 and teaching_pose(9.7)[5] > 0
    assert teaching_pose(11)[2] == 0 and teaching_pose(11)[5] == 0
    from flycraft_pictograms import robot_acts_pose
    for sample in (0,2,5,7,9,10,11,12,13,14):
        pose=robot_acts_pose(sample)
        assert pose['x']==430 and pose['walk']==0 and pose['punch']==0
    assert robot_acts_pose(7)['shrug']==1
    assert robot_acts_pose(7)['connected']==0
    assert robot_acts_pose(12)['connected']==1
    assert robot_acts_pose(16)['walk']>0
    assert robot_acts_pose(19.4)['punch']>0
    # Inspect intermediate construction and insertion frames, not just endpoints.
    for index in (3,4):
        for n in range(int(SCENES[index][2]*4)+1):
            frame=render_scene(index,n/4)
            for border in ((0,0,W,24),(0,H-24,W,24),(0,0,24,H),(W-24,0,24,H)):
                assert pygame.image.tobytes(frame.subsurface(border),'RGB')==bytes(BG)*(border[2]*border[3])
    assert locate(TOTAL)[0]==6
    for boundary in (sum(s[2] for s in SCENES[:i]) for i in range(1,7)):
        render_timeline(boundary)
        render_timeline(boundary+.25)
    for size in ((1280,720),(900,900),(1920,1080),(640,360)):
        screen=pygame.display.set_mode(size)
        present(screen,render_scene(5,6.3))
    assert not TEXT_BOUNDS, "Animation must remain wordless"
    print('PASS: scene samples and 146 middle-section frames; wordless; frame margins; repeatability; end holds; interpreter-before-movement; teacher punches; transitions; four window sizes.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--preview',action='store_true')
    parser.add_argument('--export',metavar='MP4')
    parser.add_argument('--fps',type=int,default=60)
    parser.add_argument('--output',default='artifacts/interpreter-quality')
    args=parser.parse_args()
    if args.fps<1: parser.error('--fps must be positive')
    if args.self_test or args.preview or args.export:
        os.environ.setdefault('SDL_VIDEODRIVER','dummy')
        os.environ.setdefault('SDL_AUDIODRIVER','dummy')
    pygame.init()
    try:
        if args.self_test or args.preview: verify(args.output)
        if args.export: export_video(args.export,args.fps)
        if args.self_test or args.preview or args.export: return
        screen=pygame.display.set_mode((1280,720),pygame.RESIZABLE)
        pygame.display.set_caption('FlyCraft • Interpreter Explainer')
        clock=pygame.time.Clock()
        timeline=0.0
        paused=False
        fullscreen=False
        help_visible=False
        windowed=(1280,720)
        selected=None
        running=True
        while running:
            dt=min(clock.tick(60)/1000,.1)
            seek=False
            for event in pygame.event.get():
                if event.type==pygame.QUIT: running=False
                elif event.type==pygame.KEYDOWN:
                    if event.key==pygame.K_ESCAPE: running=False
                    elif event.key==pygame.K_SPACE: paused=not paused
                    elif pygame.K_1<=event.key<=pygame.K_7:
                        selected=event.key-pygame.K_1
                        timeline=0.0
                        paused=False
                        seek=True
                    elif event.key in (pygame.K_r,pygame.K_a):
                        selected=None
                        timeline=0.0
                        paused=False
                        seek=True
                    elif event.key==pygame.K_h: help_visible=not help_visible
                    elif event.key==pygame.K_F11:
                        fullscreen=not fullscreen
                        if fullscreen:
                            windowed=screen.get_size()
                            screen=pygame.display.set_mode((0,0),pygame.FULLSCREEN)
                        else: screen=pygame.display.set_mode(windowed,pygame.RESIZABLE)
                    elif event.key in (pygame.K_LEFT,pygame.K_RIGHT):
                        limit=TOTAL if selected is None else SCENES[selected][2]
                        timeline=max(0,min(limit,timeline+(1 if event.key==pygame.K_RIGHT else -1)))
                        seek=True
                elif event.type==pygame.VIDEORESIZE and not fullscreen:
                    screen=pygame.display.set_mode(event.size,pygame.RESIZABLE)
            limit=TOTAL if selected is None else SCENES[selected][2]
            if not paused and not seek: timeline=min(limit,timeline+dt)
            TEXT_BOUNDS.clear()
            frame=render_timeline(timeline) if selected is None else render_scene(selected,timeline)
            if help_visible:
                card(frame,(210,410,1180,76),BLUE,BG)
                text(frame,'1–7 scene  •  A/R full sequence  •  Space pause  •  ←/→ seek  •  F11 fullscreen  •  H help',(800,429),23,FG,center=True)
            present(screen,frame)
    finally:
        pygame.quit()


if __name__=='__main__':
    main()
