"""Wordless, code-drawn characters and visual storytelling. No image assets."""
import math
import random
import pygame

V = pygame.Vector2
BG=(11,17,27); FG=(229,237,245); BLUE=(115,175,255)
GREEN=(90,215,140); PURPLE=(189,148,255); GOLD=(239,206,109)
RED=(242,116,123); DARK=(36,53,73); SKIN=(236,193,146)

def clip(t): return max(0.,min(1.,t))
def ease(t):
    t=clip(t)
    return t*t*(3-2*t)
def blend(a,b,t): return tuple(round(x+(y-x)*clip(t)) for x,y in zip(a,b))
def point(p): return tuple(round(x) for x in p)

def line(s,a,b,col=FG,w=5):
    pygame.draw.line(s,col,point(a),point(b),max(1,round(w)))
def circle(s,p,r,col,w=0):
    pygame.draw.circle(s,col,point(p),max(1,round(r)),round(w))
def rect(s,r,col,radius=8,w=0):
    pygame.draw.rect(s,col,pygame.Rect(*(round(x) for x in r)),round(w),border_radius=round(radius))
def path(s,pts,col=FG,w=4):
    pygame.draw.lines(s,col,False,[point(p) for p in pts],round(w))
def arrow(s,a,b,col=BLUE,w=5):
    a,b=V(a),V(b)
    if a.distance_to(b)<2:return
    d=(b-a).normalize(); base=b-d*18; n=V(-d.y,d.x)*10
    line(s,a,base,col,w)
    pygame.draw.polygon(s,col,[point(p) for p in (b,base+n,base-n)])
def check(s,p,scale=1,col=GREEN):
    p=V(p); path(s,[p+V(-18,0)*scale,p+V(-3,15)*scale,p+V(25,-21)*scale],col,7*scale)
def cross(s,p,scale=1,col=RED):
    p=V(p)
    line(s,p+V(-15,-15)*scale,p+V(15,15)*scale,col,6*scale)
    line(s,p+V(-15,15)*scale,p+V(15,-15)*scale,col,6*scale)
def plus(s,p,r=16,col=GREEN):
    line(s,V(p)-V(r,0),V(p)+V(r,0),col,6)
    line(s,V(p)-V(0,r),V(p)+V(0,r),col,6)

def floor(s,y=747):
    line(s,(96,y),(1504,y),DARK,3)

def tree(s,x,y,scale=1,hit=0):
    x+=math.sin(hit*math.pi*5)*5 if hit else 0
    rect(s,(x-24*scale,y-156*scale,48*scale,156*scale),(131,94,63),3)
    for dx,dy,w,h,col in [(-121,-288,242,145,(78,153,92)),(-86,-348,172,132,(99,183,111)),(-143,-211,99,96,(86,163,99)),(45,-227,99,112,(73,144,88))]:
        rect(s,(x+dx*scale,y+dy*scale,w*scale,h*scale),col,3)
    if hit:
        for i in range(9):
            a=i*2.399; r=20+hit*80
            p=V(x,y-93*scale)+V(math.cos(a),math.sin(a))*r
            rect(s,(p.x,p.y,7*(1-hit)+2,7*(1-hit)+2),GOLD,1)

def fly(s,p,r=34,t=0):
    x,y=p
    wing=.8+.2*math.sin(t*13)**2
    for sign in (-1,1):
        pygame.draw.ellipse(s,FG,(x+sign*r*.8-r*.6,y-r*.9,r*1.2,r*.8*wing),3)
    pygame.draw.ellipse(s,GOLD,(x-r*.66,y-r*.8,r*1.32,r*1.6))
    circle(s,(x+r*.48,y-r*.18),r*.17,RED)
    for yy in (.05,.35):line(s,(x-r*.5,y+r*yy),(x+r*.5,y+r*yy),(172,139,62),max(2,r*.07))

def chip(s,p,size=70,activity=0):
    x,y=p
    for i in range(4):
        d=(i-1.5)*size*.2
        for sign in (-1,1):
            line(s,(x+sign*size*.48,y+d),(x+sign*size*.65,y+d),PURPLE,4)
            line(s,(x+d,y+sign*size*.48),(x+d,y+sign*size*.65),PURPLE,4)
    rect(s,(x-size/2,y-size/2,size,size),(39,30,61),12)
    rect(s,(x-size/2,y-size/2,size,size),PURPLE,12,3)
    for i in range(3):
        h=size*(.17+.3*(.5+.5*math.sin(activity*2+i*1.8)))
        rect(s,(x-size*.28+i*size*.2,y+size*.25-h,size*.11,h),PURPLE,2)

def lock(s,p,scale=1):
    x,y=p
    pygame.draw.arc(s,BLUE,(x-14*scale,y-23*scale,28*scale,33*scale),0,math.pi,round(4*scale))
    rect(s,(x-21*scale,y-4*scale,42*scale,31*scale),BLUE,5*scale)
    circle(s,(x,y+8*scale),3*scale,BG)

def robot(s,x,ground=747,scale=1,t=0,walk=0,punch=0,appear=1,connected=1,shrug=0):
    """Mechanical pictogram with a glass fly compartment and visible chip."""
    bob=abs(math.sin(t*7))*5*walk
    def p(dx,dy): return V(x+dx*scale,ground+(dy-bob)*scale)
    # Legs are jointed; stride eases to zero at both ends of a walk.
    stride=math.sin(t*7)*23*walk
    for sign in (-1,1):
        hip=p(sign*29,-66); knee=p(sign*29+sign*stride*.4,-34)
        foot=p(sign*30+sign*stride,-8)
        path(s,[hip,knee,foot],BLUE,12*scale)
        rect(s,(foot.x-20*scale,ground-16*scale,46*scale,16*scale),FG,6*scale)
    rect(s,(x-69*scale,ground-(175+bob)*scale,138*scale,119*scale),DARK,25*scale)
    rect(s,(x-69*scale,ground-(175+bob)*scale,138*scale,119*scale),FG,25*scale,4*scale)
    if connected:
        chip(s,p(0,-115),49*scale*connected,t)
    else:
        rect(s,(x-25*scale,ground-(140+bob)*scale,50*scale,50*scale),BG,7*scale,2)
    # Arms connect at shoulders and extend toward the tree.
    for sign in (-1,1):
        shoulder=p(sign*74,-151)
        elbow=p(sign*(92+11*walk*math.sin(t*7)),-108)
        hand=p(sign*97,-76)
        if shrug:
            elbow=elbow.lerp(p(sign*101,-124),shrug)
            hand=hand.lerp(p(sign*119,-161),shrug)
        if sign==1 and punch:
            extension=math.sin(punch*math.pi)
            elbow=elbow.lerp(p(114,-135),extension)
            hand=hand.lerp(p(177,-132),extension)
        path(s,[shoulder,elbow,hand],BLUE,10*scale)
        circle(s,shoulder,8*scale,FG)
        rect(s,(hand.x-12*scale,hand.y-12*scale,24*scale,24*scale),FG,6*scale)
    # Transparent dome; the fly is always clearly visible inside.
    dome=p(0,-236)
    circle(s,dome,71*scale,(17,36,51))
    circle(s,dome,71*scale,BLUE,round(4*scale))
    pygame.draw.arc(s,FG,(dome.x-58*scale,dome.y-58*scale,116*scale,116*scale),1.9,2.8,max(1,round(3*scale)))
    fly(s,dome,36*scale,t)
    rect(s,(x-55*scale,ground-(180+bob)*scale,110*scale,14*scale),FG,5*scale)
    return dome,p(0,-115)

def teacher(s,x,ground=747,scale=1,t=0,walk=0,punch=0,pointing=0,facing=1):
    bob=abs(math.sin(t*7))*5*walk
    def p(dx,dy):return V(x+dx*scale*facing,ground+(dy-bob)*scale)
    stride=math.sin(t*7)*26*walk
    for sign in (-1,1):
        foot=p(sign*22+sign*stride,-9)
        path(s,[p(sign*21,-95),p(sign*24+sign*stride*.4,-50),foot],FG,15*scale)
        rect(s,(foot.x-14*scale,ground-14*scale,41*scale,14*scale),FG,5*scale)
    rect(s,(x-47*scale,ground-(211+bob)*scale,94*scale,122*scale),GREEN,25*scale)
    # Shirt collar and tie, glasses, and mortarboard distinguish the teacher.
    pygame.draw.polygon(s,FG,[point(p(-22,-208)),point(p(22,-208)),point(p(0,-177))])
    line(s,p(0,-191),p(0,-145),BG,6*scale)
    for sign in (-1,1):
        shoulder=p(sign*45,-187); elbow=p(sign*68,-151); hand=p(sign*69,-111)
        if sign==1:
            extension=math.sin(punch*math.pi) if punch else 0
            amount=max(extension,pointing)
            elbow=elbow.lerp(p(110,-155),amount)
            hand=hand.lerp(p(185,-133),amount)
        path(s,[shoulder,elbow,hand],GREEN,14*scale)
        circle(s,hand,12*scale,SKIN)
    circle(s,p(0,-258),44*scale,SKIN)
    for dx in (-18,18):
        circle(s,p(dx,-262),15*scale,BG,round(4*scale))
        circle(s,p(dx+4,-262),3*scale,BG)
    line(s,p(-3,-262),p(3,-262),BG,3*scale)
    pygame.draw.arc(s,BG,(x-12*scale,ground-(255+bob)*scale,25*scale,22*scale),math.pi,math.tau,max(1,round(3*scale)))
    pygame.draw.polygon(s,FG,[point(p(-61,-301)),point(p(0,-323)),point(p(61,-301)),point(p(0,-280))])
    line(s,p(55,-300),p(55,-268),GOLD,4*scale)
    circle(s,p(55,-265),5*scale,GOLD)

_rng=random.Random(28)
NODES=[V(math.cos(i*2.4),math.sin(i*2.4))*math.sqrt(_rng.random()) for i in range(104)]
EDGES=[(i,(i*19+17)%104) for i in range(104)]

def neural(s,center,size,t,guess=False):
    pts=[V(center)+V(p.x*size,p.y*size*.73) for p in NODES]
    for i,(a,b) in enumerate(EDGES):
        activity=max(0,math.sin(t*2.4+i*1.31))**7
        col=blend(DARK,BLUE,.13+activity*.65)
        width=2
        if guess and i%8==0:
            strength=.5+.5*math.sin(t*1.7+i)
            col=blend(DARK,GREEN if i%3 else RED,.3+strength*.7)
            width=round(2+strength*5)
        line(s,pts[a],pts[b],col,width)
    for i,p in enumerate(pts):
        a=(.5+.5*math.sin(t*2.8+i*2.1))**4
        circle(s,p,3+a*4,blend(DARK,BLUE,.3+a*.7))
    return pts

def pulse_path(s,pts,t,col=BLUE,count=3):
    pts=[V(p) for p in pts]
    path(s,pts,blend(BG,col,.35),3)
    lengths=[a.distance_to(b) for a,b in zip(pts,pts[1:])]
    total=sum(lengths)
    for j in range(count):
        d=((t*.3+j/count)%1)*total
        for a,b,length in zip(pts,pts[1:],lengths):
            if d<=length:
                circle(s,a.lerp(b,d/max(length,1)),6,col);break
            d-=length

def thought(s,center,r=130):
    circle(s,center,r,BG)
    circle(s,center,r,DARK,3)

def failure(s,t):
    floor(s)
    tree(s,1260,747,1.2)
    # Several attempts visibly converge on the SAME unsuccessful endpoint.
    start=V(300,650); end=V(805,545)
    attempt=min(2,int(t/2.1)); local=clip((t-attempt*2.1)/1.55)
    for j in range(attempt+1):
        progress=ease(local) if j==attempt else 1
        pts=[]
        for k in range(70):
            u=k/69*progress
            p=start.lerp(end,u)+V(0,math.sin(u*math.pi)*(j-1)*74)
            pts.append(p)
        path(s,pts,BLUE if j==attempt else DARK,5 if j==attempt else 3)
    opacity=1 if local<.85 or attempt==2 else clip((2.1-(t-attempt*2.1))/.55)
    if attempt:
        opacity*=ease((t-attempt*2.1)/.25)
    layer=pygame.Surface(s.get_size(),pygame.SRCALPHA)
    pos=pts[-1]
    fly(layer,pos,55,t)
    layer.set_alpha(round(opacity*255));s.blit(layer,(0,0))
    # The untouched tree stays out of reach. No graph or written explanation.
    if t>5.5:
        cross(s,(909,545),1.2)

def credit(s,t,guess=False):
    tree(s,470,733,.8)
    target=V(470,570);pos=V(238,623)
    fly(s,pos,52,t)
    angle=.6+(-.23-.6)*ease(t/2)
    arrow(s,pos+V(35*math.cos(angle),35*math.sin(angle)),pos+V(133*math.cos(angle),133*math.sin(angle)))
    pts=neural(s,(1090,415),350,t,guess)
    # One earned reward fans out among many simultaneously active connections.
    plus(s,(370,385),24)
    for j in range(9 if guess else 5):
        dest=pts[(j*17+9)%len(pts)]
        phase=(t*.35+j*.13)%1
        start=V(395,385)
        control=V(690,160+j*43)
        p=start*(1-phase)**2+control*2*(1-phase)*phase+dest*phase**2
        circle(s,p,7,GREEN)
        if phase>.80:
            circle(s,dest,12+18*(phase-.8)/.2,GOLD if not guess else (GREEN if j%2 else RED),2)
    if not guess:
        # Three candidate causes blink together, leaving no obvious winner.
        for k in (9,26,43):circle(s,pts[k],17,GOLD,2)

def new_robot(s,t):
    # First establish the idea in the abstract, before introducing a character.
    abstract=pygame.Surface(s.get_size(),pygame.SRCALPHA)
    center=V(615,432)
    neural(abstract,center,244,t)
    circle(abstract,center,269,BLUE,3)
    lock(abstract,(615,129),1.3)
    # Reward particles can no longer reach and change the brain itself.
    if t<3.3:
        u=ease(t/2)
        plus(abstract,(160,430),22)
        p=V(194,430).lerp(V(333,430),u)
        circle(abstract,p,9,GREEN)
        if t>1.5:cross(abstract,(307,374),1.1)
    growth=ease((t-2)/2.5)
    if growth:
        pygame.draw.arc(abstract,PURPLE,(279,96,672,672),-math.pi/2,-math.pi/2+math.tau*growth,5)
        # Adjustable elements live on the OUTER ring, with fixed blue wiring inside.
        for i in range(5):
            a=-math.pi/2+(i+1)*math.tau/6
            if growth<(i+1)/6:continue
            pos=center+V(math.cos(a),math.sin(a))*336
            circle(abstract,pos,21,BG)
            circle(abstract,pos,21,PURPLE,3)
            d=V(math.cos(t*.75+i),math.sin(t*.75+i))*13
            line(abstract,pos,pos+d,PURPLE,4)
        pulse_path(abstract,[(887,432),(1039,432)],t,PURPLE,2)
        chip(abstract,(1190,432),148*growth,t)
    abstract.set_alpha(round(255*(1-ease((t-8)/2))))
    s.blit(abstract,(0,0))
    # Only after the outer learning mechanism is established do we reveal the robot.
    if t>8:
        character=pygame.Surface(s.get_size(),pygame.SRCALPHA)
        floor(character)
        robot(character,640,747,1.65,t,connected=0)
        lock(character,(640,188),1.25)
        line(character,(755,563),(1040,393),DARK,3)
        thought(character,(1190,345),150)
        chip(character,(1190,345),155,t)
        character.set_alpha(round(255*ease((t-8)/2)))
        s.blit(character,(0,0))

def action_icon(s,p,kind,col=FG,size=1):
    x,y=p
    if kind=='turn':
        path(s,[(x+23*size,y+25*size),(x+23*size,y-17*size),(x-18*size,y-17*size)],col,6*size)
        arrow(s,(x+3*size,y-17*size),(x-32*size,y-17*size),col,6*size)
    else:
        rect(s,(x-28*size,y-16*size,49*size,34*size),col,7*size)
        rect(s,(x-13*size,y+7*size,29*size,17*size),col,5*size)
        for i in range(3):line(s,(x+(-14+i*11)*size,y-15*size),(x+(-14+i*11)*size,y-5*size),BG,2*size)

def robot_acts_pose(t):
    return {
        'connected':ease((t-10)/2),
        'x':430+610*ease((t-14)/4),
        'walk':math.sin(math.pi*clip((t-14)/4)) if 14<t<18 else 0,
        'punch':clip((t-19)/.8) if 19<t<19.8 else 0,
        'shrug':ease((t-5)/.6)*(1-ease((t-9)/.7)),
    }

def robot_acts(s,t):
    floor(s)
    pose=robot_acts_pose(t)
    x,walk,punch=pose['x'],pose['walk'],pose['punch']
    connected=pose['connected']
    # Routes are behind the character, with the cutaway above its head.
    line(s,(687,419),(x,615),DARK,2)
    tree(s,1260,747,1.2,punch)
    dome,c=robot(s,x,747,1.15,t,walk,punch,connected=connected,shrug=pose['shrug'])
    if t<10:
        pulse_path(s,[(1140,437),(957,487),(700,486),dome],t,BLUE,4)
        circle(s,dome,88,blend(DARK,BLUE,.4+.3*math.sin(t*3)),2)
    thought(s,(752,237),196)
    neural(s,(668,231),91,t)
    # Until the chip arrives, activity hits an open cable and the controls remain inert.
    pulse_path(s,[(757,231),(783,231)],t,BLUE,1)
    circle(s,(787,231),6,BLUE,2)
    if 4<t:
        visible=ease((t-4)/.6)
        col=blend(BG,FG,.5*visible) if connected<1 else PURPLE
        for y,kind in ((153,'turn'),(302,'punch')):
            action_icon(s,(1085,y),kind,col,1.15)
        if t<10:
            # Two unmatched sockets make the missing translation explicit.
            path(s,[(957,153),(984,153),(984,204)],DARK,3)
            path(s,[(957,302),(984,302),(984,249)],DARK,3)
            circle(s,(984,204),7,DARK,2)
            circle(s,(984,249),7,DARK,2)
            if t<9:
                pygame.draw.arc(s,GOLD,(843,185,44,42),-math.pi/2,math.pi,5)
                path(s,[(865,226),(865,245)],GOLD,5)
                circle(s,(865,262),4,GOLD)
    if t>=9:
        landing=ease((t-9)/2)
        location=V(860,110).lerp(V(863,231),landing)
        chip(s,location,98,t)
        if connected:
            pulse_path(s,[(794,231),(801,231)],t,BLUE,1)
            pulse_path(s,[(927,231),(974,231),(974,153),(1035,153)],t,PURPLE,2)
            pulse_path(s,[(927,231),(974,231),(974,302),(1035,302)],t,PURPLE,2)
    if 12<t<14:
        action_icon(s,(x,809),'turn',BLUE,1)
        action_icon(s,(1085,153),'turn',BLUE,1.15)
    if punch:
        action_icon(s,(1085,302),'punch',GREEN,1.15)

def teaching_pose(t):
    # Teacher demonstrates once, steps aside; the robot then imitates.
    tx=660+300*ease(t/2.5)+480*ease((t-4.2)/1.8)
    tw=math.sin(math.pi*clip(t/2.5)) if t<2.5 else (math.sin(math.pi*clip((t-4.2)/1.8)) if 4.2<t<6 else 0)
    tp=clip((t-3)/.8) if 3<t<3.8 else 0
    rx=295+720*ease((t-6)/3)
    rw=math.sin(math.pi*clip((t-6)/3)) if 6<t<9 else 0
    rp=clip((t-9.3)/.8) if 9.3<t<10.1 else 0
    return tx,tw,tp,rx,rw,rp

def teaching(s,t):
    floor(s)
    tx,tw,tp,rx,rw,rp=teaching_pose(t)
    # The teacher passes behind the tree when stepping out of the learner's way.
    line(s,(360,361),(rx,747-115*1.05),DARK,2)
    teacher(s,tx,747,1.1,t,tw,tp,pointing=.65*ease((t-10.2)/.7),facing=-1 if t>=6 else 1)
    tree(s,1200,747,1.2,max(tp,rp))
    dome,c=robot(s,rx,747,1.05,t,rw,rp)
    # A cutaway thought-bubble belongs to the learner: neural activity → chip.
    thought(s,(360,224),137)
    neural(s,(299,215),66,t)
    chip(s,(408,231),77,t)
    pulse_path(s,[(347,223),(360,230)],t,BLUE,1)
    # Teacher gestures become a moving demonstration token, captured by the chip.
    if 2.7<t<6:
        u=ease((t-3.3)/2)
        a=V(1090,521);b=V(700,133);d=V(408,231)
        token=a*(1-u)**2+b*2*(1-u)*u+d*u*u
        circle(s,token,27,BG)
        circle(s,token,27,GREEN,3)
        # Tiny fist pictogram, not a text label.
        rect(s,(token.x-13,token.y-10,26,20),GREEN,5)
    if t>10.5:
        check(s,(1430,352),1.5)
        circle(s,(408,231),55,GREEN,3)

def input_boundary(s,t):
    floor(s)
    tree(s,1280,747,1.2)
    dome,c=robot(s,455,747,1.5,t)
    # Large literal cutaway of the robot's interior.
    thought(s,(880,282),174)
    neural(s,(804,268),76,t)
    chip(s,(961,299),98,t)
    pulse_path(s,[(870,280),(894,293)],t,BLUE,1)
    line(s,(566,440),(738,389),DARK,3)
    # Information from the visible world reaches the fly through sight.
    pulse_path(s,[(1160,406),(1020,506),(730,517),(556,401)],t,BLUE,4)
    # An attempted shortcut of a tree icon to the chip is visibly blocked.
    if t>1:
        u=ease((t-1)/2)
        p=V(1210,256).lerp(V(1081,282),u)
        tree(s,p.x,p.y+27,.17)
        line(s,(1076,223),(1076,343),RED,6)
        if t>2.8:cross(s,(1129,213),1.1)
    lock(s,(455,228),1.05)

DRAW=(failure,credit,lambda s,t:credit(s,t,True),new_robot,robot_acts,teaching,input_boundary)

def render(index,t,duration):
    s=pygame.Surface((1600,900));s.fill(BG)
    DRAW[index](s,max(0,min(t,duration)))
    return s
