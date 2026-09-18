
import {
  Circle,
  Img,
  Line,
  Polygon,
  Rect,
  Txt,
  makeScene2D,
} from '@motion-canvas/2d';
import {all, createRef, easeInOutCubic, waitFor} from '@motion-canvas/core';

const BG = '#08111F';
const PANEL = '#0D1A2B';
const PANEL_STROKE = '#17375A';
const CYAN = '#44D5EE';
const GREEN = '#63E7B8';
const RED = '#FF6D69';
const GOLD = '#F3C663';
const WHITE = '#E6F0FF';
const MINT = '#7EF0C7';
const DARKLINE = '#35557E';

const FLY = '/assets/fly-top.svg';
const TREE = '/assets/tree-top.svg';
const BRAIN = '/assets/brain-pink.svg';

export default makeScene2D(function* (view) {
  view.fill(BG);

  const flyGood = createRef<Img>();
  const flyBad = createRef<Img>();
  const coneGood = createRef<Polygon>();
  const coneBad = createRef<Polygon>();
  const arrowGood = createRef<Line>();
  const arrowBad = createRef<Line>();
  const plusA = createRef<Txt>();
  const plusB = createRef<Txt>();
  const minusA = createRef<Txt>();
  const minusB = createRef<Txt>();

  const goodBrain = createRef<Img>();
  const badBrain = createRef<Img>();

  const gLines = Array.from({length: 4}, () => createRef<Line>());
  const gNodes = Array.from({length: 5}, () => createRef<Circle>());
  const bLines = Array.from({length: 4}, () => createRef<Line>());
  const bNodes = Array.from({length: 5}, () => createRef<Circle>());

  const centerXLeft = -480;
  const centerXRight = 480;
  const centerYTop = -180;
  const centerYBottom = 180;
  const panelW = 860;
  const panelH = 330;

  view.add(
    <>
      <Rect width={1920} height={1080} fill={BG} />

      {/* panels */}
      <Rect x={centerXLeft} y={centerYTop} width={panelW} height={panelH} radius={22} fill={PANEL} stroke={PANEL_STROKE} lineWidth={3} />
      <Rect x={centerXRight} y={centerYTop} width={panelW} height={panelH} radius={22} fill={PANEL} stroke={PANEL_STROKE} lineWidth={3} />
      <Rect x={centerXLeft} y={centerYBottom} width={panelW} height={panelH} radius={22} fill={PANEL} stroke={PANEL_STROKE} lineWidth={3} />
      <Rect x={centerXRight} y={centerYBottom} width={panelW} height={panelH} radius={22} fill={PANEL} stroke={PANEL_STROKE} lineWidth={3} />

      <Txt x={-775} y={-330} text={'TURN TOWARD'} fill={CYAN} fontSize={24} fontFamily={'JetBrains Mono, monospace'} />
      <Txt x={185} y={-330} text={'REWARDED PATH'} fill={CYAN} fontSize={24} fontFamily={'JetBrains Mono, monospace'} />
      <Txt x={-785} y={30} text={'TURN AWAY'} fill={CYAN} fontSize={24} fontFamily={'JetBrains Mono, monospace'} />
      <Txt x={205} y={30} text={'WEAKENED PATH'} fill={CYAN} fontSize={24} fontFamily={'JetBrains Mono, monospace'} />

      {/* top-left good action panel */}
      <Img src={TREE} x={-565} y={-240} width={145} />
      <Img ref={flyGood} src={FLY} x={-560} y={-95} width={120} rotation={40} />
      <Polygon
        ref={coneGood}
        points={[[0,-62],[-26,-8],[26,-8]]}
        x={-560}
        y={-95}
        rotation={40}
        fill={'rgba(68,213,238,0.18)'}
        stroke={CYAN}
        lineWidth={3}
        closed
      />
      <Line
        ref={arrowGood}
        points={[[-694,-101],[-735,-145],[-718,-200],[-655,-234]]}
        stroke={WHITE}
        lineWidth={4}
        lineCap={'round'}
      />
      <Txt ref={plusA} x={-350} y={-165} text={'+'} fill={MINT} fontSize={68} fontFamily={'JetBrains Mono, monospace'} opacity={0} />
      <Txt ref={plusB} x={-285} y={-135} text={'+'} fill={MINT} fontSize={42} fontFamily={'JetBrains Mono, monospace'} opacity={0} />

      {/* top-right rewarded brain */}
      <Img ref={goodBrain} src={BRAIN} x={480} y={-180} width={315} />
      <Line ref={gLines[0]} points={[[350,-197],[432,-230]]} stroke={DARKLINE} lineWidth={8} lineCap={'round'} />
      <Line ref={gLines[1]} points={[[432,-230],[516,-188]]} stroke={DARKLINE} lineWidth={8} lineCap={'round'} />
      <Line ref={gLines[2]} points={[[516,-188],[594,-230]]} stroke={DARKLINE} lineWidth={8} lineCap={'round'} />
      <Line ref={gLines[3]} points={[[516,-188],[603,-147]]} stroke={DARKLINE} lineWidth={8} lineCap={'round'} />
      <Circle ref={gNodes[0]} x={350} y={-197} size={26} fill={PANEL} stroke={CYAN} lineWidth={4} />
      <Circle ref={gNodes[1]} x={432} y={-230} size={26} fill={PANEL} stroke={CYAN} lineWidth={4} />
      <Circle ref={gNodes[2]} x={516} y={-188} size={26} fill={PANEL} stroke={CYAN} lineWidth={4} />
      <Circle ref={gNodes[3]} x={594} y={-230} size={26} fill={PANEL} stroke={CYAN} lineWidth={4} />
      <Circle ref={gNodes[4]} x={603} y={-147} size={26} fill={PANEL} stroke={CYAN} lineWidth={4} />

      {/* bottom-left bad action panel */}
      <Img src={TREE} x={-565} y={120} width={145} />
      <Img ref={flyBad} src={FLY} x={-560} y={265} width={120} rotation={-35} />
      <Polygon
        ref={coneBad}
        points={[[0,-62],[-26,-8],[26,-8]]}
        x={-560}
        y={265}
        rotation={-35}
        fill={'rgba(255,109,105,0.15)'}
        stroke={RED}
        lineWidth={3}
        closed
      />
      <Line
        ref={arrowBad}
        points={[[-694,271],[-732,310],[-718,365],[-655,399]]}
        stroke={WHITE}
        lineWidth={4}
        lineCap={'round'}
      />
      <Txt ref={minusA} x={-350} y={195} text={'−'} fill={RED} fontSize={88} fontFamily={'JetBrains Mono, monospace'} opacity={0} />
      <Txt ref={minusB} x={-285} y={225} text={'−'} fill={RED} fontSize={56} fontFamily={'JetBrains Mono, monospace'} opacity={0} />

      {/* bottom-right punished brain */}
      <Img ref={badBrain} src={BRAIN} x={480} y={180} width={315} />
      <Line ref={bLines[0]} points={[[350,195],[432,228]]} stroke={DARKLINE} lineWidth={8} lineCap={'round'} />
      <Line ref={bLines[1]} points={[[432,228],[516,190]]} stroke={DARKLINE} lineWidth={8} lineCap={'round'} />
      <Line ref={bLines[2]} points={[[516,190],[594,228]]} stroke={DARKLINE} lineWidth={8} lineCap={'round'} />
      <Line ref={bLines[3]} points={[[594,228],[642,188]]} stroke={DARKLINE} lineWidth={8} lineCap={'round'} />
      <Circle ref={bNodes[0]} x={350} y={195} size={26} fill={PANEL} stroke={RED} lineWidth={4} />
      <Circle ref={bNodes[1]} x={432} y={228} size={26} fill={PANEL} stroke={RED} lineWidth={4} />
      <Circle ref={bNodes[2]} x={516} y={190} size={26} fill={PANEL} stroke={RED} lineWidth={4} />
      <Circle ref={bNodes[3]} x={594} y={228} size={26} fill={PANEL} stroke={RED} lineWidth={4} />
      <Circle ref={bNodes[4]} x={642} y={188} size={26} fill={PANEL} stroke={RED} lineWidth={4} />
    </>,
  );

  function* pulseGood() {
    yield* all(
      plusA().opacity(1, 0.18),
      plusA().scale(1.14, 0.18),
      plusB().opacity(1, 0.18),
      plusB().scale(1.14, 0.18),
      ...gLines.map(line => line().stroke(GOLD, 0.18)),
      ...gLines.map(line => line().lineWidth(12, 0.18)),
      ...gNodes.map(node => node().fill(GOLD, 0.18)),
    );
    yield* all(
      ...gLines.map(line => line().stroke(GREEN, 0.22)),
      ...gLines.map(line => line().lineWidth(18, 0.22)),
      ...gNodes.map(node => node().fill(GREEN, 0.22)),
    );
  }

  function* pulseBad() {
    yield* all(
      minusA().opacity(1, 0.18),
      minusA().scale(1.14, 0.18),
      minusB().opacity(1, 0.18),
      minusB().scale(1.14, 0.18),
      ...bLines.map(line => line().stroke(GOLD, 0.18)),
      ...bLines.map(line => line().lineWidth(12, 0.18)),
      ...bNodes.map(node => node().fill(GOLD, 0.18)),
    );
    yield* all(
      ...bLines.map(line => line().stroke(RED, 0.22)),
      ...bLines.map(line => line().lineWidth(3, 0.22)),
      ...bLines.map(line => line().opacity(0.4, 0.22)),
      ...bNodes.map(node => node().fill(RED, 0.22)),
      ...bNodes.map(node => node().opacity(0.5, 0.22)),
    );
  }

  // quiet intro
  yield* waitFor(0.45);

  // good action: rotate toward tree
  yield* all(
    flyGood().rotation(-28, 0.8, easeInOutCubic),
    coneGood().rotation(-28, 0.8, easeInOutCubic),
  );
  yield* pulseGood();
  yield* waitFor(0.45);

  // bad action: rotate away from tree
  yield* all(
    flyBad().rotation(120, 0.8, easeInOutCubic),
    coneBad().rotation(120, 0.8, easeInOutCubic),
  );
  yield* pulseBad();
  yield* waitFor(0.35);

  // final simple reinforcement beat on the good path
  yield* all(
    plusA().opacity(0.5, 0.15),
    plusB().opacity(0.5, 0.15),
    ...gLines.map(line => line().lineWidth(22, 0.2)),
    ...gLines.map(line => line().stroke(GREEN, 0.2)),
    ...gNodes.map(node => node().fill(GREEN, 0.2)),
  );
  yield* waitFor(1.0);
});
