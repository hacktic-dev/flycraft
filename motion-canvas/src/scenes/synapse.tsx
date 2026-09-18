import {
  Circle,
  Line,
  Node,
  makeScene2D,
} from '@motion-canvas/2d';
import {
  all,
  createRef,
  easeInOutCubic,
  easeOutBack,
  sequence,
  waitFor,
} from '@motion-canvas/core';

const CYAN = '#45D7EE';
const MUTED = '#5E7697';
const GOLD = '#F4C95D';

export default makeScene2D(function* (view) {
  // IMPORTANT: no view.fill() call here.
  // The scene stays transparent so it can be rendered as an alpha overlay.

  const overlay = createRef<Node>();

  const n0 = createRef<Circle>();
  const n1 = createRef<Circle>();
  const n2 = createRef<Circle>();
  const n3 = createRef<Circle>();
  const n4 = createRef<Circle>();
  const n5 = createRef<Circle>();

  const l01 = createRef<Line>();
  const l12 = createRef<Line>();
  const l23 = createRef<Line>();
  const l24 = createRef<Line>();
  const l35 = createRef<Line>();
  const l45 = createRef<Line>();

  // Duplicate of the connection we "modify" so we can create a soft glow
  // without changing the rest of the network.
  const activeGlow = createRef<Line>();

  view.add(
    <Node
      ref={overlay}
      x={0}
      y={0}
      opacity={1}
    >
      {/* Base network */}
      <Line ref={l01} points={[[-360, -90], [-210, -160]]} stroke={MUTED} lineWidth={8} lineCap={'round'} end={0}/>
      <Line ref={l12} points={[[-210, -160], [-40, -80]]} stroke={MUTED} lineWidth={8} lineCap={'round'} end={0}/>
      <Line ref={l23} points={[[-40, -80], [150, -155]]} stroke={MUTED} lineWidth={8} lineCap={'round'} end={0}/>
      <Line ref={l24} points={[[-40, -80], [140, 45]]} stroke={MUTED} lineWidth={8} lineCap={'round'} end={0}/>
      <Line ref={l35} points={[[150, -155], [330, -55]]} stroke={MUTED} lineWidth={8} lineCap={'round'} end={0}/>
      <Line ref={l45} points={[[140, 45], [330, -55]]} stroke={MUTED} lineWidth={8} lineCap={'round'} end={0}/>

      {/* This sits directly on top of l24 and becomes the "stronger connection". */}
      <Line
        ref={activeGlow}
        points={[[-40, -80], [140, 45]]}
        stroke={GOLD}
        lineWidth={8}
        lineCap={'round'}
        opacity={0}
      />

      <Circle ref={n0} x={-360} y={-90} size={46} fill={'#07111F'} stroke={CYAN} lineWidth={6} scale={0}/>
      <Circle ref={n1} x={-210} y={-160} size={46} fill={'#07111F'} stroke={CYAN} lineWidth={6} scale={0}/>
      <Circle ref={n2} x={-40} y={-80} size={52} fill={'#07111F'} stroke={CYAN} lineWidth={6} scale={0}/>
      <Circle ref={n3} x={150} y={-155} size={46} fill={'#07111F'} stroke={CYAN} lineWidth={6} scale={0}/>
      <Circle ref={n4} x={140} y={45} size={46} fill={'#07111F'} stroke={CYAN} lineWidth={6} scale={0}/>
      <Circle ref={n5} x={330} y={-55} size={50} fill={'#07111F'} stroke={CYAN} lineWidth={6} scale={0}/>
    </Node>,
  );

  // 1) The small network appears over the dashboard footage.
  yield* sequence(
    0.06,
    n0().scale(1, 0.30, easeOutBack),
    n1().scale(1, 0.30, easeOutBack),
    n2().scale(1, 0.30, easeOutBack),
    n3().scale(1, 0.30, easeOutBack),
    n4().scale(1, 0.30, easeOutBack),
    n5().scale(1, 0.30, easeOutBack),
  );

  yield* all(
    l01().end(1, 0.55, easeInOutCubic),
    l12().end(1, 0.55, easeInOutCubic),
    l23().end(1, 0.55, easeInOutCubic),
    l24().end(1, 0.55, easeInOutCubic),
    l35().end(1, 0.55, easeInOutCubic),
    l45().end(1, 0.55, easeInOutCubic),
  );

  yield* waitFor(0.40);

  // 2) "Connections that had just been active" — two neurons pulse.
  yield* all(
    n2().fill(GOLD, 0.18),
    n4().fill(GOLD, 0.18),
    n2().scale(1.18, 0.18).to(1, 0.18),
    n4().scale(1.18, 0.18).to(1, 0.18),
  );

  yield* waitFor(0.18);

  // 3) The synapse itself visibly changes strength.
  yield* all(
    activeGlow().opacity(1, 0.18),
    activeGlow().lineWidth(20, 0.55, easeInOutCubic),
    l24().stroke(GOLD, 0.35),
    n2().stroke(GOLD, 0.35),
    n4().stroke(GOLD, 0.35),
  );

  yield* waitFor(1.10);

  // 4) Small second pulse to make the idea of "strengthening" unmistakable.
  yield* all(
    activeGlow().lineWidth(26, 0.24),
    n2().scale(1.12, 0.16).to(1, 0.16),
    n4().scale(1.12, 0.16).to(1, 0.16),
  );

  yield* waitFor(1.00);

  // Gentle fade so you can cut back to the untouched dashboard.
  yield* overlay().opacity(0, 0.35);
});
