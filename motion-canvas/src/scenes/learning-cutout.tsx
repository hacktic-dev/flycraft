import {
  Circle,
  Img,
  Line,
  Node,
  Rect,
  makeScene2D,
} from '@motion-canvas/2d';
import {
  all,
  createRef,
  easeInOutCubic,
  easeOutBack,
  easeOutCubic,
  sequence,
  waitFor,
} from '@motion-canvas/core';

const PAPER = '#F2F0EA';
const GRID = '#C8C5BD';
const INK = '#242526';
const GREEN = '#53C98C';
const GREEN_DARK = '#279D67';
const RED = '#EF5550';
const BLUE = '#55CDE0';
const GOLD = '#E6B64A';
const FAINT = '#9A968E';

// Replace these files in /public/assets and keep the filenames.
// Transparent PNGs work too; just update the extension here if needed.
const FLY = '/assets/fly.svg';
const BRAIN = '/assets/brain.svg';
const TREE = '/assets/tree.svg';
const REWARD = '/assets/reward.svg';
const PUNISH = '/assets/punish.svg';

const goodAngle = -14;   // toward the tree (up/right)
const badAngle = 154;    // clearly away from the tree

export default makeScene2D(function* (view) {
  view.fill(PAPER);

  const fly = createRef<Img>();
  const brain = createRef<Img>();
  const tree = createRef<Img>();
  const reward = createRef<Img>();
  const punish = createRef<Img>();

  const sensoryPath = createRef<Line>();
  const rewardPath = createRef<Line>();
  const signal = createRef<Circle>();

  // A tiny schematic drawn over the placeholder brain.
  // It remains separate from the brain artwork, so replacing brain.svg won't
  // break the animation.
  const brainOverlay = createRef<Node>();
  const n1 = createRef<Circle>();
  const n2 = createRef<Circle>();
  const n3 = createRef<Circle>();
  const n4 = createRef<Circle>();
  const n5 = createRef<Circle>();
  const good1 = createRef<Line>();
  const good2 = createRef<Line>();
  const good3 = createRef<Line>();
  const bad1 = createRef<Line>();
  const bad2 = createRef<Line>();

  // A clean paper/grid background like a tabletop explainer.
  view.add(
    <>
      {[-840, -630, -420, -210, 0, 210, 420, 630, 840].map(x => (
        <Line
          points={[[x, -540], [x, 540]]}
          stroke={GRID}
          lineWidth={2}
          opacity={0.45}
        />
      ))}
      {[-420, -210, 0, 210, 420].map(y => (
        <Line
          points={[[-960, y], [960, y]]}
          stroke={GRID}
          lineWidth={2}
          opacity={0.45}
        />
      ))}

      <Img
        ref={tree}
        src={TREE}
        x={610}
        y={-225}
        width={230}
        opacity={0}
        scale={0.7}
        rotation={3}
      />

      <Img
        ref={fly}
        src={FLY}
        x={-620}
        y={105}
        width={360}
        opacity={0}
        scale={0.7}
        rotation={82}
      />

      <Line
        ref={sensoryPath}
        points={[
          [-500, 75],
          [-330, -25],
          [-120, 50],
          [80, 125],
        ]}
        stroke={INK}
        lineWidth={7}
        lineDash={[18, 16]}
        lineCap={'round'}
        opacity={0}
        end={0}
      />

      <Img
        ref={brain}
        src={BRAIN}
        x={260}
        y={145}
        width={650}
        opacity={0}
        scale={0.75}
        rotation={-2}
      />

      <Node ref={brainOverlay} x={260} y={150} opacity={0}>
        <Line ref={good1} points={[[-170, -55], [-55, -5]]} stroke={FAINT} lineWidth={7} lineCap={'round'} />
        <Line ref={good2} points={[[ -55, -5], [65, -55]]} stroke={FAINT} lineWidth={7} lineCap={'round'} />
        <Line ref={good3} points={[[65, -55], [175, 20]]} stroke={FAINT} lineWidth={7} lineCap={'round'} />

        <Line ref={bad1} points={[[-165, 60], [-45, 92]]} stroke={FAINT} lineWidth={7} lineCap={'round'} />
        <Line ref={bad2} points={[[-45, 92], [90, 70]]} stroke={FAINT} lineWidth={7} lineCap={'round'} />

        <Circle ref={n1} x={-170} y={-55} size={28} fill={PAPER} stroke={INK} lineWidth={5}/>
        <Circle ref={n2} x={-55} y={-5} size={28} fill={PAPER} stroke={INK} lineWidth={5}/>
        <Circle ref={n3} x={65} y={-55} size={28} fill={PAPER} stroke={INK} lineWidth={5}/>
        <Circle ref={n4} x={175} y={20} size={28} fill={PAPER} stroke={INK} lineWidth={5}/>
        <Circle ref={n5} x={-45} y={92} size={28} fill={PAPER} stroke={INK} lineWidth={5}/>
      </Node>

      <Img
        ref={reward}
        src={REWARD}
        x={-520}
        y={-170}
        width={175}
        opacity={0}
        scale={0.25}
        rotation={-8}
      />

      <Img
        ref={punish}
        src={PUNISH}
        x={-540}
        y={-175}
        width={155}
        opacity={0}
        scale={0.25}
        rotation={8}
      />

      <Line
        ref={rewardPath}
        points={[
          [-475, 10],
          [-285, 0],
          [-125, 70],
          [90, 120],
        ]}
        stroke={GREEN_DARK}
        lineWidth={8}
        lineDash={[16, 15]}
        lineCap={'round'}
        opacity={0}
        end={0}
      />

      <Circle
        ref={signal}
        size={34}
        fill={GREEN}
        stroke={'#FFFFFF'}
        lineWidth={7}
        opacity={0}
      />
    </>,
  );

  // -------------------------------------------------------------------------
  // 1. Establish: fly, tree, brain. No labels; narration does the explaining.
  // -------------------------------------------------------------------------
  yield* sequence(
    0.16,
    all(tree().opacity(1, 0.35), tree().scale(1, 0.5, easeOutBack)),
    all(fly().opacity(1, 0.35), fly().scale(1, 0.5, easeOutBack)),
    all(brain().opacity(1, 0.35), brain().scale(1, 0.5, easeOutBack)),
  );
  yield* brainOverlay().opacity(0.72, 0.35);
  yield* waitFor(0.65);

  // Signal from what the fly sees into the brain.
  yield* sensoryPath().opacity(0.72, 0.25);
  yield* sensoryPath().end(1, 0.85, easeOutCubic);
  yield* waitFor(0.45);

  // -------------------------------------------------------------------------
  // 2. GOOD ACTION: the fly literally turns TOWARD the tree.
  // -------------------------------------------------------------------------
  yield* fly().rotation(goodAngle, 0.9, easeInOutCubic);
  yield* waitFor(0.25);

  // Literal reward: an apple pops down next to the fly and gets "consumed".
  reward().position([-515, -145]);
  yield* all(
    reward().opacity(1, 0.18),
    reward().scale(1, 0.38, easeOutBack),
    reward().rotation(4, 0.38, easeOutBack),
  );
  yield* reward().position([-500, 15], 0.48, easeInOutCubic);
  yield* all(
    reward().scale(0.22, 0.22),
    reward().opacity(0, 0.22),
    fly().scale(1.07, 0.12).to(1, 0.18),
  );

  // Reward signal travels from fly to brain.
  rewardPath().stroke(GREEN_DARK);
  rewardPath().end(0);
  yield* rewardPath().opacity(0.82, 0.18);
  yield* rewardPath().end(1, 0.62, easeOutCubic);

  signal().fill(GREEN);
  signal().position([-470, 5]);
  yield* signal().opacity(1, 0.08);
  yield* signal().position([110, 112], 0.62, easeInOutCubic);
  yield* signal().opacity(0, 0.12);

  // The recently active path strengthens.
  yield* all(
    good1().stroke(GREEN_DARK, 0.32),
    good2().stroke(GREEN_DARK, 0.32),
    good3().stroke(GREEN_DARK, 0.32),
    good1().lineWidth(13, 0.32),
    good2().lineWidth(13, 0.32),
    good3().lineWidth(13, 0.32),
    n1().fill(GREEN, 0.25),
    n2().fill(GREEN, 0.25),
    n3().fill(GREEN, 0.25),
    n4().fill(GREEN, 0.25),
  );
  yield* waitFor(1.0);

  // -------------------------------------------------------------------------
  // 3. BAD ACTION: reset and turn clearly AWAY from the tree.
  // -------------------------------------------------------------------------
  yield* all(
    rewardPath().opacity(0, 0.2),
    good1().stroke(FAINT, 0.25),
    good2().stroke(FAINT, 0.25),
    good3().stroke(FAINT, 0.25),
    good1().lineWidth(7, 0.25),
    good2().lineWidth(7, 0.25),
    good3().lineWidth(7, 0.25),
    n1().fill(PAPER, 0.25),
    n2().fill(PAPER, 0.25),
    n3().fill(PAPER, 0.25),
    n4().fill(PAPER, 0.25),
  );

  yield* fly().rotation(badAngle, 0.9, easeInOutCubic);
  yield* waitFor(0.2);

  // Literal punishment: red lightning drops onto the fly, with a tiny shake.
  punish().position([-520, -185]);
  yield* all(
    punish().opacity(1, 0.15),
    punish().scale(1, 0.3, easeOutBack),
  );
  yield* punish().position([-500, -10], 0.32, easeInOutCubic);
  yield* sequence(
    0.05,
    fly().x(-635, 0.06),
    fly().x(-605, 0.06),
    fly().x(-630, 0.06),
    fly().x(-610, 0.06),
    fly().x(-620, 0.08),
  );
  yield* all(punish().opacity(0, 0.18), punish().scale(0.3, 0.18));

  // A red teaching signal reaches the brain.
  rewardPath().stroke(RED);
  rewardPath().end(0);
  yield* rewardPath().opacity(0.82, 0.16);
  yield* rewardPath().end(1, 0.62, easeOutCubic);

  signal().fill(RED);
  signal().position([-470, 5]);
  yield* signal().opacity(1, 0.08);
  yield* signal().position([110, 112], 0.62, easeInOutCubic);
  yield* signal().opacity(0, 0.12);

  // A different active path gets weaker instead.
  yield* all(
    bad1().stroke(RED, 0.18),
    bad2().stroke(RED, 0.18),
    bad1().lineWidth(11, 0.18),
    bad2().lineWidth(11, 0.18),
  );
  yield* all(
    bad1().opacity(0.22, 0.55),
    bad2().opacity(0.22, 0.55),
    bad1().lineWidth(2.5, 0.55),
    bad2().lineWidth(2.5, 0.55),
  );
  yield* waitFor(0.85);

  // -------------------------------------------------------------------------
  // 4. "So in theory..." — repeated reward makes the useful path dominant.
  //    Still almost entirely visual: three fast reward cycles.
  // -------------------------------------------------------------------------
  yield* rewardPath().opacity(0, 0.2);
  yield* fly().rotation(55, 0.35);

  const attempts = [
    {angle: 22, width: 8},
    {angle: 2, width: 11},
    {angle: goodAngle, width: 16},
  ];

  for (const attempt of attempts) {
    yield* fly().rotation(attempt.angle, 0.44, easeInOutCubic);

    reward().position([-510, -95]);
    reward().scale(0.32);
    reward().opacity(0);
    yield* all(
      reward().opacity(1, 0.10),
      reward().scale(0.68, 0.18, easeOutBack),
    );
    yield* reward().position([-495, 5], 0.22, easeInOutCubic);
    yield* all(reward().opacity(0, 0.12), reward().scale(0.18, 0.12));

    yield* all(
      good1().stroke(GREEN_DARK, 0.20),
      good2().stroke(GREEN_DARK, 0.20),
      good3().stroke(GREEN_DARK, 0.20),
      good1().lineWidth(attempt.width, 0.20),
      good2().lineWidth(attempt.width, 0.20),
      good3().lineWidth(attempt.width, 0.20),
      n1().fill(GREEN, 0.18),
      n2().fill(GREEN, 0.18),
      n3().fill(GREEN, 0.18),
      n4().fill(GREEN, 0.18),
    );
    yield* waitFor(0.22);
  }

  // Final readable state: fly aimed at the tree, strong green route in brain.
  yield* all(
    fly().rotation(goodAngle, 0.4, easeInOutCubic),
    good1().lineWidth(17, 0.3),
    good2().lineWidth(17, 0.3),
    good3().lineWidth(17, 0.3),
    good1().stroke(GREEN_DARK, 0.3),
    good2().stroke(GREEN_DARK, 0.3),
    good3().stroke(GREEN_DARK, 0.3),
  );
  yield* waitFor(1.5);
});
