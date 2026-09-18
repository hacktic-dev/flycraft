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

const BG = '#07111F';
const BG2 = '#0B1830';
const CYAN = '#41D7EE';
const CYAN_DARK = '#1EAAC5';
const GOLD = '#F0C45E';
const GREEN = '#71F0C0';
const RED = '#FF6C67';
const ORANGE = '#F7A14E';
const DIM = '#2A3E5C';
const DIM2 = '#1A2740';

const FLY = '/assets/fly.svg';
const BRAIN = '/assets/brain.svg';
const TREE = '/assets/tree.svg';
const REWARD = '/assets/reward_block.svg';
const PUNISH = '/assets/punish_block.svg';

export default makeScene2D(function* (view) {
  view.fill(BG);

  const bgGlow1 = createRef<Circle>();
  const bgGlow2 = createRef<Circle>();

  const fly = createRef<Img>();
  const tree = createRef<Img>();
  const brain = createRef<Img>();

  const sensory = createRef<Line>();
  const teaching = createRef<Line>();
  const pulse = createRef<Circle>();

  const reward = createRef<Img>();
  const punish = createRef<Img>();

  const overlay = createRef<Node>();
  const nodes = Array.from({length: 6}, () => createRef<Circle>());
  const goodPath = Array.from({length: 4}, () => createRef<Line>());
  const badPath = Array.from({length: 3}, () => createRef<Line>());
  const bgPath = Array.from({length: 5}, () => createRef<Line>());

  view.add(
    <>
      <Rect width={1920} height={1080} fill={BG} />
      <Circle ref={bgGlow1} x={-560} y={-280} size={700} fill={CYAN} opacity={0.05} blur={120}/>
      <Circle ref={bgGlow2} x={620} y={260} size={820} fill={ORANGE} opacity={0.03} blur={160}/>

      <Img ref={tree} src={TREE} x={650} y={-215} width={235} opacity={0} scale={0.78} rotation={4}/>
      <Img ref={brain} src={BRAIN} x={170} y={170} width={600} opacity={0} scale={0.82} rotation={-5}/>
      <Img ref={fly} src={FLY} x={-590} y={110} width={310} opacity={0} scale={0.8} rotation={180}/>

      <Line
        ref={sensory}
        points={[[-440, 82], [-275, 30], [-70, 58], [85, 115]]}
        stroke={CYAN}
        lineWidth={7}
        lineDash={[22, 18]}
        lineCap={'round'}
        opacity={0}
        end={0}
      />

      <Node ref={overlay} x={170} y={165} opacity={0}>
        <Line ref={bgPath[0]} points={[[-180,-44],[-72,-8]]} stroke={DIM} lineWidth={6} lineCap={'round'} />
        <Line ref={bgPath[1]} points={[[-72,-8],[48,-52]]} stroke={DIM} lineWidth={6} lineCap={'round'} />
        <Line ref={bgPath[2]} points={[[48,-52],[170,6]]} stroke={DIM} lineWidth={6} lineCap={'round'} />
        <Line ref={bgPath[3]} points={[[-165,72],[-35,102]]} stroke={DIM2} lineWidth={6} lineCap={'round'} />
        <Line ref={bgPath[4]} points={[[-35,102],[100,74]]} stroke={DIM2} lineWidth={6} lineCap={'round'} />

        <Line ref={goodPath[0]} points={[[-180,-44],[-72,-8]]} stroke={DIM} lineWidth={6} lineCap={'round'} />
        <Line ref={goodPath[1]} points={[[-72,-8],[48,-52]]} stroke={DIM} lineWidth={6} lineCap={'round'} />
        <Line ref={goodPath[2]} points={[[48,-52],[170,6]]} stroke={DIM} lineWidth={6} lineCap={'round'} />
        <Line ref={goodPath[3]} points={[[170,6],[235,-46]]} stroke={DIM} lineWidth={6} lineCap={'round'} />

        <Line ref={badPath[0]} points={[[-165,72],[-35,102]]} stroke={DIM2} lineWidth={6} lineCap={'round'} />
        <Line ref={badPath[1]} points={[[-35,102],[100,74]]} stroke={DIM2} lineWidth={6} lineCap={'round'} />
        <Line ref={badPath[2]} points={[[100,74],[165,138]]} stroke={DIM2} lineWidth={6} lineCap={'round'} />

        <Circle ref={nodes[0]} x={-180} y={-44} size={28} fill={BG2} stroke={CYAN} lineWidth={5} />
        <Circle ref={nodes[1]} x={-72} y={-8} size={28} fill={BG2} stroke={CYAN} lineWidth={5} />
        <Circle ref={nodes[2]} x={48} y={-52} size={28} fill={BG2} stroke={CYAN} lineWidth={5} />
        <Circle ref={nodes[3]} x={170} y={6} size={28} fill={BG2} stroke={CYAN} lineWidth={5} />
        <Circle ref={nodes[4]} x={-35} y={102} size={28} fill={BG2} stroke={RED} lineWidth={5} />
        <Circle ref={nodes[5]} x={165} y={138} size={28} fill={BG2} stroke={RED} lineWidth={5} />
      </Node>

      <Img ref={reward} src={REWARD} x={-505} y={-175} width={150} opacity={0} scale={0.2} rotation={-10}/>
      <Img ref={punish} src={PUNISH} x={-510} y={-170} width={150} opacity={0} scale={0.2} rotation={12}/>

      <Line
        ref={teaching}
        points={[[-470, 12], [-270, 8], [-70, 70], [102, 126]]}
        stroke={GREEN}
        lineWidth={8}
        lineDash={[18, 16]}
        lineCap={'round'}
        opacity={0}
        end={0}
      />

      <Circle
        ref={pulse}
        size={32}
        fill={GREEN}
        stroke={CYAN}
        lineWidth={8}
        opacity={0}
        shadowBlur={24}
        shadowColor={GREEN}
      />
    </>,
  );

  // 1. Establish objects
  yield* sequence(
    0.1,
    all(tree().opacity(1, 0.28), tree().scale(1, 0.42, easeOutBack)),
    all(brain().opacity(1, 0.3), brain().scale(1, 0.45, easeOutBack)),
    all(fly().opacity(1, 0.25), fly().scale(1, 0.4, easeOutBack)),
  );
  yield* overlay().opacity(1, 0.25);
  yield* waitFor(0.4);

  // Sensory path from fly to brain
  yield* sensory().opacity(0.7, 0.14);
  yield* sensory().end(1, 0.75, easeOutCubic);
  yield* waitFor(0.35);

  // 2. Good action - fly visibly turns from LEFT to RIGHT, towards the tree
  yield* all(
    fly().rotation(25, 0.85, easeInOutCubic),
    fly().position([-520, 60], 0.85, easeInOutCubic),
  );
  yield* waitFor(0.2);

  // Reward cube
  reward().position([-490, -150]);
  reward().scale(0.2);
  reward().opacity(0);
  yield* all(
    reward().opacity(1, 0.12),
    reward().scale(1, 0.32, easeOutBack),
    reward().rotation(0, 0.32),
  );
  yield* reward().position([-460, -18], 0.3, easeInOutCubic);
  yield* all(
    reward().opacity(0, 0.14),
    reward().scale(0.18, 0.14),
    fly().scale(1.05, 0.08).to(1, 0.14),
  );

  teaching().stroke(GREEN);
  teaching().end(0);
  yield* teaching().opacity(0.9, 0.12);
  yield* teaching().end(1, 0.55, easeOutCubic);

  pulse().fill(GREEN);
  pulse().stroke(CYAN);
  pulse().position([-470, 10]);
  yield* pulse().opacity(1, 0.08);
  yield* pulse().position([112, 128], 0.56, easeInOutCubic);
  yield* pulse().opacity(0, 0.08);

  yield* all(
    goodPath[0]().stroke(CYAN, 0.24), goodPath[1]().stroke(CYAN, 0.24),
    goodPath[2]().stroke(CYAN, 0.24), goodPath[3]().stroke(CYAN, 0.24),
    goodPath[0]().lineWidth(12, 0.24), goodPath[1]().lineWidth(12, 0.24),
    goodPath[2]().lineWidth(12, 0.24), goodPath[3]().lineWidth(12, 0.24),
    nodes[0]().fill(CYAN, 0.2), nodes[1]().fill(CYAN, 0.2),
    nodes[2]().fill(CYAN, 0.2), nodes[3]().fill(CYAN, 0.2),
  );
  yield* waitFor(0.9);

  // 3. Bad action - reset and turn away from tree
  yield* all(
    teaching().opacity(0, 0.15),
    goodPath[0]().stroke(DIM, 0.2), goodPath[1]().stroke(DIM, 0.2),
    goodPath[2]().stroke(DIM, 0.2), goodPath[3]().stroke(DIM, 0.2),
    goodPath[0]().lineWidth(6, 0.2), goodPath[1]().lineWidth(6, 0.2),
    goodPath[2]().lineWidth(6, 0.2), goodPath[3]().lineWidth(6, 0.2),
    nodes[0]().fill(BG2, 0.2), nodes[1]().fill(BG2, 0.2),
    nodes[2]().fill(BG2, 0.2), nodes[3]().fill(BG2, 0.2),
  );

  yield* all(
    fly().rotation(210, 0.8, easeInOutCubic),
    fly().position([-600, 130], 0.8, easeInOutCubic),
  );
  yield* waitFor(0.14);

  punish().position([-500, -160]);
  punish().scale(0.2);
  punish().opacity(0);
  yield* all(
    punish().opacity(1, 0.12),
    punish().scale(1, 0.28, easeOutBack),
  );
  yield* punish().position([-485, -12], 0.24, easeInOutCubic);
  yield* sequence(
    0.03,
    fly().x(-615, 0.05),
    fly().x(-590, 0.05),
    fly().x(-610, 0.05),
    fly().x(-595, 0.05),
    fly().x(-600, 0.06),
  );
  yield* all(
    punish().opacity(0, 0.1),
    punish().scale(0.15, 0.1),
  );

  teaching().stroke(RED);
  teaching().end(0);
  yield* teaching().opacity(0.9, 0.12);
  yield* teaching().end(1, 0.55, easeOutCubic);

  pulse().fill(RED);
  pulse().stroke(ORANGE);
  pulse().position([-470, 10]);
  yield* pulse().opacity(1, 0.08);
  yield* pulse().position([112, 128], 0.56, easeInOutCubic);
  yield* pulse().opacity(0, 0.08);

  yield* all(
    badPath[0]().stroke(RED, 0.18),
    badPath[1]().stroke(RED, 0.18),
    badPath[2]().stroke(RED, 0.18),
    badPath[0]().lineWidth(11, 0.18),
    badPath[1]().lineWidth(11, 0.18),
    badPath[2]().lineWidth(11, 0.18),
    nodes[4]().fill(RED, 0.18),
    nodes[5]().fill(RED, 0.18),
  );
  yield* all(
    badPath[0]().opacity(0.25, 0.42),
    badPath[1]().opacity(0.25, 0.42),
    badPath[2]().opacity(0.25, 0.42),
    badPath[0]().lineWidth(2.5, 0.42),
    badPath[1]().lineWidth(2.5, 0.42),
    badPath[2]().lineWidth(2.5, 0.42),
  );
  yield* waitFor(0.8);

  // 4. Repeated rewarded attempts - a bit rougher / more tactile
  yield* all(
    teaching().opacity(0, 0.12),
    nodes[4]().fill(BG2, 0.18),
    nodes[5]().fill(BG2, 0.18),
    badPath[0]().stroke(DIM2, 0.18),
    badPath[1]().stroke(DIM2, 0.18),
    badPath[2]().stroke(DIM2, 0.18),
  );

  const poses = [
    {rot: 120, pos: [-575, 115], width: 7},
    {rot: 70, pos: [-550, 90], width: 10},
    {rot: 25, pos: [-520, 60], width: 15},
  ];

  for (const pose of poses) {
    yield* all(
      fly().rotation(pose.rot, 0.34, easeInOutCubic),
      fly().position(pose.pos, 0.34, easeInOutCubic),
    );

    reward().position([-490, -115]);
    reward().scale(0.24);
    reward().opacity(0);
    yield* all(
      reward().opacity(1, 0.08),
      reward().scale(0.75, 0.18, easeOutBack),
    );
    yield* reward().position([-468, -18], 0.18, easeInOutCubic);
    yield* all(reward().opacity(0, 0.08), reward().scale(0.14, 0.08));

    yield* all(
      goodPath[0]().stroke(CYAN, 0.18), goodPath[1]().stroke(CYAN, 0.18),
      goodPath[2]().stroke(CYAN, 0.18), goodPath[3]().stroke(CYAN, 0.18),
      goodPath[0]().lineWidth(pose.width, 0.18), goodPath[1]().lineWidth(pose.width, 0.18),
      goodPath[2]().lineWidth(pose.width, 0.18), goodPath[3]().lineWidth(pose.width, 0.18),
      nodes[0]().fill(CYAN, 0.18), nodes[1]().fill(CYAN, 0.18),
      nodes[2]().fill(CYAN, 0.18), nodes[3]().fill(CYAN, 0.18),
    );
    yield* waitFor(0.18);
  }

  yield* all(
    fly().rotation(25, 0.3, easeInOutCubic),
    fly().position([-520, 60], 0.3, easeInOutCubic),
    goodPath[0]().lineWidth(16, 0.25), goodPath[1]().lineWidth(16, 0.25),
    goodPath[2]().lineWidth(16, 0.25), goodPath[3]().lineWidth(16, 0.25),
  );
  yield* waitFor(1.2);
});
