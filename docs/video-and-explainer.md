# Video and explainer track

Alongside the simulation, the repo contains a small **YouTube/video-production**
track. None of it reads simulation state, checkpoints or metrics — it is a
presentation layer bundled in the same repository.

---

## 1. Pictogram interpreter animation

`flycraft_interpreter.py` is a deterministic, **wordless** narration animation
built with pygame. It draws at a logical 1600x900 canvas and delegates all
character/scene drawing to `flycraft_pictograms.py`. Frame state is derived
purely from animation time (frame-rate independent).

Seven scenes, ~80 seconds total:

| # | Title | Caption | Length |
|---|---|---|---|
| 1 | The learning hit a ceiling | 01 / THE PLATEAU | 7.0 s |
| 2 | Which connections actually helped? | 02 / THE CREDIT PROBLEM | 9.0 s |
| 3 | The reward system was guessing | 03 / THE WRONG CONNECTIONS | 7.0 s |
| 4 | Keep the brain. Learn around it. | 04 / A NEW APPROACH | 15.0 s |
| 5 | A little interpreter turns activity into action | 05 / THE ROBOT | 21.0 s |
| 6 | Learn to predict an expert | 06 / THE TEACHER | 12.0 s |
| 7 | The interpreter only sees the brain | 07 / THE INPUT BOUNDARY | 9.0 s |

### CLI

```powershell
# Interactive window (keys 1-7 scenes, A/R restart, Space pause, arrows seek, H help, F11 fullscreen)
& $python flycraft_interpreter.py

# Render the verification PNGs (3 per scene + contact sheet)
& $python flycraft_interpreter.py --self-test
& $python flycraft_interpreter.py --preview

# Export a 1080p60 silent MP4
& $python flycraft_interpreter.py --export artifacts/interpreter-quality/flycraft-pictograms.mp4 --fps 60
```

`--output` (default `artifacts/interpreter-quality`) controls where verification
PNGs are written. MP4 export uses H.264/libx264, CRF 17, 1920x1080, silent, with
`+faststart`, via `ffmpeg` on PATH or the bundled `imageio_ffmpeg`.

### Launcher and aliases

- `run-learning-visualizer.cmd` runs `flycraft_learning_loop_v6.py`, which is a
  5-line compatibility shim forwarding to `flycraft_interpreter.main`.
- `flycraft_learning_loop_v6_old.py` + `run-learning-visualizer-old.cmd` are the
  previous standalone "learning visualizer" (arena + brain panels, staged
  idle/reward/punish/strengthen/weaken animation). It reuses
  `flycraft_pictograms` for artwork and is kept only as an archived alternative.

### `flycraft_pictograms.py`

A wordless, code-drawn character/animation library (no image assets). It exposes
`render(index, t, duration)` returning a 1600x900 `pygame.Surface`, plus the
drawing primitives and pose helpers used by the interpreter's self-test.

---

## 2. Motion Canvas overlay project

`motion-canvas/` is a [Motion Canvas](https://motioncanvas.io) v3.17.2
TypeScript/Vite project (`fly-synapse-overlay-motion-canvas`). It is a
deliberately minimal transparent overlay for the narration line:

> "If I wanted the fly to learn, I should modify the connections inside its
> brain."

```powershell
cd motion-canvas
npm install
npm start        # or: npm run dev  ->  vite --host 0.0.0.0
```

- 1920x1080, transparent background, 30 fps preview / 60 fps render.
- Exporter: image-sequence PNG (`output/project/`).
- The project currently wires in only the `synapse` scene (`src/project.ts`).
  Additional scenes (`learning`, `learning-cutout`, `learning-dark-cutout`) are
  present but not registered.
- Assets live in `motion-canvas/public/assets/`.
- Rendering is done through the Motion Canvas UI; there is no npm build/render
  script.

`motion-canvas/README.md` describes the intended use over darkened
FlyCraft/dashboard footage.

---

## 3. Relationship to the simulation

- Nothing in `src/flycraft/` imports these scripts or the motion-canvas project.
- The animation is illustrative: neural graphics show concepts, not measured
  training results.
- The exported MP4 is silent; narration is added during editing.
