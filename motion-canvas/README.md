# Fly synapse overlay — Motion Canvas

A deliberately minimal transparent overlay for the narration:

> “If I wanted the fly to learn, I should modify the connections inside its brain.”

It contains only a tiny neural network. One pair of neurons becomes active and
the connection between them visibly changes from a thin muted synapse into a
thick gold synapse.

There is **no background** in the Motion Canvas scene, so it is intended to sit
over darkened FlyCraft/dashboard footage.

## Run

```powershell
cd "fly-synapse-overlay-motion-canvas"
npm install
npm start
```

Main file:

```text
src/scenes/synapse.tsx
```

## Important for transparency

Render/export to a format that preserves alpha. A PNG image sequence is the
simplest reliable option; import the sequence into Resolve and place it above
the darkened dashboard footage.

Do not add `view.fill(...)` to the scene unless you intentionally want a
background.

## Easy positioning

The whole graphic is inside:

```tsx
<Node ref={overlay} x={0} y={0}>
```

Change `x`, `y`, or `scale` on that node to place it over whichever part of the
dashboard you want.
