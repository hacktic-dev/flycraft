# Troubleshooting

## PowerShell refuses to run scripts

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then rerun the command.

## `Run .\setup.ps1 first.`

The project-local Python environment is missing:

```powershell
.\setup.ps1
```

## `Missing genuine breaking telemetry`

Training requires the CraftGround runtime patch that exposes real block-breaking
progress. `train.ps1`/`train-readout.ps1` normally prepare it automatically. Run
it manually if needed, then restart Minecraft so Gradle rebuilds:

```powershell
. .\env.ps1
& $python scripts\prepare_training_runtime.py
```

## Module import errors (`No module named 'flycraft'` / `'doom'`)

You are not in the project environment. Load it first (wrappers do this for
you):

```powershell
. .\env.ps1
```

`env.ps1` sets `PYTHONPATH` to `src` and `vendor/doomfly` and points at the local
`.venv`.

## R8 `TypeError: only size-1 arrays can be converted to Python scalars`

The R8 adapter feeds vector-valued currents. This was fixed by the v6 adapter
(`R8VisualMemoryBrain`). Verify:

```powershell
. .\env.ps1
& $python scripts\verify_r8_input.py
```

## White/blank Minecraft window

`scripts/prepare_runtime.py` presents the real framebuffer and matches the
window aspect ratio. Re-run it and close any running Minecraft client:

```powershell
. .\env.ps1
& $python scripts\prepare_runtime.py
```

## `Decoder differs from checkpoint; use an explicit Branch or Evaluate configuration, not Resume`

`Resume` refuses to change the decoder because it would reset the decoder's
filter state. To compare decoders, use `-Mode Branch` (new timeline) or `-Mode
Evaluate` (frozen test).

## `Checkpoint is immutable`

Numbered/named checkpoints are never overwritten. Use a different label, or
start a `Branch`/`Fresh` run.

## `Plasticity collapse threshold reached; abort_on_collapse=true`

The KC→MBON11 plastic population saturated or drifted past the configured
thresholds. Inspect the last `PLASTICITY ...` lines and metrics before continuing.
You can raise the alarm/abort thresholds or disable `abort_on_collapse` in the
config, but investigate first — collapse means the rule is no longer learning
usefully.

## Which checkpoint does `latest` mean?

```powershell
Get-Content .\artifacts\training\latest.json
Get-Content ".\artifacts\training\run-...\checkpoints\latest.json"
Get-Content ".\artifacts\training\run-...\checkpoints\best.json"
```

Then evaluate it:

```powershell
.\train.ps1 -Mode Evaluate -Checkpoint ".\artifacts\training\run-...\checkpoints\best.json" -Episodes 20 -NoPreview
```

## Start over without deleting old data

Just run `Fresh` again; a new timestamped run directory is created and old runs
remain intact.

```powershell
.\train.ps1 -Mode Fresh -Steps 100000 -NoPreview
```

## Activity playback is spikes-only

Older recordings have no `voltage-20hz/` directory. The player falls back to the
spike view automatically.

## Benchmark directories must be new

Live benchmark labels create recording directories. Use a fresh `--label` for
each live run.

## Force-killing the terminal

Avoid it, especially while recording. Encoders and checkpoint writers need to
finalize cleanly. Use `Ctrl+C`, which triggers the clean shutdown path.
