# Setup and environment

FlyCraft keeps its entire toolchain and generated data **inside the project** so
that nothing global is modified and the large pieces can be excluded from Git.

```text
.tools/      local Python, JDK, LLVM, uv, GLEW, Gradle, native build cache
.venv/       project Python environment
vendor/      pinned DOOMFLY and CraftGround checkouts + connectome data
artifacts/   all run output (baselines, training, recordings, proofs)
```

These four directories are in `.gitignore`.

---

## 1. `setup.ps1`

```powershell
.\setup.ps1
.\setup.ps1 -SkipDataAudit   # skip the long full connectome data audit on repeat setup
```

What it does, in order:

1. Creates `.tools`, `vendor`, `artifacts` and initializes a Git repo if needed.
2. Clones and pins the upstream checkouts:
   - `nftechie/doomfly` at `71ecf53d78eaffaf1a57ed7b0ccf5d458abc9f33`
   - `yhs0602/CraftGround` at `18eba01a87a8481bc4fbb54b42fee1428c0c3719`
3. Installs `uv`, Python 3.11.16 (into `.tools/python`) and the pinned Python
   dependencies from `requirements.lock.txt` (with DOOMFLY's build constraints).
4. Runs `scripts/download.py` (verified JDK 21, LLVM-mingw, GLEW and MaleCNS
   connectome data).
5. Loads `env.ps1` and runs `scripts/build_kernel.py` (native neural kernel).
6. Imports and prepares the connectome graph if missing:
   `python -m doom.connectome malecns_v1` and `python -m doom.prepare`.
7. Optionally runs the full data audit `python -m doom.audit_data`.
8. Runs `scripts/prepare_runtime.py` (Windows CraftGround patches, see below).
9. Configures and builds the CraftGround native library with CMake/Ninja,
   copies `glew32.dll`, and builds Minecraft's Java side with Gradle.
10. Runs `scripts/test_brain.py`, `python -m flycraft.activity` (observer build)
    and `python -m flycraft.anatomy` (fetch published anatomical positions).

Finish with:

```powershell
.\test.ps1
```

for live Minecraft verification.

### Manual equivalents

If you need to redo a step without a full setup:

```powershell
. .\env.ps1
& $python scripts/download.py
& $python scripts/build_kernel.py
& $python scripts/prepare_runtime.py
& $python scripts/prepare_training_runtime.py
& $python -m flycraft.activity
& $python -m flycraft.anatomy
```

---

## 2. `env.ps1`

`env.ps1` is loaded (dot-sourced) by every wrapper. It sets, for the current
process only:

- `JAVA_HOME` to the local JDK
- `PATH` to the local `.venv\Scripts`, JDK `bin`, LLVM `bin`, GLEW `bin`
- `PYTHONPATH` to `src` and `vendor/doomfly`
- `GRADLE_USER_HOME`, `CRAFTGROUND_JVM_MAX_MEMORY=3G`
- `CC`/`CXX` to the local clang, `CMAKE_GENERATOR=Ninja`
- `PYTHONUNBUFFERED=1`

It defines `$python` (the project interpreter) and throws
`Run .\setup.ps1 first.` if `.venv` is missing.

---

## 3. CraftGround runtime patches

FlyCraft patches the installed CraftGround package (in `.venv`) rather than
forking it. These scripts are idempotent.

### `scripts/prepare_runtime.py` — Windows build/run adaptations

- Static-links the native library and sets its output directory (CMakeLists).
- Disables Unix-socket orphan cleanup on Windows.
- Allows the protocol process to exit cleanly (saves world/cleanup).
- Shuts the Minecraft client down on the client thread.
- Keeps the **visible Minecraft window at the capture aspect ratio** (re-enables
  CraftGround's `update_override_resolutions` and passes `--width/--height` to
  `runClient`).
- Presents the actual framebuffer instead of a blank/white client area.

Run once after updating an existing checkout, then close any running Minecraft
client before relaunching:

```powershell
. .\env.ps1
& $python scripts\prepare_runtime.py
```

### `scripts/prepare_training_runtime.py` — genuine breaking telemetry

Training needs real block-breaking progress. This script:

- adds a Mixin accessor (`FlycraftBreakingAccessor`) for Minecraft's
  `currentBreakingProgress` / `currentBreakingPos` / `breakingBlock`,
- registers it in the mixin JSON,
- extends `MinecraftEnv.kt` to accept a `flycraft_target x y z` command and
  publish `flycraft.progress_bits` (float bits), `flycraft.target_present` and
  the target coordinates through the existing integer statistics transport.

`train.ps1` and `train-readout.ps1` run this automatically before launching.
If you see `Missing genuine breaking telemetry`, run it manually and restart
Minecraft so Gradle rebuilds.

---

## 4. `scripts/download.py`

Downloads and SHA-256-verifies:

- Temurin **JDK 21** (`jdk.zip`),
- **LLVM-mingw** (`llvm.zip`),
- MaleCNS connectome data into `vendor/doomfly/connectome_data/malecns_v1/`.

Extraction is skipped when the expected executables already exist.

---

## 5. `scripts/build_kernel.py`

Compiles the **unchanged** upstream `vendor/doomfly/doom/kernel.cpp` into
`vendor/doomfly/outputs/doom/libneural.so` (a Windows PE DLL with the upstream
filename) and writes a sidecar JSON with the source and binary hashes and the
model revision. The frozen baseline verifies the original weight hash against
`model-lock.json`.

---

## 6. `scripts/test_brain.py` / `scripts/test_game.py`

- `test_brain.py` — builds `FrozenFly`, steps 40 synthetic frames, asserts spikes
  and forward drive, re-verifies frozen weights, writes
  `artifacts/neural-proof.json`.
- `test_game.py` — live CraftGround test of RGB capture, turning, walking,
  arena reset and attack-based block breaking, writing
  `artifacts/craftground-proof/`.

---

## 7. Version lock

`model-lock.json` records the pinned upstream revisions, the model identity and
the original neural weight hash. `FrozenFly` and `LearningFly` both assert that
the loaded pre-visual weights match it.
