"""Export FlyCraft's real arena into a player-anchored Minecraft 1.21 filming datapack.

Run from the FlyCraft project root after sourcing env.ps1:

    . .\env.ps1
    & $python .\scripts\export_film_datapack_anchored.py

The generated datapack lets you run:

    /function flycraft_film:arena_auto

wherever you are. That location becomes the FlyCraft training origin
(0, -59, 0). You can then fly away in spectator while the scheduled arena
steps continue building at that saved location.

Tree positions are converted using the exact same offset, so the generated
random training targets line up with the moved arena.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np

from flycraft.game import arena_commands
from flycraft.training_game import randomized_start


NAMESPACE = "flycraft_film"
ANCHOR_TAG = "flycraft_film_anchor"

# FlyCraft's arena is centred around x/z 0 and the player stands at y=-59.
# Mapping this point to the marker position makes the target level y=-58 become
# one block above the marker, matching the real training geometry.
SOURCE_ORIGIN = (0.0, -59.0, 0.0)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def function_path(root: Path, name: str) -> Path:
    # Minecraft 1.21 uses singular "function".
    return root / "data" / NAMESPACE / "function" / f"{name}.mcfunction"


def write_function(root: Path, name: str, lines):
    path = function_path(root, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(lines, str):
        lines = [lines]
    path.write_text(
        "\n".join(str(line).lstrip("/") for line in lines) + "\n",
        encoding="utf-8",
    )


def is_world_visual_command(command: str) -> bool:
    s = command.strip().lstrip("/").lower()
    if s.startswith(("fill ", "setblock ", "clone ", "place ")):
        return True
    if s.startswith("execute ") and any(
        token in s
        for token in (" run fill ", " run setblock ", " run clone ", " run place ")
    ):
        return True
    return False


def is_clear_command(command: str) -> bool:
    s = command.strip().lstrip("/").lower()
    return (
        ("fill " in s and (" minecraft:air" in s or s.endswith(" air")))
        or ("setblock " in s and (" minecraft:air" in s or s.endswith(" air")))
    )


_NUMBER = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$")


def _relative_token(token: str, axis: int) -> str:
    """Convert one absolute numeric coordinate into a ~offset from SOURCE_ORIGIN."""
    if token.startswith(("~", "^")):
        # Already relative/local: leave it untouched.
        return token
    if not _NUMBER.match(token):
        raise ValueError(f"Expected numeric coordinate, got {token!r}")

    value = float(token)
    delta = value - SOURCE_ORIGIN[axis]

    if abs(delta) < 1e-9:
        return "~"

    # Keep integers clean for readable mcfunctions.
    if abs(delta - round(delta)) < 1e-9:
        return f"~{int(round(delta))}"

    return f"~{delta:.6f}".rstrip("0").rstrip(".")


def _rewrite_direct_command(command: str) -> str:
    """Rewrite absolute coords in fill/setblock/clone to offsets from SOURCE_ORIGIN."""
    parts = command.split()
    if not parts:
        return command

    op = parts[0].lower()

    if op == "setblock":
        if len(parts) < 5:
            raise ValueError(f"Malformed setblock command: {command}")
        parts[1] = _relative_token(parts[1], 0)
        parts[2] = _relative_token(parts[2], 1)
        parts[3] = _relative_token(parts[3], 2)
        return " ".join(parts)

    if op == "fill":
        if len(parts) < 8:
            raise ValueError(f"Malformed fill command: {command}")
        for base in (1, 4):
            parts[base + 0] = _relative_token(parts[base + 0], 0)
            parts[base + 1] = _relative_token(parts[base + 1], 1)
            parts[base + 2] = _relative_token(parts[base + 2], 2)
        return " ".join(parts)

    if op == "clone":
        if len(parts) < 10:
            raise ValueError(f"Malformed clone command: {command}")
        # source begin, source end, destination begin
        for base in (1, 4, 7):
            parts[base + 0] = _relative_token(parts[base + 0], 0)
            parts[base + 1] = _relative_token(parts[base + 1], 1)
            parts[base + 2] = _relative_token(parts[base + 2], 2)
        return " ".join(parts)

    if op == "place":
        # FlyCraft currently does not need this for the arena. Refuse rather than
        # silently moving a structure incorrectly if that changes later.
        raise ValueError(
            f"Cannot safely relocate this 'place' command automatically: {command}"
        )

    raise ValueError(f"Unsupported visual command: {command}")


def relocate_command(command: str) -> str:
    """Relocate a visual command to be relative to the filming anchor."""
    command = command.strip().lstrip("/")

    # Preserve an existing execute prefix, but rewrite the command after its
    # final " run ". This covers the wrapped commands emitted by arena_commands().
    lower = command.lower()
    marker = " run "
    if lower.startswith("execute ") and marker in lower:
        idx = lower.rfind(marker)
        prefix = command[: idx + len(marker)]
        tail = command[idx + len(marker):]
        return prefix + _rewrite_direct_command(tail)

    return _rewrite_direct_command(command)


def at_anchor(command: str) -> str:
    return (
        f"execute at @e[type=minecraft:marker,tag={ANCHOR_TAG},limit=1] "
        f"run {command}"
    )


def zip_folder(folder: Path, output_zip: Path):
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    if output_zip.exists():
        output_zip.unlink()
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for path in folder.rglob("*"):
            if path.is_file():
                z.write(path, path.relative_to(folder))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--step-delay",
        type=int,
        default=20,
        help="Ticks between arena build steps (default: 20 = 1 second).",
    )
    parser.add_argument(
        "--tree-delay",
        type=int,
        default=60,
        help="Ticks before delayed tree spawn (default: 60 = 3 seconds).",
    )
    parser.add_argument(
        "--tree-options",
        type=int,
        default=12,
        help="Number of genuine randomized target positions to export.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed; defaults to config/baseline.json seed.",
    )
    args = parser.parse_args()

    if args.step_delay < 1 or args.tree_delay < 1 or args.tree_options < 1:
        raise SystemExit("Delays/tree-options must be positive.")

    game_config = load_json(ROOT / "config" / "baseline.json")
    training = load_json(ROOT / "training.json")

    seed = int(game_config.get("seed", 0) if args.seed is None else args.seed)
    rng = np.random.default_rng(seed)

    raw_commands = [str(c).strip().lstrip("/") for c in arena_commands(game_config)]
    visual_commands = [c for c in raw_commands if is_world_visual_command(c)]
    relocated = [relocate_command(c) for c in visual_commands]

    if not relocated:
        raise RuntimeError("No visible arena-building commands found.")

    artifact_dir = ROOT / "artifacts"
    build_root = artifact_dir / "flycraft-film-datapack"
    zip_path = artifact_dir / "flycraft-film-datapack.zip"
    manifest_path = artifact_dir / "flycraft-film-commands.txt"

    if build_root.exists():
        shutil.rmtree(build_root)
    build_root.mkdir(parents=True)

    (build_root / "pack.mcmeta").write_text(
        json.dumps(
            {
                "pack": {
                    "pack_format": 48,
                    "description": "FlyCraft filming helpers - anchored arena",
                }
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    manifest = [
        "FlyCraft filming datapack",
        f"Source origin mapped to marker: {SOURCE_ORIGIN}",
        "",
        "Original -> relocated visual commands:",
    ]

    step_names = []
    for i, (original, moved) in enumerate(zip(visual_commands, relocated), 1):
        name = f"arena/step_{i:02d}"
        step_names.append(name)

        # Each delayed step explicitly finds the saved marker, so /schedule no
        # longer needs to preserve the original player's execution position.
        write_function(build_root, name, at_anchor(moved))
        manifest.append(f"{i:02d} ORIGINAL: {original}")
        manifest.append(f"   MOVED:    {moved}")

    # Save the current player's location as the new arena origin.
    anchor_lines = [
        f"kill @e[type=minecraft:marker,tag={ANCHOR_TAG}]",
        f'summon minecraft:marker ~ ~ ~ {{Tags:["{ANCHOR_TAG}"]}}',
        'tellraw @s {"text":"FlyCraft filming origin saved here.","color":"aqua"}',
    ]
    write_function(build_root, "anchor_here", anchor_lines)

    # Immediate build: reset anchor to the caller, then execute all steps.
    write_function(
        build_root,
        "arena_all",
        [
            f"function {NAMESPACE}:anchor_here",
            *[f"function {NAMESPACE}:{name}" for name in step_names],
        ],
    )

    # Delayed build: save anchor NOW, then schedule context-independent steps.
    auto_lines = [f"function {NAMESPACE}:anchor_here"]
    for i, name in enumerate(step_names, 1):
        delay = i * args.step_delay
        auto_lines.append(f"schedule function {NAMESPACE}:{name} {delay}t append")
    auto_lines.append(
        f'tellraw @s {{"text":"Arena build scheduled here ({len(step_names)} steps).","color":"green"}}'
    )
    write_function(build_root, "arena_auto", auto_lines)

    clear_relocated = [
        relocate_command(c) for c in visual_commands if is_clear_command(c)
    ]
    if clear_relocated:
        write_function(
            build_root,
            "arena_clear",
            [at_anchor(c) for c in clear_relocated],
        )

    # Genuine FlyCraft randomized targets, converted to offsets from SOURCE_ORIGIN.
    targets = []
    seen = set()
    attempts = 0
    while len(targets) < args.tree_options and attempts < args.tree_options * 50:
        attempts += 1
        start = randomized_start(rng, training["environment"])
        target = tuple(int(v) for v in start["target"])
        if target not in seen:
            seen.add(target)
            targets.append(target)

    if not targets:
        raise RuntimeError("randomized_start() produced no tree targets.")

    def rel_xyz(target):
        x, y, z = target
        return (
            _relative_token(str(x), 0),
            _relative_token(str(y), 1),
            _relative_token(str(z), 2),
        )

    candidate_rel = [rel_xyz(t) for t in targets]

    manifest.append("")
    manifest.append("Randomized target positions:")
    for i, (absolute, relative) in enumerate(zip(targets, candidate_rel), 1):
        manifest.append(f"{i:02d}: absolute={absolute} relative={relative}")

    # For filming, represent each genuine randomized target as a compact oak tree.
    # The BOTTOM trunk block replaces the red floor marker one block below the FlyCraft target log coordinate.
    # The real training task targets a single oak log; the extra trunk/leaves are
    # visual dressing for the narration/filming shot only.
    #
    # Shape, relative to the red floor marker below the target:
    #   trunk: y 0..3
    #   leaves: 3x3 at y=3 and y=4, plus a small cross/top at y=5
    tree_blocks = []

    # Trunk.
    for dy in range(0, 4):
        tree_blocks.append((0, dy, 0, "minecraft:oak_log"))

    # Two compact canopy layers.
    for dy in (3, 4):
        for dx in range(-1, 2):
            for dz in range(-1, 2):
                # Don't overwrite the trunk in the lower canopy layer.
                if dx == 0 and dz == 0 and dy == 3:
                    continue
                tree_blocks.append(
                    (dx, dy, dz, "minecraft:oak_leaves[persistent=true]")
                )

    # Crown.
    for dx, dz in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
        tree_blocks.append(
            (dx, 5, dz, "minecraft:oak_leaves[persistent=true]")
        )

    def relative_tree_command(target, block, *, material_override=None):
        tx, ty, tz = target

        # The FlyCraft target log coordinate is one block ABOVE the red floor
        # marker. For the cinematic full tree, put the bottom trunk into that
        # red marker block itself so the tree grows out of the floor instead
        # of floating one block above it.
        base_y = ty - 1

        dx, dy, dz, block_name = block
        x = _relative_token(str(tx + dx), 0)
        y = _relative_token(str(base_y + dy), 1)
        z = _relative_token(str(tz + dz), 2)
        material = material_override or block_name
        return at_anchor(f"setblock {x} {y} {z} {material}")

    # FILMING VERSION: use one genuine randomized target only.
    #
    # Do NOT touch every possible randomized target position. The previous
    # version did that and accidentally painted red_concrete blocks all over
    # the arena. tree_delayed/tree_spawn only use targets[0], so only manage
    # that one footprint.
    film_target = targets[0]

    def film_tree_place_commands(target):
        return [
            relative_tree_command(target, block)
            for block in tree_blocks
        ]

    def film_tree_clear_commands(target):
        commands = []
        for block in tree_blocks:
            dx, dy, dz, _ = block

            # The bottom trunk replaces the arena's red target marker.
            # Restore ONLY that one block when clearing the filming tree.
            if (dx, dy, dz) == (0, 0, 0):
                commands.append(
                    relative_tree_command(
                        target,
                        block,
                        material_override="minecraft:red_concrete",
                    )
                )
            else:
                commands.append(
                    relative_tree_command(
                        target,
                        block,
                        material_override="minecraft:air",
                    )
                )
        return commands

    place_tree = film_tree_place_commands(film_target)
    clear_tree = film_tree_clear_commands(film_target)

    # Spawning overwrites the single red marker with the bottom oak log and
    # builds the rest of the tree. It does not alter any other red blocks.
    write_function(
        build_root,
        "tree/tree_01",
        place_tree,
    )

    write_function(
        build_root,
        "tree_spawn",
        [f"function {NAMESPACE}:tree/tree_01"],
    )
    write_function(build_root, "tree_clear", clear_tree)

    # tree_delayed uses the existing arena anchor. It intentionally does NOT
    # move the anchor to the spectator camera's current position.
    write_function(
        build_root,
        "tree_delayed",
        [
            f"schedule function {NAMESPACE}:tree/tree_01 {args.tree_delay}t append",
            f'tellraw @s {{"text":"Tree scheduled in {args.tree_delay/20:.1f}s at the saved arena origin.","color":"yellow"}}',
        ],
    )

    cancel_lines = [
        f"schedule clear {NAMESPACE}:{name}" for name in step_names
    ] + [
        f"schedule clear {NAMESPACE}:tree/tree_01"
    ]
    write_function(build_root, "cancel", cancel_lines)

    write_function(
        build_root,
        "anchor_remove",
        [f"kill @e[type=minecraft:marker,tag={ANCHOR_TAG}]"],
    )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("\n".join(manifest) + "\n", encoding="utf-8")

    readme = f"""FlyCraft anchored filming datapack
================================

Wherever you run:

/function {NAMESPACE}:arena_auto

becomes FlyCraft coordinate (0, -59, 0).

The function saves an invisible marker there, then builds the real arena
relative to that marker. You are free to fly away in spectator mode while it
continues building.

Main commands:

/function {NAMESPACE}:arena_auto
    Save HERE as the arena origin and build one step every
    {args.step_delay/20:.1f}s.

/function {NAMESPACE}:arena_all
    Save HERE as the arena origin and build everything immediately.

/function {NAMESPACE}:anchor_here
    Move the filming origin to your current position without building.

/function {NAMESPACE}:tree_delayed
    Spawn a filming oak tree whose bottom log is at a genuine randomized training target after
    {args.tree_delay/20:.1f}s relative to the saved arena origin.

/function {NAMESPACE}:tree_spawn
    Spawn it immediately.

/function {NAMESPACE}:tree_clear
    Remove generated candidate trees from the saved arena.

/function {NAMESPACE}:cancel
    Cancel scheduled arena/tree functions.

Suggested workflow:
1. Go to the exact spot where you want the centre/origin of the arena.
2. /function {NAMESPACE}:arena_auto
3. Immediately fly away in spectator and frame the shot.
4. Once built, use /function {NAMESPACE}:tree_delayed.
5. Fly your camera; the tree appears at the arena, not at your new camera position.
"""
    (build_root / "README.txt").write_text(readme, encoding="utf-8")

    zip_folder(build_root, zip_path)

    print(f"Generated {len(relocated)} anchored arena steps.")
    print(f"Mapped source origin {SOURCE_ORIGIN} to wherever arena_auto is called.")
    print(f"Generated one filming tree at a genuine randomized FlyCraft target position.")
    print(f"Datapack ZIP: {zip_path}")
    print(f"Manifest: {manifest_path}")
    print()
    print("Replace the old datapack ZIP in the Minecraft world's datapacks folder,")
    print("run /reload, stand/fly where you want the arena origin, then:")
    print(f"  /function {NAMESPACE}:arena_auto")


if __name__ == "__main__":
    main()
