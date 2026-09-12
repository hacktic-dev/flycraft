import json
import numpy as np

from flycraft.learning import LearningFly
from flycraft.game import make_game, initialize_game, pixels, map_action

with open("training.json", "r", encoding="utf-8-sig") as f:
    training = json.load(f)

with open("config/baseline.json", "r", encoding="utf-8-sig") as f:
    game_config = json.load(f)

fly = LearningFly(training["plasticity"])
brain = fly.brain
kc = np.asarray(brain.circuit["kc"], dtype=np.int32)

env = None

highest = -1e9
highest_index = None
highest_tick = None
highest_substep = None
total_spikes = 0

try:
    env = make_game(game_config)
    obs = initialize_game(env, game_config)

    print("=== HIGH-RESOLUTION KC VOLTAGE TEST ===")

    # 20 Minecraft ticks = 1 second.
    for game_tick in range(20):
        rgb = pixels(obs, game_config)

        tick_counts = np.zeros(brain.n, dtype=np.int64)

        # Diagnostic-only sampling: fifty 1 ms RGB steps (including R8).
        # This changes visual filter cadence; it is not an exact training replay.
        for sub in range(50):
            counts, _ = brain.rgb_step(
                rgb,
                1.0,
                learning=False,
                stimulation=None
            )

            tick_counts += counts

            v = brain.v[kc]
            i = int(np.argmax(v))
            vmax = float(v[i])

            if vmax > highest:
                highest = vmax
                highest_index = int(kc[i])
                highest_tick = game_tick
                highest_substep = sub

        kc_spikes = int(tick_counts[kc].sum())
        total_spikes += kc_spikes

        # Decode the accumulated 50 ms activity so Minecraft still moves.
        control = fly.decoder.decode(tick_counts, 0.05)
        action = map_action(control, game_config)
        obs = env.step(action)[0]

        print(
            f"game tick {game_tick:2d} | "
            f"KC spikes {kc_spikes:3d} | "
            f"highest-ever V {highest:8.4f} mV"
        )

finally:
    if env is not None:
        env.close()

print("\n=== RESULT ===")
print("Total KC spikes:", total_spikes)
print("Highest KC voltage actually observed:", highest, "mV")
print("Threshold:", -45.0, "mV")
print("Distance below threshold:", -45.0 - highest, "mV")

if highest_index is not None:
    print("KC array index:", highest_index)
    print("KC body ID:", brain.ids[highest_index])
    print("Minecraft tick:", highest_tick)
    print("1 ms slice:", highest_substep)