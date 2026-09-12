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
c = brain.circuit

kc = np.asarray(c["kc"], dtype=np.int32)
plastic_pre = np.asarray(c["pre"], dtype=np.int32)
kc_mask_indices = np.flatnonzero(c["kc_mask"])

print("=== STRUCTURE ===")
print("KC count:", len(kc))
print("Plastic edges:", len(c["edges"]))
print("Unique plastic KC sources:", len(np.unique(plastic_pre)))
print("KC == kc_mask indices:", np.array_equal(kc, kc_mask_indices))
print("All plastic sources are KCs:", bool(np.all(c["kc_mask"][plastic_pre])))

env = None

total_kc_spikes = 0
max_kc_voltage = -1e9
closest_tick = None

try:
    env = make_game(game_config)
    obs = initialize_game(env, game_config)

    print("\n=== LIVE MINECRAFT TEST ===")

    for tick in range(200):  # 10 seconds
        rgb = pixels(obs, game_config)

        control, neural = fly.step(rgb, record=False)

        kc_counts = fly.last_counts[kc]
        kc_v = fly.last_voltage[kc]
        elig = brain.eligibility[kc]

        spikes = int(kc_counts.sum())
        vmax = float(kc_v.max())

        total_kc_spikes += spikes

        if vmax > max_kc_voltage:
            max_kc_voltage = vmax
            closest_tick = tick

        # Let the fly drive Minecraft normally.
        action = map_action(control, game_config)
        obs = env.step(action)[0]

        if tick % 20 == 0 or spikes:
            print(
                f"tick {tick:3d} | "
                f"KC spikes {spikes:4d} | "
                f"V min {kc_v.min():7.3f} | "
                f"mean {kc_v.mean():7.3f} | "
                f"max {kc_v.max():7.3f} mV | "
                f"KC > -46mV {np.count_nonzero(kc_v > -46):4d} | "
                f"elig nonzero {np.count_nonzero(elig):4d} | "
                f"elig max {elig.max():.6f}"
            )

    print("\n=== RESULT ===")
    print("Total KC spikes:", total_kc_spikes)
    print("Highest KC voltage observed:", max_kc_voltage, "mV")
    print("Spike threshold: -45.0 mV")
    print("Closest approach occurred at tick:", closest_tick)

    v = fly.last_voltage[kc]

    print("\nFinal KC voltage distribution:")
    for threshold in (-52, -50, -48, -47, -46, -45.5, -45):
        print(
            f"  > {threshold:5.1f} mV:",
            int(np.count_nonzero(v > threshold)),
            "/",
            len(v)
        )

finally:
    if env is not None:
        env.close()