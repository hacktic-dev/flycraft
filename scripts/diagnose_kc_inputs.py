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

# All existing edges whose POSTSYNAPTIC neuron is a KC.
incoming_edges = np.flatnonzero(np.isin(brain.post, kc))
incoming_pre = (
    np.searchsorted(brain.ptr, incoming_edges, side="right") - 1
).astype(np.int32)

incoming_post = brain.post[incoming_edges]
incoming_weight = brain.weight[incoming_edges]

print("=== KC INPUT CONNECTIVITY ===")
print("KCs:", len(kc))
print("Incoming KC edges:", len(incoming_edges))
print("Unique presynaptic neurons:", len(np.unique(incoming_pre)))

print("\nWeights into KCs:")
print("  min :", float(incoming_weight.min()))
print("  mean:", float(incoming_weight.mean()))
print("  max :", float(incoming_weight.max()))
print("  positive:", int(np.count_nonzero(incoming_weight > 0)))
print("  negative:", int(np.count_nonzero(incoming_weight < 0)))

unique_pre = np.unique(incoming_pre)

total_pre_spikes = 0
ticks_with_pre_activity = 0
max_pre_spikes = 0
highest_kc_voltage = -1e9

env = None

try:
    env = make_game(game_config)
    obs = initialize_game(env, game_config)

    print("\n=== 10 SECOND LIVE TEST ===")

    for tick in range(200):
        rgb = pixels(obs, game_config)

        control, neural = fly.step(rgb, record=False)

        counts = fly.last_counts
        pre_spikes = int(counts[unique_pre].sum())
        active_pre = int(np.count_nonzero(counts[unique_pre]))

        kc_spikes = int(counts[kc].sum())
        kc_v = fly.last_voltage[kc]

        total_pre_spikes += pre_spikes
        max_pre_spikes = max(max_pre_spikes, pre_spikes)

        if pre_spikes:
            ticks_with_pre_activity += 1

        highest_kc_voltage = max(
            highest_kc_voltage,
            float(kc_v.max())
        )

        # Crude amount of spike-weight input aimed at KCs this tick.
        weighted_input = (
            counts[incoming_pre] * incoming_weight
        )

        if tick % 20 == 0:
            print(
                f"tick {tick:3d} | "
                f"KC-input pre spikes {pre_spikes:6d} | "
                f"active presyn {active_pre:5d} | "
                f"weighted input {weighted_input.sum():9.3f} | "
                f"KC spikes {kc_spikes:4d} | "
                f"KC Vmax {kc_v.max():7.3f}"
            )

        action = map_action(control, game_config)
        obs = env.step(action)[0]

finally:
    if env is not None:
        env.close()

print("\n=== RESULT ===")
print("Total spikes from neurons directly connected into KCs:", total_pre_spikes)
print("Ticks with any KC-input presynaptic activity:", ticks_with_pre_activity, "/ 200")
print("Maximum presynaptic spikes in one tick:", max_pre_spikes)
print("Highest KC voltage:", highest_kc_voltage, "mV")
print("KC threshold: -45.0 mV")