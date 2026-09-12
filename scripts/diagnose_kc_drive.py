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

# All edges terminating on KCs.
incoming_edges = np.flatnonzero(np.isin(brain.post, kc))
incoming_pre = (
    np.searchsorted(brain.ptr, incoming_edges, side="right") - 1
).astype(np.int32)

incoming_post = brain.post[incoming_edges]
incoming_weight = brain.weight[incoming_edges]

print("=== KC INPUT WEIGHTS ===")
print("Incoming edges:", len(incoming_edges))
print("Positive edges:", np.count_nonzero(incoming_weight > 0))
print("Negative edges:", np.count_nonzero(incoming_weight < 0))
print("Zero edges:", np.count_nonzero(incoming_weight == 0))
print("Weight min:", incoming_weight.min())
print("Weight mean:", incoming_weight.mean())
print("Weight max:", incoming_weight.max())

# KC index -> compact 0..N-1 index
kc_lookup = np.full(brain.n, -1, dtype=np.int32)
kc_lookup[kc] = np.arange(len(kc))

post_compact = kc_lookup[incoming_post]

max_tick_net = np.full(len(kc), -np.inf)
max_tick_positive = np.zeros(len(kc))
total_net = np.zeros(len(kc))
total_positive = np.zeros(len(kc))
total_negative = np.zeros(len(kc))

best_voltage = np.full(len(kc), -np.inf)

env = None

try:
    env = make_game(game_config)
    obs = initialize_game(env, game_config)

    for tick in range(200):
        rgb = pixels(obs, game_config)
        control, neural = fly.step(rgb, record=False)

        counts = fly.last_counts

        edge_input = counts[incoming_pre] * incoming_weight

        net = np.zeros(len(kc), dtype=np.float64)
        pos = np.zeros(len(kc), dtype=np.float64)
        neg = np.zeros(len(kc), dtype=np.float64)

        np.add.at(net, post_compact, edge_input)
        np.add.at(pos, post_compact, np.maximum(edge_input, 0))
        np.add.at(neg, post_compact, np.minimum(edge_input, 0))

        max_tick_net = np.maximum(max_tick_net, net)
        max_tick_positive = np.maximum(max_tick_positive, pos)

        total_net += net
        total_positive += pos
        total_negative += neg

        kc_v = fly.last_voltage[kc]
        best_voltage = np.maximum(best_voltage, kc_v)

        if tick % 20 == 0:
            hottest = np.argmax(kc_v)

            print(
                f"tick {tick:3d} | "
                f"best KC V {kc_v[hottest]:7.3f} mV | "
                f"net input this tick {net[hottest]:8.3f} | "
                f"positive {pos[hottest]:8.3f} | "
                f"negative {neg[hottest]:8.3f}"
            )

        action = map_action(control, game_config)
        obs = env.step(action)[0]

finally:
    if env is not None:
        env.close()

print("\n=== MOST DRIVEN KCs ===")

order = np.argsort(best_voltage)[::-1][:20]

for rank, i in enumerate(order, 1):
    print(
        f"{rank:2d}. "
        f"index={kc[i]:6d} "
        f"body={brain.ids[kc[i]]} "
        f"Vmax={best_voltage[i]:7.3f} "
        f"max_net_tick={max_tick_net[i]:8.3f} "
        f"max_positive_tick={max_tick_positive[i]:8.3f} "
        f"total_net={total_net[i]:9.3f} "
        f"total_pos={total_positive[i]:9.3f} "
        f"total_neg={total_negative[i]:9.3f}"
    )

print("\n=== SUMMARY ===")
print("Best KC voltage:", best_voltage.max(), "mV")
print("Largest positive input to one KC in one tick:", max_tick_positive.max())
print("Largest net input to one KC in one tick:", max_tick_net.max())
print("KC threshold: -45.0 mV")