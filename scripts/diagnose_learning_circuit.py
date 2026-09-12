import json
import numpy as np

from flycraft.learning import LearningFly

ROOT_CONFIG = "training.json"

with open(ROOT_CONFIG, "r", encoding="utf-8-sig") as f:
    config = json.load(f)

fly = LearningFly(config["plasticity"])
brain = fly.brain
c = brain.circuit

print("\n=== CIRCUIT STRUCTURE ===")
print("neurons:", brain.n)
print("plastic edges:", len(c["edges"]))

kc = np.asarray(c["kc"], dtype=np.int64)
kc_mask = np.asarray(c["kc_mask"], dtype=bool)
plastic_pre = np.asarray(c["pre"], dtype=np.int64)

print("c['kc'] count:", len(kc))
print("kc_mask true:", int(kc_mask.sum()))
print("plastic presynaptic entries:", len(plastic_pre))
print("unique plastic presynaptic neurons:", len(np.unique(plastic_pre)))

print("\n=== INDEX CONSISTENCY ===")

mask_indices = np.flatnonzero(kc_mask)

print(
    "c['kc'] exactly equals kc_mask indices:",
    np.array_equal(np.sort(kc), np.sort(mask_indices)),
)

plastic_unique = np.unique(plastic_pre)

print(
    "all plastic presynaptic neurons are in kc_mask:",
    bool(np.all(kc_mask[plastic_unique])),
)

print(
    "plastic presynaptic neurons missing from c['kc']:",
    len(set(plastic_unique.tolist()) - set(kc.tolist())),
)

print(
    "KCs with no plastic outgoing edge:",
    len(set(kc.tolist()) - set(plastic_unique.tolist())),
)

print("\n=== BODY-ID SANITY CHECK ===")
print("first 10 KC array indices:")
print(kc[:10])

print("corresponding MaleCNS body IDs:")
print(brain.ids[kc[:10]])

print("\n=== CURRENT KC STATE ===")
print("KC spike counts:")
print("  total:", int(brain.counts[kc].sum()))
print("  active KCs:", int(np.count_nonzero(brain.counts[kc])))

v = brain.v[kc]

print("KC membrane voltage:")
print("  min :", float(v.min()), "mV")
print("  mean:", float(v.mean()), "mV")
print("  max :", float(v.max()), "mV")
print("  threshold: -45.0 mV")

elig = brain.eligibility[kc]

print("KC eligibility:")
print("  nonzero:", int(np.count_nonzero(elig)), "/", len(elig))
print("  max:", float(elig.max()))

print("\n=== DAN ===")
dan = np.asarray(c["dan"], dtype=np.int64)
print("DAN count:", len(dan))
print("DAN indices:", dan)
print("DAN body IDs:", brain.ids[dan])

print("\nDone.")