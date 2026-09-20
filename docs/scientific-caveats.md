# Scientific caveats

This project is an experiment built around a reconstructed connectome and
simplified neural dynamics. These distinctions should remain explicit in any
interpretation, paper, or video.

- The MaleCNS wiring/connectome is reconstructed biological anatomy, but the
  simulated physiology is simplified.
- The Minecraft visual input uses the existing retinal sampling model (plus the
  experimental DOOMFLY v6 R8 colour adapter in learning mode). It is not a
  literal claim about the fly's subjective visual experience.
- The descending-neuron → Minecraft control mapping is engineered.
- Minecraft target position, angle, distance, raycast and breaking progress are
  external trainer/evaluation telemetry, **not** fly sensory input.
- Training uses DOOMFLY's existing experimental plasticity mechanisms plus
  clearly-labelled engineered rules:
  - `gamma1-fatigue-selective-signed-v4` — FlyCraft's fatigue-selective signed
    rule on existing KC→MBON11 edges.
  - `action-conditioned-steering-v2` — an engineered action-credit rule, not a
    reconstructed fly motor-learning circuit.
  - `frozen-connectome-readout-v1` — a small external supervised readout; the
    connectome is frozen and does not learn.
- Positive reward is an evaluation metric. Positive **teaching** is an explicit
  experimental FlyCraft signal, not a claim of a reconstructed appetitive
  dopamine pathway.
- Strong negative teaching may schedule the separately modelled PPL101 aversive
  pulse.
- Changed plastic weights do not by themselves prove useful learning.
  Improvement should be judged through frozen randomized evaluation.
- Replay/evaluation freezes plasticity so checkpoint comparisons do not alter the
  states being compared.
- The DOOMFLY v6 colour visual adapter and the bilateral decoder calibration are
  experimental and unvalidated; treat their outputs accordingly.
- The explainer animation is illustrative. Its neural graphics do not report
  measured training results.

## Accurate wording for the supervised readout

> The connectome was frozen. I trained a small motor readout to translate the
> fly simulation's neural activity into Minecraft controls using demonstrations,
> then let the learned controller act autonomously.

Do not claim that biological fly synapses learned Minecraft in that mode.

## Accurate wording for signed-reinforcement training

> I connected the reconstructed fly brain to Minecraft and let an experimental
> plasticity rule modify a small set of existing Kenyon-cell-to-MBON synapses in
> response to externally computed teaching signals. Behaviour was judged by
> frozen randomized evaluation, not by the fact that weights changed.
