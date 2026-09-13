"""Cheap unit checks for natural-scene bilateral calibration."""
import numpy as np
from flycraft.decoder import calibrate_rate_samples, command_from_rates, validated_config

rng = np.random.default_rng(7)
# Deliberately asymmetric sides, like the real DNp20 baseline problem.
latent = rng.normal(0, 1, 500)
left = 7.0 + 2.0*latent + rng.normal(0, .7, len(latent))
right = 30.0 + 8.0*latent + rng.normal(0, 2.0, len(latent))
fit = calibrate_rate_samples(left, right, deadband=.05)
cfg = {'mode':'balanced-bilateral-v2','neuron_type':'DNp20',**{k:fit[k] for k in
    ('left_center_hz','right_center_hz','left_scale_hz','right_scale_hz','opponent_offset','response_scale')},
    'deadband':.05,'max_yaw_deg_per_second':120.,'polarity':1}
validated_config(cfg)
commands = np.array([command_from_rates(l,r,cfg)[0] for l,r in zip(left,right)])
assert abs(commands.mean()) < 1e-10, commands.mean()
assert .10 < np.median(np.abs(commands)) < .40
# Symmetric z-score perturbations must produce opposite steering signs.
base_l,base_r=fit['left_center_hz'],fit['right_center_hz']
ls,rs=fit['left_scale_hz'],fit['right_scale_hz']
a=command_from_rates(base_l-ls,base_r+rs,cfg)[0]
b=command_from_rates(base_l+ls,base_r-rs,cfg)[0]
assert a>0 and b<0,(a,b)
# Existing v1 remains valid for old checkpoints/configs.
validated_config({'mode':'balanced-bilateral-v1','neuron_type':'DNp20','left_reference_hz':7.,'right_reference_hz':30.,'regularizer':1.,'deadband':.05,'max_yaw_deg_per_second':120.,'polarity':1})
print('BILATERAL DECODER V2 TESTS PASSED')
