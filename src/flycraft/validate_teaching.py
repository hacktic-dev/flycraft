"""Dependency-light checks for FlyCraft's balanced Run-2 teaching transform."""
import copy
import json
import math
from pathlib import Path

from .training_metrics import teaching_signal


def state(*, angle_deg=60.0, distance=4.0, progress=.50, on_target=False, target_present=True):
    return {
        'angle': math.radians(angle_deg),
        'distance': float(distance),
        'progress': float(progress),
        'on_target': bool(on_target),
        'target_present': bool(target_present),
    }


def changed(base, **kwargs):
    out=copy.deepcopy(base)
    if 'angle_deg' in kwargs:out['angle']=math.radians(kwargs.pop('angle_deg'))
    out.update(kwargs)
    return out


def main():
    root=Path(__file__).resolve().parents[2]
    config=json.loads((root/'training.json').read_text())['teaching']
    base=state()

    cases=[]
    for name,plus,minus in [
        ('aim',changed(base,angle_deg=50),changed(base,angle_deg=70)),
        ('distance',changed(base,distance=3.75),changed(base,distance=4.25)),
        ('break',changed(base,progress=.60),changed(base,progress=.40)),
    ]:
        p=teaching_signal(base,plus,config)[0]
        n=teaching_signal(base,minus,config)[0]
        if not math.isclose(p,-n,rel_tol=1e-12,abs_tol=1e-12):
            raise AssertionError(f'{name} is not symmetric: {p} vs {n}')
        cases.append((name,p,n))

    acquired=teaching_signal(changed(base,on_target=False),changed(base,on_target=True),config)[0]
    lost=teaching_signal(changed(base,on_target=True),changed(base,on_target=False),config)[0]
    if not math.isclose(acquired,-lost,rel_tol=1e-12,abs_tol=1e-12):
        raise AssertionError(f'crosshair is not symmetric: {acquired} vs {lost}')
    cases.append(('crosshair',acquired,lost))

    neutral=teaching_signal(base,copy.deepcopy(base),config)[0]
    if neutral != 0.0:raise AssertionError(f'unchanged state should be neutral, got {neutral}')

    print('BALANCED TEACHING CHECK: PASS')
    for name,p,n in cases:print(f'  {name:9s} + {p:+.6f} / - {n:+.6f} / sum {p+n:+.3e}')
    print(f'  neutral    {neutral:+.6f}')


if __name__=='__main__':main()
