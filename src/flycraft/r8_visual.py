"""DOOMFLY v6 visual-input adapter layered onto FlyCraft Run-2 memory.

This intentionally ports only the visual-input assumptions from DOOMFLY's
``doom_learning_v6.visual`` at the repository revision pinned by FlyCraft.
The learning rule is FlyCraft's reversible gamma1-centered-signed-v2 rule.

Visual assumptions inherited from DOOMFLY v6:
- keep the existing R1-R6 luminance input unchanged;
- infer display coordinates for annotated R8p/R8y cells from their outgoing
  contacts, using the same viewport transform as the R1-R6 anchors;
- drive R8p from linear-sRGB blue and R8y from linear-sRGB green;
- make existing R8->aMe12 contacts excitatory while retaining their magnitudes;
- update the visual low-pass/current in chunks of at most 10 ms.

These are modeling assumptions, not validated fly photoreceptor physiology.
"""
import hashlib
import math
import time

import numpy as np

import doom_learning.brain as legacy
from doom_learning.flycraft_v2 import CenteredMemoryBrain
from doom.game import retinal_samples
from doom_learning.common import annotations, digest
from doom_learning_v6.visual import projection as doomfly_v6_projection


def _sha256(array):
    return hashlib.sha256(np.asarray(array).tobytes()).hexdigest()


class R8VisualMemoryBrain(CenteredMemoryBrain):
    """Run-2 reversible plasticity brain with the DOOMFLY v6 RGB/R8 adapter."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Keep a provenance hash of the untouched pinned whole-graph weights so
        # model-lock.json can still verify that the expected upstream graph was
        # loaded before the v6 visual sign assumption is applied.
        self.pre_visual_weight_sha256 = _sha256(self.weight)

        a = annotations(self.ids)
        self.r8, self.r8_uv, self.r8_confidence = doomfly_v6_projection(self, a)
        self.r8_channel = np.where(a.type.iloc[self.r8].eq('R8p'), 2, 1).astype(np.int32)
        self.r8_light = np.zeros(len(self.r8), dtype=np.float32)

        corrected = []
        for i in np.flatnonzero(a.type.fillna('').str.startswith('R8')):
            edges = np.arange(self.ptr[i], self.ptr[i + 1])
            selected = edges[a.type.iloc[self.post[edges]].eq('aMe12').to_numpy()]
            corrected.extend(selected.tolist())
            self.weight[selected] = np.abs(self.weight[selected])
        self.corrected_edges = np.asarray(corrected, dtype=np.int64)

        # v1 checkpoint provenance includes initial_weight_sha256. Match the v6
        # adapter by defining the corrected visual model as the new initial state.
        self.initial_weight_sha256 = digest(self.weight)
        self.fields.append('r8_light')
        self.initial['r8_light'] = self.r8_light.copy()

        self.visual_report = {
            'model': 'r8-rgb-ame12-v1',
            'source': 'DOOMFLY doom_learning_v6.visual at pinned repository revision',
            'mapped_R8p': int((self.r8_channel == 2).sum()),
            'mapped_R8y': int((self.r8_channel == 1).sum()),
            'mapped_R8_total': int(len(self.r8)),
            'known_unmapped': int(a.type.isin(['R8p', 'R8y']).sum() - len(self.r8)),
            'projection_confidence_median': float(np.median(self.r8_confidence)),
            'projection_below_80_percent': int((self.r8_confidence < .8).sum()),
            'corrected_existing_edges': len(corrected),
            'corrected_edge_sha256': digest(self.corrected_edges),
            'coordinate_inference': 'Modal column of outgoing contacts to column-annotated targets; same viewport transform as R1-R6.',
            'spectrum': 'Linear sRGB B for R8p; G for R8y. R7, dorsal and untyped R8 receive no invented optical drive.',
            'physiology': 'R8 to aMe12 net sign positive; original contact magnitudes retained. Other R8 targets retain baseline sign.',
            'validated': False,
        }

    def step(self, luminance, duration_ms, *, learning=False, stimulation=None, lamina_bias=12.):
        """Neural step plus Run-2 eligibility/recovery bookkeeping.

        Native v1 LTD is always disabled here. The native kernel is retained only
        for full-graph neural integration; signed reversible plasticity is applied
        by CenteredMemoryBrain after each <=10 ms visual/neural bin.
        """
        light=np.asarray(luminance)
        if light.shape!=(len(self.retina),) or not np.isfinite(light).all():
            raise ValueError('Invalid retinal input')
        steps=round(duration_ms/self.dt)
        if not math.isfinite(duration_ms) or steps<1 or not math.isfinite(lamina_bias):
            raise ValueError('Invalid interval/current')

        self.luminance+=(1-math.exp(-steps*self.dt/10))*(np.clip(light,0,1)-self.luminance)
        self.drive.fill(0)
        self.drive[self.lamina]=lamina_bias
        self.drive[self.retina]=30*self.luminance/(.02+self.luminance)

        if stimulation is not None:
            pulses=stimulation if isinstance(stimulation,list) else [stimulation]
            for indices,current in pulses:
                ix=np.asarray(indices,dtype=np.int32)
                amplitude=np.asarray(current,dtype=np.float32)
                if (ix.ndim!=1 or np.any(ix<0) or np.any(ix>=self.n) or
                        not np.isfinite(amplitude).all() or amplitude.shape not in [(),ix.shape]):
                    raise ValueError('Invalid external stimulation')
                self.drive[ix]+=amplitude

        self.counts.fill(0)
        clock=np.asarray([self.cursor],dtype=np.int64)
        c=self.circuit
        arrays=[self.ptr,self.post,self.weight,self.v,self.g,self.refractory,self.drive,
                self.previous_drive,self.queue,self.queue_count,clock]
        start=time.perf_counter()
        self.advance(self.n,*[x.ctypes.data for x in arrays],steps,self.dt,
            *[getattr(self,k).ctypes.data for k in ['counts','active','active_flag','nactive','last']],
            c['kc_mask'].ctypes.data,c['dan_index'].ctypes.data,self.eligibility.ctypes.data,self.eligibility_last.ctypes.data,
            len(c['edges']),c['edges'].ctypes.data,c['pre'].ctypes.data,self.baseline_plastic.ctypes.data,c['gain'].ctypes.data,
            self.eta,legacy.PARAMETERS['eligibility_tau_ms'],legacy.PARAMETERS['minimum_efficacy_fraction'],0,
            self.modulation.ctypes.data,self.modulation_last.ctypes.data)
        elapsed=time.perf_counter()-start
        self.cursor=int(clock[0])
        self.sim_ms=self.cursor*self.dt
        self.total_spikes+=int(self.counts.sum())
        counts=self.counts.copy()
        self.after_neural_bin(counts,duration_ms,learning=learning)
        return counts,elapsed

    def rgb_step(self, frame, duration_ms, **kwargs):
        """Advance from one RGB frame using DOOMFLY v6's visual-input method."""
        # Sample the held image once, but retain every 10 ms filter/current
        # update and its original floating-point operation order.
        cached = kwargs.pop('_visual_input', None)
        samples = kwargs.pop('_retinal_samples', None)
        if cached is None:
            frame = np.asarray(frame)
            if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
                raise ValueError('RGB uint8 required')

            h, w = frame.shape[:2]
            x = np.minimum((self.r8_uv[:, 0] * (w - 1)).astype(int), w - 1)
            y = np.minimum((self.r8_uv[:, 1] * (h - 1)).astype(int), h - 1)
            values = frame[y, x, self.r8_channel].astype(np.float32) / 255.0
            # Exact sRGB -> linear-light transfer used by DOOMFLY v6.
            values = np.where(values <= .04045, values / 12.92, ((values + .055) / 1.055) ** 2.4)

        else:
            samples, values = cached

        samples = retinal_samples(frame, self.uv) if samples is None else samples

        # This is deliberate: DOOMFLY v6 never lets one visual-current/filter
        # update span more than 10 ms, even when the game/control interval is
        # longer. The same Minecraft frame is held during these sub-intervals.
        if duration_ms > 10:
            ticks = round(duration_ms / self.dt)
            total = np.zeros(self.n, dtype=np.int32)
            wall = 0.0
            while ticks:
                n = min(100, ticks)
                counts, elapsed = self.rgb_step(frame, n * self.dt, _visual_input=(samples, values), **kwargs)
                total += counts
                wall += elapsed
                ticks -= n
            self.counts[:] = total
            return total, wall

        steps = round(duration_ms / self.dt)
        self.r8_light += (1 - math.exp(-steps * self.dt / 10.0)) * (values - self.r8_light)

        extra = kwargs.pop('stimulation', None)
        pulses = [] if extra is None else list(extra) if isinstance(extra, list) else [extra]
        pulses.append((self.r8, 30 * self.r8_light / (.02 + self.r8_light)))

        return self.step(samples, duration_ms, stimulation=pulses, **kwargs)

    def configuration_signature(self):
        return {
            **super().configuration_signature(),
            **{k: digest(getattr(self, k)) for k in ['r8', 'r8_uv', 'r8_channel', 'corrected_edges']},
        }
