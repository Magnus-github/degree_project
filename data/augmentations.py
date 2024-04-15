from scipy import signal
import random


class TemporalScale:
    def __init__(self, factor=0.9, p=0.5):
        self.factor = factor
        self.p = p

    def temporal_scale(self, pose_sequence):
        num = int(pose_sequence.shape[0]*self.factor)
        return signal.resample(pose_sequence, num, axis=0)

    def __call__(self, data):
        if random.random() < self.p:
            return self.temporal_scale(data)
        return data

    
class AmplitudeScale:
    def __init__(self, factor=0.9, p=0.5):
        self.factor = factor
        self.p = p

    def _amplitude_scale(self, pose_sequence):
        return pose_sequence*self.factor
    
    def __call__(self, data):
        if random.random() < self.p:
            return self._amplitude_scale(data)
        return data


class RandScale:
    def __init__(self, magnitude: int = 1, p=0.5, class_agnostic=False):
        factors = [1 + magnitude*0.2, 1 - magnitude*0.2]
        factor = random.choice(factors)
        temporal_scale = TemporalScale(factor, p)
        amplitude_scale = AmplitudeScale(factor, p)
        self.transform = random.choice([temporal_scale, amplitude_scale])
        self.class_agnostic = class_agnostic

    def __call__(self, sample):
        return self.transform(sample)
