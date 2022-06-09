import io
from dataclasses import dataclass, field
import json
import math
import numba
from numba import jit, typeof
import numpy as np
import random
from typing import Optional, Tuple

@jit(nopython=True)
def _fast_sawtooth(phase: float, _, __) -> float:
    return (phase % 1) * 2 - 1

@jit(nopython=True)
def _fast_square(phase: float, _, __) -> float:
    return 1 if (phase % 1) < .5 else -1

@jit(nopython=True)
def _fast_triangle(phase: float, _, __) -> float:
    return min(phase % 1, 1 - (phase % 1)) * 4 - 1

@jit(nopython=True)
def _fast_halfsin(phase: float, _, __) -> float:
    return math.sin(phase * 2 * math.pi) if (phase % 1) < .5 else 0

@jit(nopython=True)
def _fast_quartersin(phase: float, _, __) -> float:
    return math.sin(phase * 2 * math.pi) if (phase % 1) < .25 else 0

@jit(nopython=True)
def _fast_rectsin(phase: float, _, __) -> float:
    return abs(math.sin(phase * 2 * math.pi))

@jit(nopython=True)
def _fast_pm(phase: float, amt: float, ratio: float) -> float:
    return math.sin(2 * math.pi * (phase + amt * math.sin(phase * ratio * 2 * math.pi)))

_fast_funcs = (
    _fast_sawtooth, _fast_square, _fast_triangle, _fast_halfsin,
    _fast_quartersin, _fast_rectsin, _fast_pm)

@jit(nopython=True)
def _fast_play(
        n_samples: int, new_freq: float, new_coeffs: np.ndarray, new_gain: float,
        new_voice: float, old_freq: float, old_coeffs: np.ndarray, old_gain: float,
        old_voice: float, speed: float, cache: np.ndarray, index: int, phase: float,
        funcid=_fast_sawtooth, pm_amt: float=0, pm_freq: float=0) ->\
        Tuple[np.ndarray, float, np.ndarray, float, float, np.ndarray, int, float]:
    samples: np.ndarray = np.zeros(n_samples)
    for i in range(n_samples):
        # pulse: float = (phase % 1) * 2 - 1 # Sawtooth in [-1, 1]
        # pulse = 1 if (phase % 1) < .5 else -1 # Square in [-1, 1]
        # pulse = min(phase % 1, 1 - (phase % 1)) * 4 - 1 # Triangle in [-1, 1]
        # pulse = math.sin((phase + .5 * math.sin(phase * 2 * math.pi * 10)) * 2 * math.pi) # PM
        # pulse = math.sin(phase * 2 * math.pi) if (phase % 1) < .5 else 0 # halfsin
        # pulse = math.sin(phase * 2 * math.pi) if (phase % 1) < .25 else 0 # quartersin
        # pulse = abs(math.sin(phase * 2 * math.pi)) # rectsin
        pulse = funcid(phase, pm_amt, pm_freq)
        noise: float = random.random() * 2 - 1 # Noise in [-1, 1]
        old_voice += (new_voice - old_voice) * speed
        pulse = noise + (pulse - noise) * old_voice
        old_gain += (new_gain - old_gain) * speed
        # Taken almost verbatim from the finnish synth repo
        # Limits speed of coeffients between 2^-1 and 2^-5 which mostly avoid instability
        # Original was 2^-1 to 2^-6 or something like that
        hspeed: float = min(5, max(1, 7 + math.log10(old_gain)))
        hspeed = 2 ** -hspeed
        for j in range(len(new_coeffs)):
            coeff: float = old_coeffs[j]
            coeff += (new_coeffs[j] - coeff) * hspeed
            pulse -= cache[index - 1 - j] * coeff
            old_coeffs[j] = coeff
        cache[index] = pulse
        index = (index + 1) % len(new_coeffs)
        samples[i] = min(1.0, max(-1.0, pulse * old_gain ** .5))
        old_freq += (new_freq - old_freq) * speed
        phase += old_freq
    return (samples, old_freq, old_coeffs, old_gain, old_voice, cache, index, phase)
        
# _fast_play.inspect_types()

@dataclass
class LPC:
    """
    An LPC filter with gain, coefficients, and voice param
    """
    coefficients: np.ndarray
    gain: float
    voice: float
    
    def order(self) -> int:
        return len(self.coefficients)
    
    def todict(self) -> dict:
        return {'coefficients': self.coefficients.tolist(), 'gain': self.gain, 'voice': self.voice}
    
    @staticmethod
    def fromdict(d: dict) -> Optional['LPC']:
        return LPC(np.array(d['coefficients']), d['gain'], d['voice'])

@dataclass
class LPCPlayer:
    """
    Runs an LPC
    """
    order: int
    speed: float = 2**-12
    gain: float = 1
    voice: float = 1
    frequency: float = 0.5
    index: int = 0
    phase: float = 0
    cache: np.ndarray = field(init=False)
    coefficients: np.ndarray = field(init=False)
    
    def __post_init__(self):
        self.cache = np.zeros(self.order)
        self.coefficients = np.zeros(self.order)
    
    def prime(self, lpc: LPC, frequency: float) -> None:
        """Call before the first call to play with the first frame to be played to set up
        startingn values"""
        if self.order != lpc.order():
            raise AttributeError(f'Order of LPCPlayer {self.order} does not match order of LPC {lpc.order()}')
        self.gain = lpc.gain
        self.voice = lpc.voice
        self.coefficients = lpc.coefficients.copy()
        self.cache = np.zeros(self.order)
        self.frequency = frequency
        self.index = 0
        self.phase = 0
    
    def play(self,
             lpc: LPC,
             frequency: float,
             n_samples: int,
             funcid: int=0,
             pm: Tuple[float, float]=(0,0)) -> np.ndarray:
        """Plays an LPC and returns the array of samples"""
        if self.order != lpc.order():
            raise AttributeError(f'Order of LPCPlayer {self.order} does not match order of LPC {lpc.order()}')
        (samples, self.frequency, self.coefficients, self.gain,\
            self.voice, self.cache, self.index, self.phase) =\
            _fast_play(n_samples, frequency, lpc.coefficients, lpc.gain, lpc.voice, self.frequency,\
                self.coefficients, self.gain, self.voice, self.speed, self.cache, self.index,\
                self.phase, _fast_funcs[funcid], *pm)
        return samples