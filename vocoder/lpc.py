import io
from dataclasses import dataclass, field
import json
import numpy as np
import random
from typing import Optional

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
    speed: float = .074
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
    
    def play(self, lpc: LPC, frequency: float, n_samples: int) -> np.ndarray:
        """Plays an LPC and returns the array of samples"""
        if self.order != lpc.order():
            raise AttributeError(f'Order of LPCPlayer {self.order} does not match order of LPC {lpc.order()}')
        samples: np.ndarray = np.zeros(n_samples)
        for i in range(n_samples):
            pulse: float = (self.phase % 1) * 2 - 1 # Sawtooth in [-1, 1]
            noise: float = random.random() * 2 - 1 # Noise in [-1, 1]
            self.voice += (lpc.voice - self.voice) * self.speed
            pulse = noise + (pulse - noise) * self.voice
            for j in range(self.order):
                coeff: float = self.coefficients[j]
                coeff += (lpc.coefficients[j] - coeff) * self.speed
                pulse -= self.cache[self.index - 1 - j] * coeff
                self.coefficients[j] = coeff
            self.cache[self.index] = pulse
            self.index = (self.index + 1) % self.order
            self.gain += (lpc.gain - self.gain) * self.speed
            samples[i] = min(1.0, max(-1.0, pulse * self.gain ** .5))
            self.frequency += (frequency - self.frequency) * self.speed
            self.phase += self.frequency
        return samples