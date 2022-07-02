from dataclasses import dataclass, field
import json
import numpy as np
from os import path
import random
from typing import List, Dict, ClassVar, Tuple

from lpyc_tts_shotgunllama import lpc

@dataclass
class Phoneme:
    frames: List[lpc.LPC]
    continuous: bool
    framerate: int
    
    def play_on(self, player: lpc.LPCPlayer, duration: float, frequency: float,\
            prime: bool = False, *, frame_size: float = .01, vibrato: float = 0,
            funcid: int=0, pm: Tuple[float, float]=(0,0)) -> np.ndarray:
        if duration < 0 or not self.continuous:
            n_frames: int = len(self.frames)
            i_frames: List[int] = list(range(n_frames))
        else:
            n_frames: int = int((duration + frame_size * .00099) // frame_size)
            i_frames: List[int] = [random.choice(range(len(self.frames))) for _ in range(n_frames)]
        n_samples: int = round(frame_size * self.framerate)
        samples: np.ndarray = np.zeros(n_samples * n_frames)
        
        if prime:
            player.prime(self.frames[i_frames[0]],\
                frequency / self.framerate)
        
        v_accum: float = 0
        i_frame: int
        for i, i_frame in enumerate(i_frames):
            v_accum += random.random() * vibrato - vibrato / 2
            v_accum = min(vibrato, max(-vibrato, v_accum))
            samples[i*n_samples : (i+1)*n_samples] =\
                player.play(self.frames[i_frame],\
                frequency * (1 + v_accum) / self.framerate,\
                n_samples, funcid, pm)
                
        return samples
    
    @staticmethod
    def fromdict(d: dict) -> 'Phoneme':
        frames: List[lpc.LPC] = list(map(lpc.LPC.fromdict, d['frames']))
        return Phoneme(frames, d['continuous'], d['framerate'])

@dataclass
class Phonology:
    phonemes: Dict[str, Phoneme]
    framerate: int = field(init=False)
    player: lpc.LPCPlayer = field(init=False)
    
    _len_markers: ClassVar[Dict[str, float]] = {
        '!': 1.5,
        '~': 1 / 1.5
    }
    _freq_markers: ClassVar[Dict[str, float]] = {
        '>': 2 ** (1/24),
        '<': 2 ** (-1/24)
    }
    _rest_markers: ClassVar[Dict[str, float]] = {
        ',': .2,
        ';': .4,
        '.': .6
    }
    
    def __post_init__(self) -> None:
        if self.phonemes:
            first: Phoneme = next(iter(self.phonemes.values()))
            self.player = lpc.LPCPlayer(first.frames[0].order())
            self.framerate = first.framerate
    
    def play_str(self, sentence: str, *, base_freq: float = 100, phoneme_len: float = .15,\
            vibrato: float = .03) -> np.ndarray:
        samps: np.ndarray = np.array([], dtype=float)
        words: List[str] = sentence.split()
        word: str
        for word in words:
            prime: bool = True
            sounds: List[str] = word.split('-')
            sound: str
            for sound in sounds:
                if sound[0] == "'":
                    sound = sound[1:]
                    prime = True
                rest: float = 0
                while sound and sound[0] in Phonology._rest_markers:
                    rest += Phonology._rest_markers[sound[0]]
                    sound = sound[1:]
                if rest:
                    samps = np.concatenate((samps, np.zeros(round(rest * self.framerate))))
                lenmul: float = 1
                while sound and sound[0] in Phonology._len_markers:
                    lenmul *= Phonology._len_markers[sound[0]]
                    sound = sound[1:]
                freqmul: float = 1
                while sound and sound[0] in Phonology._freq_markers:
                    freqmul *= Phonology._freq_markers[sound[0]]
                    sound = sound[1:]
                if sound.isupper():
                    sound = sound.lower()
                    freqmul *= 2 ** (1/6)
                if sound and sound in self.phonemes:
                    samps = np.concatenate((samps, self.phonemes[sound].play_on(\
                        self.player, phoneme_len * lenmul, base_freq * freqmul, prime,\
                        vibrato=vibrato)))
                    prime = False
            samps = np.concatenate((samps, np.zeros(round(self.framerate * .1))))
        return samps
    
    def sing_str(self, sentence: str, *, base_freq: float = 100, phoneme_len: float = .15,
            duration: float=.25, vibrato: float = .03, funcid: int=0,
            true_vib: Tuple[float, float]=(0,0),
            pm: Tuple[float, float]=(0,0)) -> np.ndarray:
        samps: np.ndarray = np.array([], dtype=float)
        words: List[str] = sentence.split()
        word: str
        for word in words:
            prime: bool = True
            sounds: List[str] = word.split('-')
            sound: str
            len_left: float = duration
            lens: List[float] = [-1] * len(sounds)
            cont_ctr: int = 0
            for i, sound in enumerate(sounds):
                phon = self.phonemes[sound]
                if not phon.continuous:
                    lens[i] = len(phon.frames) * .01
                    len_left -= lens[i]
                else:
                    cont_ctr += 1
            for i, sound in enumerate(sounds):
                phon = self.phonemes[sound]
                if phon.continuous:
                    lens[i] = len_left / cont_ctr
                if sound and sound in self.phonemes:
                    samps = np.concatenate((samps, self.phonemes[sound].play_on(\
                        self.player, lens[i], base_freq, prime,\
                        vibrato=vibrato, pm=pm, funcid=funcid)))
                    prime = False
        return samps
    
    @staticmethod
    def load(names: List[str], basedir: str) -> 'Phonology':
        phonemes: Dict[str, Phoneme] = {}
        name: str
        for name in names:
            try:
                with open(path.join(basedir, name+'.json')) as file:
                    d: dict = json.load(file)
                    phoneme: Phoneme = Phoneme.fromdict(d)
                    phonemes[name.lower()] = phoneme
            except Exception as e:
                print(f'Error loading phoneme {name}: {e}')
        return Phonology(phonemes)