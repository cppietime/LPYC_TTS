import json
import math
import numpy as np
import random
import struct
import wave

from vocoder import analyze, lpc, phoneme

order = 48
step = 441 * 1

player = lpc.LPCPlayer(order)

phonology = phoneme.Phonology.load([
    'a', 'aa', 'ae', 'au',
    'e', 'ee',
    'i',
    'u', 'uu',
    'o', 'oo',
    'y', 'w', 'r',
    'l', 'm', 'n', 'ng',
    's', 'z', 'f', 'v', 'c', 'j', 'th', 'thh',
    'p', 'b', 'k', 'g', 't', 'd'
], '.')
output = phonology.play_str("'i-z-'thh-e-r-'s-U-m-th-ee-ng-g-'y-uu-'w-A-N-t .", base_freq=120)
# output = phonology.sing_str("p-a-k", base_freq=100, duration=3, funcid=0, vibrato=0, pm=(.25,4))


data = bytearray()
for sample in output:
    short = min(32767, max(-32768, int(32767 * sample)))
    data += (short & 0xffff).to_bytes(2, 'little')
data = bytes(data)

with wave.open('out.wav', 'w') as wav:
    wav.setnchannels(1)
    wav.setframerate(44100)
    wav.setsampwidth(2)
    wav.setcomptype('NONE', 'not compressed')
    wav.setnframes(len(data) // 2)
    wav.writeframes(data)
    