import json
import math
import numpy as np
import random
import struct
import wave

import numpy as np

from lpyc_tts_shotgunllama import lpc
from lpyc_tts_shotgunllama.player import phoneme

order = 48
step = 441 * 1

player = lpc.LPCPlayer(order)

phonology = phoneme.Phonology.load([
    'a', 'aa', 'ae', 'au',
    'e', 'ee',
    'i',
    'u', 'uu',
    'o', 'oo',
    'y', 'w', 'r', 'h',
    'l', 'm', 'n', 'ng',
    's', 'z', 'f', 'v', 'c', 'j', 'th', 'thh',
    'p', 'b', 'k', 'g', 't', 'd'
], '.')
# output = phonology.play_str("'i-z-'thh-e-r-'s-U-m-th-ee-ng-g-'y-uu-'w-A-N-t .", base_freq=120)
output = phonology.sing_str("p-ae-k", base_freq=100, duration=1.5, funcid=0, vibrato=0.01, pm=(.3, .5))
output = np.concatenate((
    output,
    phonology.sing_str("m-aa-y-n", base_freq=100 * 2 ** (2/12), duration=1.5, funcid=0, vibrato=0.01, pm=(.1, .5))))
output = np.concatenate((
    output,
    phonology.sing_str("i-z", base_freq=100 * 2 ** (3/12), duration=1.5, funcid=0, vibrato=0.01, pm=(.5, .5))))
output = np.concatenate((
    output,
    phonology.sing_str("h-a-a-y", base_freq=100 * 2 ** (5/12), duration=1.5, funcid=0, vibrato=0.01, pm=(.5, .5))))
output1 = output

output = phonology.sing_str("p-a", base_freq=100, duration=1.5, funcid=0, vibrato=0.01, pm=(.3, .5))
output = np.concatenate((
    output,
    phonology.sing_str("g-i-n", base_freq=100 * 2 ** (2/12), duration=1.5, funcid=0, vibrato=0.01, pm=(.1, .5))))
output = np.concatenate((
    output,
    phonology.sing_str("d-a", base_freq=100 * 2 ** (3/12), duration=1.5, funcid=0, vibrato=0.01, pm=(.5, .5))))
output = np.concatenate((
    output,
    phonology.sing_str("g-i-n", base_freq=100 * 2 ** (5/12), duration=1.5, funcid=0, vibrato=0.01, pm=(.5, .5))))
keep = min(len(output), len(output1))

output = output[:keep] * .5 + output1[:keep] * .5


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
    