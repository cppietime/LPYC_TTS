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
with open('h.json') as file:
    phoneme_h = phoneme.Phoneme.fromdict(json.load(file))
with open('e.json') as file:
    phoneme_e = phoneme.Phoneme.fromdict(json.load(file))
with open('l.json') as file:
    phoneme_l = phoneme.Phoneme.fromdict(json.load(file))
with open('o.json') as file:
    phoneme_o = phoneme.Phoneme.fromdict(json.load(file))
with open('uu.json') as file:
    phoneme_uu = phoneme.Phoneme.fromdict(json.load(file))
with open('g.json') as file:
    phoneme_g = phoneme.Phoneme.fromdict(json.load(file))
with open('aa.json') as file:
    phoneme_aa = phoneme.Phoneme.fromdict(json.load(file))
with open('s.json') as file:
    phoneme_s = phoneme.Phoneme.fromdict(json.load(file))
with open('w.json') as file:
    phoneme_w = phoneme.Phoneme.fromdict(json.load(file))
with open('r.json') as file:
    phoneme_r = phoneme.Phoneme.fromdict(json.load(file))
with open('d.json') as file:
    phoneme_d = phoneme.Phoneme.fromdict(json.load(file))
with open('a.json') as file:
    phoneme_a = phoneme.Phoneme.fromdict(json.load(file))
with open('y.json') as file:
    phoneme_y = phoneme.Phoneme.fromdict(json.load(file))
with open('u.json') as file:
    phoneme_u = phoneme.Phoneme.fromdict(json.load(file))
with open('v.json') as file:
    phoneme_v = phoneme.Phoneme.fromdict(json.load(file))
with open('p.json') as file:
    phoneme_p = phoneme.Phoneme.fromdict(json.load(file))
with open('m.json') as file:
    phoneme_m = phoneme.Phoneme.fromdict(json.load(file))

# output = np.concatenate((\
    # np.zeros(5000),\
    # phoneme_a.play_on(player, .2, 90, True),\
    # phoneme_y.play_on(player, .1, 90),\
    # np.zeros(6000),\
    # phoneme_l.play_on(player, .2, 90),\
    # phoneme_u.play_on(player, .15, 90),\
    # phoneme_v.play_on(player, .25, 85),\
    # np.zeros(6000),\
    # phoneme_y.play_on(player, .2, 90),\
    # phoneme_uu.play_on(player, .3, 84),\
    # np.zeros(9000),\
    # phoneme_p.play_on(player, -1, 88),\
    # phoneme_a.play_on(player, .2, 90),\
    # phoneme_m.play_on(player, .2, 88),\
    # phoneme_e.play_on(player, .2, 94),\
    # phoneme_l.play_on(player, .15, 90),\
    # phoneme_u.play_on(player, .2, 90),\
    # np.zeros(5000)
# ))

phonology = phoneme.Phonology.load([
    'a', 'aa', 'ae', 'au',
    'e', 'ee',
    'i',
    'u', 'uu',
    'o', 'oo',
    'y', 'w', 'r',
    's', 'z', 'f', 'v', 'c', 'j', 'th', 'thh',
    'p', 'b', 'k', 'g', 't', 'd'
], '.')
output = phonology.play_str('thh-u r-E-ee-s i-z O-uu-v-r')

# for __ in range(1):
    # output += [0] * (44100 // 7)
    # player.prime(frames[0], freq / rate)
    
    # for frame in frames:
        # output += player.play(frame, (freq - 25 + random.random() * 50) / rate, step).tolist()
    
    # output += [0] * (44100 // 7)

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
    