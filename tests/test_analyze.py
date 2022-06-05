import math
import numpy as np
import random
import struct
import wave

from vocoder import analyze, lpc

with wave.open('palparkpace.wav', 'r') as wav:
    n: int = wav.getnframes()
    c: int = wav.getnchannels()
    width: int = wav.getsampwidth()
    rate: int = wav.getframerate()
    frames = wav.readframes(n)
    print(wav.getparams())

signal = np.zeros(n)
for i, _ in enumerate(signal):
    s = 0
    for j in range(i * width * c + width, i * width * c, -1):
        s <<= 8
        s += frames[j - 1]
    if s >= 1 << (width * 8 - 1):
        s -= 1 << (width * 8)
    signal[i] = s * 2 ** -(width * 8 - 1)

order = 48
step = 441 * 1

player = lpc.LPCPlayer(order)

frames = analyze.analyze(signal, order, step * 2, step, None)
print(frames[0])

output = []
cache = [0] * order
cache_i = 0
m, M = 0, 0
# coeffs = frames[0][0].copy()
i = 0
phase = 0
freq = 150
# voice = frames[0][2]
# gain = frames[0][1]

threshold = 1e-10
decay = 300
interp = 1 - threshold ** (1 / decay)

for __ in range(1):
    output += [0] * (44100 // 7)
    player.prime(frames[0], freq / rate)
    
    for frame in frames:
        output += player.play(frame, (freq - 25 + random.random() * 50) / rate, step).tolist()
        # ac: float = frame[2]
        
        # ac = ac ** 2
        
        # for _ in range(step * 1):
            # pulse = (phase % (44100 / freq)) / (44100 / freq) * 2 - 1
            # voice = voice + (ac - voice) * interp
            # pulse *= voice
            # pulse += (1 - ac) * random.random() * 2 - 1
            # phase += 1# + math.sin(i * math.pi * 2 / 44100 * 1.4) * .1
            # i += 1
            # for j in range(order):
                # # interp = 2 ** -8
                # ocoef = coeffs[j]
                # ncoef = frame[0][j]
                # coeff = ocoef + (ncoef - ocoef) * interp
                # coeffs[j] = coeff
                # new_pulse = pulse - cache[cache_i - j - 1] * coeff
                # if np.isinf(new_pulse):
                    # print(pulse, new_pulse, coeff, interp, cache[cache_i - j - 1], frame, coeffs, cache)
                    # exit(1)
                # pulse = new_pulse
            # cache[cache_i] = pulse
            # cache_i = (cache_i + 1) % order
            # gain = gain + (frame[1] - gain) * interp
            # pulse *= gain ** .5
            # m = min(m, pulse)
            # M = max(M, pulse)
            # output.append(pulse)
    
    output += [0] * (44100 // 7)

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
    