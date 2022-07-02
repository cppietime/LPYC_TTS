import argparse
import math
import numpy as np
import sys
from typing import Tuple, List, Optional, Callable, Union

from lpyc_tts_shotgunllama import lpc
from lpyc_tts_shotgunllama.analyzer import windows

def calc_burg(signal: np.ndarray, max_order: int) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Calculates LPC coefficients and gains for orders 1..max_order for a given signal
    using Burg's method of minimizing forward and backward propagation error.
    
    The signal is NOT windowed by this function. Window the argument before calling
    this function if necessary.
    
    signal: an array-like of the pre-windowed signal to analyze
    max_order: the maximum order LPC coefficients and gain to calculate
    """
    signal: np.ndarray = np.asanyarray(signal, dtype=float)
    
    lpc_order_coeffs: List[np.ndarray] = []
    lpc_order_gains: np.ndarray = np.zeros(max_order)
    error_f: np.ndarray = signal.copy()
    error_b: np.ndarray = signal.copy()
    N: int = len(signal)
    rho: float = sum(abs(signal**2)) / N
    coeffs: np.ndarray = np.zeros(0)
    
    order: int
    for order in range(max_order):
        error_f = error_f[1:]
        error_b = error_b[:-1]
        num: float = -2 * np.dot(error_f, error_b)
        den: float = np.dot(error_f, error_f) + np.dot(error_b, error_b)
        reflection: float = num / den
        if reflection != reflection:
            reflection = 0
        
        rho *= 1 - reflection ** 2
        lpc_order_gains[order] = rho
        
        error_f, error_b = error_f + reflection * error_b, error_b + reflection * error_f
        coeffs = coeffs + reflection * coeffs[::-1]
        coeffs = np.concatenate((coeffs, [reflection]))
        lpc_order_coeffs.append(coeffs)
    
    return lpc_order_coeffs, lpc_order_gains

def autocorrelation(signal: np.ndarray, offset: int = 1) -> float:
    num: float = 0
    den0: float = 0
    den1: float = 0
    N: int = len(signal)
    for i in range(N - offset):
        num += signal[i] * signal[i + offset]
        den0 += signal[i] ** 2
        den1 += signal[i + offset] ** 2
    return num / (den0 * den1) ** .5

def analyze(signal: np.ndarray,\
    order: int, window_size: int, step_size: int,\
    window_type: Optional[Union[str, Callable[[np.ndarray], np.ndarray]]] = None,\
    progressive: bool = False)\
        -> Union[List[lpc.LPC], List[List[lpc.LPC]]]:
    """
    Analyzes a signal and returns a list of frames, each frame a tuple of coefficients and gain
    
    signal: array-like of input signal
    order: order of LPC returned
    window_size: length of each frame
    step_size: stride between frames
    progressive: True to return each order up to the specified max order
    """
    signal = np.asanyarray(signal, dtype=float)
    N: int = len(signal)
    
    frames: Union[List[lpc.LPC], List[List[lpc.LPC]]] = []
    if window_type is not None:
        if isinstance(window_type, str):
            if window_type.lower() == 'none':
                window_type = None
            else:
                window_type = windows.windows[window_type.lower()]
    
    start: int
    for start in range(0, N, step_size):
        sample: np.ndarray = signal[start : start + window_size]
        windowed: np.ndarray = sample
        if window_type is not None:
            windowed = window_type(windowed)
        ac: float = autocorrelation(windowed) ** 2
        _coeffs, _gain = calc_burg(windowed, order)
        if not progressive:
            coeffs: np.ndarray = _coeffs[-1]
            gain: float = _gain[-1]
            frames.append(lpc.LPC(coeffs, gain, ac))
        else:
            frame: List[lpc.LPC] = []
            for o in range(order):
                frame.append(lpc.LPC(_coeffs[o], _gain[0], ac))
            frames.append(frame)
    
    return frames

def main():
    import io, json, sys, wave
    parser: argparse.ArgumentParser = argparse.ArgumentParser('Read a WAV file and convert it to saved LPC data')
    parser.add_argument('-o', '--order', type=int, required=True, help='Filter order')
    parser.add_argument('-f', '--window_type', type=str, default='none', help='Type of windowing function(none, Hann, Hamming, or Welch)')
    parser.add_argument('-s', '--step_size', type=float, default=.01, help='Stride of step size in seconds')
    parser.add_argument('-w', '--window_size', type=float, default=0, help='Duration of window in seconds')
    parser.add_argument('ipath', type=str, help='Path to input .WAV file')
    parser.add_argument('opath', type=str, nargs='?', default='', help='Path to output .WAV file')
    args: argparse.Namespace = parser.parse_args()
    try:
        with wave.open(args.ipath, 'r') as file:
            rate: int = file.getframerate()
            channels: int = file.getnchannels()
            width: int = file.getsampwidth()
            nframes: int = file.getnframes()
            frames: bytes = file.readframes(nframes)
    except Exception as e:
        print(f'Could not open wav file {args.ipath}: {e}', file=sys.stderr)
        exit(1)
    step_size: int = int(rate * args.step_size)
    window_size: int = int(rate * (args.window_size or (args.step_size * 2)))
    samples: np.ndarray = np.zeros(nframes)
    stride: int = channels * width
    for n in range(nframes):
        sample: int = int.from_bytes(frames[n * stride : n * stride + width],\
            'little',\
            signed = width > 1)
        if width == 1:
            sample = round(sample * 255 / 127) - 128
        sample /= 1 << (width * 8 - 1)
        samples[n] = sample
    frames: List[lpc.LPC] = analyze(samples, args.order, window_size, step_size, args.window_type)
    output: io.IOBase
    try:
        if not args.opath:
            output = sys.stdout
        else:
            output = open(args.opath, 'w')
        json.dump({
            'framerate': rate,
            'step_size': step_size,
            'window_size': window_size,
            'window_type': args.window_type,
            'order': args.order,
            'frames': list(map(lpc.LPC.todict, frames))
        }, output)
        if output is not sys.stdout:
            output.close()
    except Exception as e:
        print(f'Error writing to specified output {e}', file=sys.stderr)
        exit(2)

if __name__ == '__main__':
    main()