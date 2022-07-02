import math
import numpy as np

def sine_window(alpha: float) -> np.ndarray:
    def _fn(signal: np.ndarray) -> np.ndarray:
        """
        Applies a sine-window to the provided signal and returns the result
        Does not modify the input parameter
        """
        N: int = len(signal)
        windowed: np.ndarray = np.asanyarray(signal, dtype=float)
        
        n: int
        for n in range(N):
            window: float = alpha - (1 - alpha) * math.cos(2 * math.pi * (n + 1) / (N + 1))
            windowed[n] *= window
        
        return windowed
    return _fn

hann = sine_window(0.5)
hamming = sine_window(0.54)

def welch(signal: np.ndarray) -> np.ndarray:
    """
    Applies a Welch window to the provided signal and returns the result
    Does not modify the input parameter
    """
    N: int = len(signal)
    windowed: np.ndarray = np.asanyarray(signal, dtype=float)
    
    n: int
    for n in range(N):
        window: float = 1 - ( (n + 1 - (N + 1)/2) / ((N + 1)/2) ) ** 2
        windowed[n] *= window
    
    return windowed

windows = {
    'hann': hann,
    'hamming': hamming,
    'welch': welch
}