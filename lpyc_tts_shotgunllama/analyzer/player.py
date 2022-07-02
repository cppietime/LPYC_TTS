from dataclasses import dataclass, field
import io
import json
import numpy as np
import pyaudio as pa
import random
import sys
from threading import Lock, Thread
import tkinter as tk
from tkinter import filedialog as fd, Checkbutton, IntVar
from typing import List, Tuple, Optional, Union
import wave

from lpyc_tts_shotgunllama import lpc

@dataclass
class Console:
    frames: List[lpc.LPC]
    framerate: int
    freq: float = 90
    player: lpc.LPCPlayer = field(init=False)
    audio: pa.PyAudio = field(init=False, default_factory=pa.PyAudio)
    stream: pa.Stream = field(init=False, default=None)
    data: bytes = b'\x00'
    data_index: int = 0
    playing: bool = False
    lock: Lock = Lock()
    
    def toggle(self):
        if not self.stream:
            return
        if self.stream.is_active():
            self.stream.stop_stream()
        else:
            print('starting')
            self.data_index = 0
            self.stream.start_stream()
        self.playing = not self.playing
    
    def play(self, indices: List[int], duration: float = 0.01,\
            dst: Optional[Union[str, io.IOBase]] = None, shuffle: bool = False,\
            repeat: bool = True)\
            -> None:
        if not indices:
            return
        if shuffle:
            indices = list(indices) * max(1, 20 // len(indices))
            random.shuffle(indices)
        Thread(target = self._play, args = (indices, duration, dst, repeat)).start()
    
    def _play(self, indices: List[int], duration: float = 0.01,\
            dst: Optional[Union[str, io.IOBase]] = None, repeat: bool = True)\
            -> None:
        self.lock.acquire()
        try:
            self.player.prime(self.frames[indices[0]], self.freq / self.framerate)
            sample_array: bytearray = bytearray()
            
            index: int
            for index in indices:
                frame_played: np.ndarray = self.player.play(self.frames[index],\
                    self.freq / self.framerate,\
                    round(self.framerate * duration))
                sample_array += (frame_played * 32767).astype('<h').tobytes()
            
            if not repeat:
                sample_array += b'\x00' * (2 * 44100 // 2)
            
            if dst is None:
                # stream: pa.Stream = self.audio.open(rate=self.framerate,\
                    # channels=1, format=pa.get_format_from_width(2), output=True)
                # stream.write(bytes(sample_array))
                # stream.close()
                self.data = bytes(sample_array)
                self.data_index = 0
            else:
                with wave.open(dst, 'w') as wav:
                    wav.setnframes(len(sample_array) // 2)
                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(self.framerate)
                    wav.setcomptype('NONE', 'not compressed')
                    wav.writeframes(sample_array)
        finally:
            self.lock.release()
    
    def load(self, src: io.IOBase) -> None:
        d: dict = json.load(src)
        self.framerate = d['framerate']
        self.frames = list(map(lpc.LPC.fromdict, d['frames']))
        self.player = lpc.LPCPlayer(self.frames[0].order())
        if self.stream:
            self.stream.close()
            print('Closed')
        self.stream = self.audio.open(format=pa.get_format_from_width(2), channels=1,\
            rate=self.framerate, output=True, stream_callback=self.callback)
    
    def load_dlg(self) -> None:
        file: io.IOBase = fd.askopenfile(filetypes=(('JSON', '*.json'),), defaultextension='.json')
        if file is None:
            return
        try:
            self.load(file)
        finally:
            file.close()
    
    def callback(self, in_data: None, frame_count: int, time_info: dict, status_flags: int) ->\
            Tuple[bytes, int]:
        byte_count = frame_count * 2
        data: bytes = self.data[self.data_index:self.data_index+byte_count]
        byte_count -= len(data)
        data += self.data * (byte_count // len(self.data)) +\
            self.data[:byte_count % len(self.data)]
        self.data_index = (self.data_index + len(data))
        flag: int = pa.paContinue
        self.data_index %= len(self.data)
        return (data, flag)
    
    def save(self, start: int, end: int, shuffle: bool) -> None:
        start, end = min(start, end), max(start, end)
        file: io.IOBase = fd.asksaveasfile(filetypes=(('JSON', '*.json'),('All files', '*.*')),\
            defaultextension='.json')
        if not file:
            return
        try:
            json.dump({
                'framerate': self.framerate,
                'order': self.frames[0].order(),
                'continuous': shuffle,
                'frames': list(map(lpc.LPC.todict, self.frames[start:end]))
            }, file)
        except Exception as e:
            print(e, file=sys.stderr)
        finally:
            file.close()

def main():
    console: Console = Console([], 0)
    file: io.IOBase
    with open('d.json', 'r') as file:
        console.load(file)
    
    scale_length: int = 900
    master: tk.Tk = tk.Tk()
    
    shuffle: IntVar = IntVar()
    repeat: IntVar = IntVar(value=1)
    
    start: tk.Scale = tk.Scale(master, from_=0, to=len(console.frames)-1, length=scale_length, orient=tk.HORIZONTAL)
    end: tk.Scale = tk.Scale(master, from_=0, to=len(console.frames)-1, length=scale_length, orient=tk.HORIZONTAL)
    freq: tk.Scale = tk.Scale(master, from_=60, to=200, length=scale_length // 2, orient=tk.HORIZONTAL)
    dur: tk.Scale = tk.Scale(master, from_=1, to=20, length=scale_length // 2, orient=tk.HORIZONTAL)
    
    update = lambda _: (setattr(console, 'freq', freq.get()),\
        console.play(range(start.get(), end.get()+1), dur.get()/100,\
        shuffle=shuffle.get()!=0, repeat=repeat.get()!=0))
    
    button_start: tk.Button = tk.Button(master, text="Stop/start", command=console.toggle)
    check_shuffle: tk.Checkbutton = tk.Checkbutton(master, variable=shuffle, onvalue=1, offvalue=0,
        command = lambda: update(None))
    check_repeat: tk.Checkbutton = tk.Checkbutton(master, variable=repeat, onvalue=1, offvalue=0,
        command = lambda: update(None))
    button_save: tk.Button = tk.Button(master, text="Save",\
        command=lambda: console.save(start.get(), end.get()+1, shuffle.get()!=0))
    button_load: tk.Button = tk.Button(master, text="Load", command=lambda:\
        (console.load_dlg(), end.configure(to=len(console.frames)-1),\
        start.configure(to=len(console.frames)-1)))
    start.grid(row=0, column=0, columnspan=6)
    end.grid(row=1, column=0, columnspan=6)
    freq.grid(row=2, column=0, columnspan=3)
    dur.grid(row=2, column=3, columnspan=3)
    button_start.grid(row=3, column=0)
    check_shuffle.grid(row=3, column=1)
    check_repeat.grid(row=3, column=2)
    button_save.grid(row=3, column=3)
    button_load.grid(row=3, column=4)
    start.bind('<ButtonRelease-1>', update)
    end.bind('<ButtonRelease-1>', update)
    freq.bind('<ButtonRelease-1>', update)
    dur.bind('<ButtonRelease-1>', update)
    tk.mainloop()

if __name__ == '__main__':
    main()