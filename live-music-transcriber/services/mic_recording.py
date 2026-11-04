import queue
import sounddevice as sd
import numpy as np

SAMPLE_RATE = 48000
CHANNELS = 1

audio_queue = queue.Queue()

def _callback(indata, frames, time, status):
    if status:
        print(status)
    audio_queue.put(indata.copy())  # copy: sounddevice reuses the buffer

def open_stream():
    return sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS,
                          dtype=np.float32, callback=_callback)

def read_chunk(seconds):
    """Block until `seconds` of audio has arrived, then return it as 1D."""
    needed = int(seconds * SAMPLE_RATE)
    blocks, total = [], 0
    while total < needed:
        block = audio_queue.get()
        blocks.append(block)
        total += len(block)
    return np.concatenate(blocks).flatten()
