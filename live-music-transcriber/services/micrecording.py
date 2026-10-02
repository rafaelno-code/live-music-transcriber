import sounddevice as sd
import numpy as np

SAMPLE_RATE = sd.default.samplerate = 48000  
sd.default.channels = 1

def record_audio():
    duration = 10
    array = sd.rec(int(duration * SAMPLE_RATE), dtype=np.float32)
    sd.wait()
    return array