import librosa
import numpy as np

fmin = librosa.note_to_hz('C2')
fmax = librosa.note_to_hz('C7')

def detect_frequencies(audio, sr):
    audio = audio.flatten()
    return librosa.pyin(audio, sr=sr, fmin=fmin, fmax=fmax, hop_length=512)[0]

def detect_notes(frequencies):
    return [librosa.hz_to_note(f) for f in frequencies]

def get_notes_length(notes, sr):
    note_lengths = [] # in seconds
    current_note = notes[0]
    frames = 0
    for i in range(len(notes)):
        if notes[i] == current_note:
            frames += 1
        else:
            time = librosa.frames_to_time(frames, sr=sr, hop_length=512)
            note_lengths.append((current_note, time))
            current_note = notes[i]
            frames = 1
    note_lengths.append((current_note, frames * 512 / sr))
    
    return note_lengths

