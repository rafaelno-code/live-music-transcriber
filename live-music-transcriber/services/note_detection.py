import librosa
import numpy as np

fmin = librosa.note_to_hz('C2')
fmax = librosa.note_to_hz('C7')

FRAME_LENGTH = 2048
HOP_LENGTH = 512

ONSET_LOOKBACK = FRAME_LENGTH // HOP_LENGTH  # frames until a window no longer overlaps the current one

def detect_frequencies(audio, sr, noise_floor, prev_rms, min_prob=0.3, onset_ratio=1.5):
    f0, _, voiced_prob = librosa.pyin(audio, sr=sr, fmin=fmin, fmax=fmax,
                                      frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH, center=False)
    rms = librosa.feature.rms(y=audio, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH, center=False)[0]
    f0[(rms < noise_floor) | (voiced_prob < min_prob)] = np.nan
    onsets = detect_onsets(rms, prev_rms, noise_floor, onset_ratio)
    return f0, onsets, rms[-(ONSET_LOOKBACK + 1):]

def detect_onsets(rms, prev_rms, noise_floor, ratio):
    """Flag the frames where loudness starts jumping above its recent minimum."""
    # inf padding: frames with no history yet can never look like a jump
    history = np.concatenate([np.full(ONSET_LOOKBACK + 1, np.inf), prev_rms, rms])[-(len(rms) + ONSET_LOOKBACK + 1):]
    recent_min = np.lib.stride_tricks.sliding_window_view(history, ONSET_LOOKBACK)[:-1].min(axis=1)
    level = history[ONSET_LOOKBACK:]
    rising = (level > noise_floor) & (level > ratio * recent_min)
    return rising[1:] & ~rising[:-1]  # only the first frame of each jump

def note_name(note):
    return librosa.midi_to_note(note) if note else ''

def snap_to_note(midi, current, candidate, tolerance):
    if np.isnan(midi):
        return ''
    for held in (current, candidate):
        if held and abs(midi - held) <= tolerance:
            return held
    return int(round(midi))

def track_notes(f0, onsets, state, sr, min_frames=10, min_rest_frames=None, tolerance=0.7):
    current, frames, candidate, candidate_frames, split_at = state
    finished = []
    for midi, onset in zip(librosa.hz_to_midi(f0), onsets):
        note = snap_to_note(midi, current, candidate, tolerance)
        if current is None:
            current = note
        if onset and current and split_at is None and frames >= min_frames:
            split_at = frames  # possible re-attack: where current would end if the same note carries on
        if note == current:
            frames += 1 + candidate_frames  # glitch over, keep its frames
            candidate, candidate_frames = None, 0
            if split_at is not None and frames - split_at >= min_frames:
                finished.append((note_name(current), split_at))
                frames -= split_at
                split_at = None
        elif note == candidate:
            candidate_frames += 1
            octave_jump = note and current and (note - current) % 12 == 0
            if candidate_frames >= (min_rest_frames if not note or octave_jump else min_frames):
                finished.append((note_name(current), frames))
                current, frames = candidate, candidate_frames
                candidate, candidate_frames = None, 0
                split_at = None  # the attack belonged to the new note
        else:
            frames += candidate_frames
            candidate, candidate_frames = note, 1
    return finished, (current, frames, candidate, candidate_frames, split_at)
