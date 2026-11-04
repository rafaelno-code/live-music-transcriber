from services import mic_recording as mic, note_detection as nd
from services.timing_service import TimingService
import librosa
import numpy as np
import time

def describe(note, frames, denomination):
    return f"{note or 'Rest'} - {denomination} ({frames} frames)"

# lines finished notes up with the measures and prints them, with a header whenever a new measure starts
def show(timing, finished, snap, min_rest, shown_measure):
    for note, length in finished:
        for measure, note, frames, denomination in timing.place(note, length, snap, min_rest):
            if measure != shown_measure:
                print(f"\n--- Measure {measure} ---")
                shown_measure = measure
            if frames >= snap:  # anything shorter is timing slop around a bar line, not a note or rest
                print(describe(note, frames, denomination))
    return shown_measure

# keeps the measures moving in real time: a new measure starts as soon as the audio reaches its bar line.
# A note held (or a rest still going) over the bar line is shown up to the bar line right away; the rest of it
# shows in the new measure once it ends
def advance_measures(timing, state, snap, min_rest, shown_measure):
    current, frames, candidate, candidate_frames, split_at = state
    started = current is not None
    # within `snap` of the bar line counts as there: the last few frames of each chunk are only analysed with the next one
    while timing.frames_to_bar() < snap or (started and frames >= timing.frames_to_bar() - snap):
        part = min(frames, timing.frames_to_bar()) if started else 0
        if part >= snap:
            shown_measure = show(timing, [(nd.note_name(current), part)], snap, min_rest, shown_measure)
            frames -= part
            if split_at is not None:  # keep a possible re-attack pointing at the same frame
                split_at = split_at - part if split_at > part else None
        timing.close_measure()
        shown_measure = timing.measure
        print(f"\n--- Measure {shown_measure} ---")
    return (current, frames, candidate, candidate_frames, split_at), shown_measure

def count_in(timing, measures=2):
    for _ in range(measures):
        for beat in range(1, timing.beats + 1):
            print(beat, end=' ', flush=True)
            time.sleep(timing.beat_time)
        print()

def main():
    sr = mic.SAMPLE_RATE
    tempo = int (input("Enter tempo (BPM): "))
    time_signature = list(map(int, input("Enter time signature (e.g., 4/4): ").split('/')))
    timing = TimingService(tempo, sr, time_signature)
    min_frames = timing.beat_frames / 8  # minimum length of a note in frames

    min_rest_frames = timing.beat_frames / 4  # shorter silences are the gap between two notes, not a rest
    chunk_seconds = timing.beat_time
    with mic.open_stream() as stream:
        try:
            print("Reading room... needs silence")
            ambient = mic.read_chunk(1.0)
            noise_floor = 3 * librosa.feature.rms(y=ambient)[0].mean()
            print("Count-in:")
            count_in(timing)
            mic.discard_buffered()  # recording starts on beat 1, not during the count-in
            # audio reaches us stream.latency late, so the first that much was captured before the downbeat
            buffer = mic.read_chunk(stream.latency)[int(stream.latency * sr):]
            prev_rms = np.empty(0, dtype=np.float32)
            state = (None, 0, None, 0, None)
            print("Listening... (Ctrl+C to stop)")
            shown_measure = None
            while True:
                buffer = np.concatenate([buffer, mic.read_chunk(chunk_seconds)])
                n_frames = 1 + (len(buffer) - nd.FRAME_LENGTH) // nd.HOP_LENGTH
                f0, onsets, prev_rms = nd.detect_frequencies(buffer, sr, noise_floor, prev_rms)
                buffer = buffer[n_frames * nd.HOP_LENGTH:]  # keep the tail for the next chunk

                finished, state = nd.track_notes(f0, onsets, state, min_frames, min_rest_frames)
                shown_measure = show(timing, finished, min_frames, min_rest_frames, shown_measure)
                state, shown_measure = advance_measures(timing, state, min_frames, min_rest_frames, shown_measure)
        except KeyboardInterrupt:
            current, frames, _, candidate_frames, _ = state
            show(timing, [(nd.note_name(current), frames + candidate_frames)], min_frames, min_rest_frames, shown_measure)
            print("\nStopped.")

if __name__ == "__main__":
    main()
