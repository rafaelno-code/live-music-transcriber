from services import micrecording as mic, notedetection as nd
import librosa
import numpy as np

def main():
    sr = mic.SAMPLE_RATE
    tempo = int (input("Enter tempo (BPM): "))
    time_signature = list(map(int, input("Enter time signature (e.g., 4/4): ").split('/')))
    beat_time = 60 / tempo
    beat_frames = beat_time * sr / nd.HOP_LENGTH
    min_frames = beat_frames / 4  # minimum length of a note in frames

    min_rest_frames = min_frames 
    chunk_seconds = beat_time
    with mic.open_stream():
        try:
            print("Reading room... needs silence")
            ambient = mic.read_chunk(1.0)
            noise_floor = 3 * librosa.feature.rms(y=ambient)[0].mean()
            print("Listening... (Ctrl+C to stop)")
            buffer = np.empty(0, dtype=np.float32)
            prev_rms = np.empty(0, dtype=np.float32)
            state = (None, 0, None, 0, None)
            while True:
                buffer = np.concatenate([buffer, mic.read_chunk(chunk_seconds)])
                n_frames = 1 + (len(buffer) - nd.FRAME_LENGTH) // nd.HOP_LENGTH
                f0, onsets, prev_rms = nd.detect_frequencies(buffer, sr, noise_floor, prev_rms)
                buffer = buffer[n_frames * nd.HOP_LENGTH:]  # keep the tail for the next chunk

                finished, state = nd.track_notes(f0, onsets, state, sr, min_frames, min_rest_frames)
                for note, length in finished:
                    print(f"Note: {note or 'Rest'}, Length: {length} frames")
        except KeyboardInterrupt:
            current, frames, _, candidate_frames, _ = state
            print(f"Note: {nd.note_name(current) or 'Rest'}, Length: {(frames + candidate_frames)} frames")
            print("\nStopped.")
            
if __name__ == "__main__":
    main()