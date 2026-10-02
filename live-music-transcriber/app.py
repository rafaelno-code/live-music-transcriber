from services import micrecording as mic, notedetection as nd


def main():
    print("Recording audio...")
    audio = mic.record_audio()
    sr = 48000  # sample rate
    print("Getting frequencies...")
    f0 = nd.detect_frequencies(audio, sr)
    print("Detecting notes...")
    notes = nd.detect_notes(f0)
    note_lengths = nd.get_notes_length(notes, sr)
    for note, length in note_lengths:
        print(f"Note: {note}, Length: {length}s")
    total_time = sum(length for _, length in note_lengths)
    print(f"Total time: {total_time}s")

if __name__ == "__main__":
    main()