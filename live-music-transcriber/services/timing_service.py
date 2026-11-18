from services import note_detection as nd

NOTE_NAMES = ['whole', 'half', 'quarter', 'eighth', 'sixteenth']  # each lasts half as long as the one before

class TimingService:

    def __init__(self, tempo: int, sr: int, time_signature: list[int]):
        top, bottom = time_signature
        if bottom & (bottom - 1):
            raise ValueError(f"Bottom of a time signature must be a power of 2, got {top}/{bottom}")
        self.time_signature = (top, bottom)
        self.tempo = tempo
        self.sr = sr
        self.beat_time = 60 / tempo
        self.beat_dvision = self.divison_of_beats(top)
        self.beats = self.num_of_beats(top)
        self.beat_frames = self.beat_time * sr / nd.HOP_LENGTH
        self.measure_frames = self.beats * self.beat_frames
        self.note_frames = self.note_value_frames() 
        # three eighth triplets fill a quarter note. Compound meters already split the beat in three, so none there
        self.triplet_frames = None if self.is_compound(top) else self.plain_note_frames()['quarter'] / 3
        self.measure = 1  # number of the measure being filled
        self.position = 0  # frames from the downbeat of measure 1 to the end of the last placed note
        self.pending = []  # (note, frames) held back while they might be the start of a triplet

    # takes a finished note (or rest) and returns the pieces ready to print (see place). Notes about a triplet long are
    # held back: three in a row that start on a quarter-note beat and add up to a quarter note are an eighth triplet.
    # Anything else lets the held notes go as ordinary notes.
    def add(self, note, frames, snap, min_rest):
        if self.triplet_frames and note and abs(frames - self.triplet_frames) <= snap and (self.pending or self.on_quarter(snap)):
            self.pending.append((note, frames))
            if len(self.pending) < 3:
                return []
            group, self.pending = self.pending, []
            name = 'Eighth Triplet' if abs(sum(f for _, f in group) - 3 * self.triplet_frames) <= snap else None
            return [piece for n, f in group for piece in self.place(n, f, snap, min_rest, name)]
        return self.flush(snap, min_rest) + self.place(note, frames, snap, min_rest)

    # lets go of any held-back notes as ordinary notes
    def flush(self, snap, min_rest):
        group, self.pending = self.pending, []
        return [piece for n, f in group for piece in self.place(n, f, snap, min_rest)]

    # whether the next note would start within `snap` of a quarter-note beat
    def on_quarter(self, snap):
        quarter = 3 * self.triplet_frames
        offset = (self.position - round((self.measure - 1) * self.measure_frames)) % quarter
        return min(offset, quarter - offset) <= snap

    # lines a finished note (or rest, note == '') up with the measures, splitting it at bar lines, and returns the pieces
    # as (measure number, note, frames, denomination). A rest is broken into the longest rests that fit (see rest_values).
    # Less than `snap` frames over a bar line isn't split off: it's timing slop, not a note.
    # `name` overrides the note value (used for triplets).
    def place(self, note, frames, snap, min_rest, name=None):
        pieces = []
        while frames > 0:
            to_bar = self.frames_to_bar()
            if to_bar < snap:  # starts on (or a hair before) the bar line, so it belongs to the next measure
                self.close_measure()
                continue
            part = to_bar if frames - to_bar >= snap else frames
            if note:
                pieces.append((self.measure, note, part, f'{name or self.denomination(part)} Note'))
            else:
                for name, length in self.rest_values(part, snap, min_rest):
                    pieces.append((self.measure, note, length, f'{name} Rest'))
            self.position += part
            frames -= part
        return pieces

    # splits a silence into rests, longest first: take the longest rest that fits, subtract it, and repeat until
    # less than `min_rest` frames are left (that leftover is dropped). Dotted values are skipped, so a dotted
    # quarter of silence becomes a quarter rest and an eighth rest. A rest up to `snap` frames longer than what's
    # left still fits: a silence measured a few frames short of a quarter is still a quarter rest.
    def rest_values(self, frames, snap, min_rest):
        values = [(name, length) for name, length in self.note_frames.items() if not name.startswith('Dotted')]
        rests = []
        while frames >= min_rest:
            fit = next(((name, length) for name, length in values if length <= frames + snap), None)
            if fit is None:
                break
            name, length = fit
            rests.append((name, min(length, frames)))
            frames -= length
        return rests

    # frames from the end of the last note (placed or held back) to the next bar line
    def frames_to_bar(self):
        return round(self.measure * self.measure_frames) - self.position - sum(f for _, f in self.pending)

    def close_measure(self):
        self.measure += 1

    # returns total number of beats in a measure based on time signature
    def num_of_beats(self, meter):
        if self.is_compound(meter):
            return meter // 3 
        return meter

    # returns how beats are divided based on compound or simple meter
    def divison_of_beats(self, meter):
        return 3 if self.is_compound(meter) else 2

    def is_compound(self, meter):
        return meter > 3 and meter % 3 == 0

    # frames each note value lasts at this tempo, longest first, skipping values longer than a measure
    def note_value_frames(self):
        values = {}
        for name, frames in self.plain_note_frames().items():
            values[f'Dotted {name.title()}'] = frames * 1.5
            values[name.title()] = frames
        return self.longest_first(values)

    # frames of each plain (undotted) note, starting from the beat note and doubling / halving from there
    def plain_note_frames(self):
        top, bottom = self.time_signature
        beat_index = bottom.bit_length() - 1  # position of the bottom note in NOTE_NAMES: 4 -> quarter, 8 -> eighth
        beat_note_frames = self.beat_frames
        if self.is_compound(top):  # the beat is a dotted note one size up: 6/8 is counted in dotted quarters
            beat_index -= 1
            beat_note_frames /= 1.5
        return {name: beat_note_frames * 2 ** (beat_index - i) for i, name in enumerate(NOTE_NAMES)}

    # sorts longest first and rounds to whole frames, dropping values longer than a measure
    def longest_first(self, values):
        return {name: round(frames) for name, frames in sorted(values.items(), key=lambda v: -v[1])
                if frames <= self.measure_frames * (1 + 1e-9)}  # 1e-9: float rounding on exact fits

    # closest note value to a length in frames (by relative difference)
    def denomination(self, frames):
        return min(self.note_frames, key=lambda name: abs(frames - self.note_frames[name]) / self.note_frames[name])
