import pickle

from pathlib import Path

import music21.stream as m_stream
import music21.pitch as m_pitch
import music21.chord as m_chord
import music21.note as m_note

from . import INVPITCH_TOKENIZER_PATH


def get_parts_from_stream(piece):
    parts = [part for part in piece if isinstance(part, m_stream.Part)]
    return parts


def pitch_tokeniser_maker():
    post = {"end": 0}
    for i in map(str, range(36, 82)):
        post[i] = len(post)
    post["Rest"] = len(post)

    return post


def load_tokeniser():
    with open(INVPITCH_TOKENIZER_PATH, "rb") as tokenizer_fp:
        dic = pickle.load(tokenizer_fp)
    return dic


def chord_from_pitches(pitches):
    chord = m_chord.Chord(map(int, (pitch for pitch in pitches if pitch >= 36)))

    try:
        root = m_pitch.Pitch(chord.root()).pitchClass
    except:
        return 49

    match chord.quality:
        case "major":
            return root
        case "minor":
            return root + 12
        case "diminished":
            return root + 24
        case "augmented":
            return root + 36
        case "other":
            return 48


def indices_to_stream(token_list, filename=None):
    # Load tokeniser
    with open(INVPITCH_TOKENIZER_PATH, "rb") as tokenizer_fp:
        inverse_t = pickle.load(tokenizer_fp)

    # Define stream parts: voices soprano, alto, tenor & bass
    sop_part = m_stream.Part()
    sop_part.id = "soprano"

    alto_part = m_stream.Part()
    alto_part.id = "alto"

    tenor_part = m_stream.Part()
    tenor_part.id = "tenor"

    bass_part = m_stream.Part()
    bass_part.id = "bass"

    score = m_stream.Stream([sop_part, bass_part, alto_part, tenor_part])
    for j, token in enumerate(token_list.squeeze().numpy()):
        try:
            note = inverse_t[token]
        except:
            continue

        if isinstance(note, m_note.Rest) or note == "Rest":
            n = m_note.Rest()
        else:
            n = m_note.Note(int(note))

        dur = 0.25
        n.quarterLength = dur

        score[(j % 5) - 1].append(n)

    # If filename is given, save stream to file.
    if filename:
        score.write("midi", fp=Path(filename))
        print(f"SAVED sample to {str(filename)}")

    return score


def smooth_rhythm(stream, filename):
    score = m_stream.Stream()
    parts = get_parts_from_stream(stream)

    for part in parts:
        new_part = m_stream.Part()

        current_pitch = -1
        current_offset = 0.0
        current_dur = 0.0

        for n in part.notesAndRests.stream().flatten():
            if isinstance(n, m_note.Rest):
                if current_pitch == 129:
                    current_dur += 0.25
                else:
                    if current_pitch > -1:
                        if current_pitch < 128:
                            note = m_note.Note(current_pitch)
                        else:
                            note = m_note.Rest()
                        note.quarterLength = current_dur
                        new_part.insert(current_offset, note)

                        current_pitch = 129
                        current_offset = n.offset
                        current_dur = 0.25

            else:
                if n.pitch.midi == current_pitch:
                    current_dur += 0.25
                else:
                    if current_pitch > -1:
                        if current_pitch < 128:
                            note = m_note.Note(current_pitch)
                        else:
                            note = m_note.Rest()
                        note.quarterLength = current_dur
                        new_part.insert(current_offset, note)

                    current_pitch = n.pitch.midi
                    current_offset = n.offset
                    current_dur = 0.25

        if current_pitch < 128:
            note = m_note.Note(current_pitch)
        else:
            note = m_note.Rest()
        note.quarterLength = current_dur
        new_part.insert(current_offset, note)

        score.append(new_part)

    score.write("midi", fp=Path(filename))
    print(f"SAVED rhythmically 'smoothed' sample to {str(filename)}")
