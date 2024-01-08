import pickle

import music21.stream as m_stream
import music21.pitch as m_pitch
import music21.chord as m_chord

"""
Contains utility functions for dataset preprocessing
"""


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
    with open("tokenisers/pitch_only.p", "rb") as fp:
        dic = pickle.load(fp)
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
