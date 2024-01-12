import numpy as np

from .instruments import get_instrument

def np_perform_all_transpositions(parts, tonic, chords):
    mylist = []
    tonics = []
    my_chords = []
    try:
        t_r = np_transposable_range_for_piece(parts)
    except:
        print("error getting transpose range")
        return mylist
    lower = t_r[0]
    higher = t_r[1] + 1
    quals = [np_get_quals_from_chord(x) for x in chords]
    for i in range(lower, higher):
        try:
            roots = [(x + 12 + i) % 12 for x in chords]
            transposed_piece = np.zeros((len(parts[0]), 4), dtype=int)
            chord_prog = [
                np_chord_from_root_qual(roots[i], quals[i])
                for i in range(len(chords))
            ]
            for j in range(4):
                tp = parts[j] + i
                transposed_piece[:, j] = tp[:]
        except:
            print("ERROR: empty return")
        else:
            mylist.append(transposed_piece)
            tonics.append((tonic + i) % 12)
            my_chords.append(chord_prog)
    return mylist, tonics, my_chords


def np_transposable_range_for_part(part, inst):
    if not isinstance(inst, str):
        inst = str(inst)
    part_range = np_get_part_range(part)
    instrument = get_instrument(inst)

    lower_transposable = instrument.lowestNote - part_range[0]
    higher_transposable = instrument.highestNote - part_range[1]

    # suggests there's perhaps no musical content in this score
    if higher_transposable - lower_transposable >= 128:
        lower_transposable = 0
        higher_transposable = 0
    return min(0, lower_transposable), max(0, higher_transposable)


def np_transposable_range_for_piece(parts):
    insts = ["soprano", "alto", "tenor", "bass"]

    lower = -127
    higher = 127

    for i in range(len(parts)):
        t_r = np_transposable_range_for_part(parts[i], insts[i])
        if t_r[0] > lower:
            lower = t_r[0]
        if t_r[1] < higher:
            higher = t_r[1]
    # suggests there's perhaps no musical content in this score
    if higher - lower >= 128:
        lower = 0
        higher = 0
    return lower, higher


def np_get_part_range(part):
    mn = min(part)

    if mn < 36:
        p = sorted(part)
        c = 1
        while mn < 36:
            mn = p[c]
            c += 1

    return [mn, max(part)]


def np_convert_major_minor(piece, tonic, mode):
    _piece = piece

    for i in range(len(_piece)):
        s = _piece[i][0] if _piece[i][0] < 36 else (_piece[i][0] - tonic) % 12
        b = _piece[i][3] if _piece[i][3] < 36 else (_piece[i][3] - tonic) % 12
        a = _piece[i][1] if _piece[i][1] < 36 else (_piece[i][1] - tonic) % 12
        t = _piece[i][2] if _piece[i][2] < 36 else (_piece[i][2] - tonic) % 12

        parts = [s, a, t, b]

        for n in range(len(parts)):
            if mode == "major":
                if parts[n] in [4, 9]:
                    _piece[i][n] -= 1
            elif mode == "minor":
                if parts[n] in [3, 8, 10]:
                    _piece[i][n] += 1
            else:
                raise ValueError(f"mode must be minor or major, received {mode}")

    return _piece


def np_get_quals_from_chord(chord):
    if chord < 12:
        qual = "major"
    elif chord < 24:
        qual = "minor"
    elif chord < 36:
        qual = "diminished"
    elif chord < 48:
        qual = "augmented"
    elif chord == 48:
        qual = "other"
    else:
        qual = "none"

    return qual


def np_chord_from_root_qual(root, qual):
    if qual == "major":
        chord = root
    elif qual == "minor":
        chord = root + 12
    elif qual == "diminished":
        chord = root + 24
    elif qual == "augmented":
        chord = root + 36
    elif qual == "other":
        chord = 48
    elif qual == "none":
        chord = 49

    return chord
