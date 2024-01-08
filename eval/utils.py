import pickle

from pathlib import Path

import music21.stream as m_stream
import music21.note as m_note

import matplotlib.pyplot as plt


from preprocessing.utils import get_parts_from_stream


def indices_to_stream(token_list, filename=None):
    # Load tokeniser
    with open("tokenisers/inverse_pitch_only.p", "rb") as fp:
        inverse_t = pickle.load(fp)

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


def plot_loss_acc_curves(log: Path | str):
    train_loss, train_acc = [], []
    val_loss, val_acc = [], []

    # Read a given log file, if exists
    with open(log, "r") as f:
        txt = f.read()

    for line in txt.split("\n"):
        if "finished" in line:
            components = line.split(" ")
            loss = components[5]
            loss = loss[:-1]
            acc = components[7]

            if "train phase" in line:
                train_acc.append(float(acc))
                train_loss.append(float(loss))
            else:
                val_acc.append(float(acc))
                val_loss.append(float(loss))

    plt.figure(1)
    plt.subplot(121)
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.xlabel("epochs")
    plt.legend(["train loss", "val loss"], loc="upper left")
    plt.ylim(0, 6)

    plt.subplot(122)
    plt.plot(train_acc)
    plt.plot(val_acc)
    plt.xlabel("epochs")
    plt.legend(["train acc", "val acc"], loc="upper left")
    plt.ylim(0, 100)
    plt.show()

    plt.show()
