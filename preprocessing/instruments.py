from collections import namedtuple

# Voice object: instance of tuple with named fields
Voice = namedtuple("Voice", ("instrumentId", "lowestNote", "highestNote"))

# Instruments
soprano_voice = Voice("soprano", 60, 81)
alto_voice = Voice("alto", 53, 77)
tenor_voice = Voice("tenor", 45, 72)
bass_voice = Voice("bass", 36, 64)

def get_instrument(inst_name):
    if not isinstance(inst_name, str):
        inst_name = str(inst_name)

    # Handle a few scenarios where multiple instruments could be scored
    match inst_name.lower():
        case "bass":
            return bass_voice
        case "b.":
            return bass_voice
        case "tenor":
            return tenor_voice
        case "alto":
            return alto_voice
        case "soprano":
            return soprano_voice
        case "s.":
            return soprano_voice
        case "canto":
            return soprano_voice
        case _:
            raise RuntimeError("Invalid instrument name")


def get_part_range(part):
    notes = part.pitches
    
    # Get midi value of the notes
    midi = list(map(lambda pitch: pitch.midi, notes))
    
    return [min(midi), max(midi)]

