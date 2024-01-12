from .dataset import BachChoralesDataset

MAX_SEQ = 2880
N_PITCH = 48
N_CHORD = 50
N_TOKENS = 98  # N_PITCH + N_CHORD

INVPITCH_TOKENIZER_PATH = "TonicNet/audio/tokenisers/inverse_pitch_only.p"
PITCH_TOKENIZER_PATH = "TonicNet/audio/tokenisers/pitch_only.p"

DEFAULT_SOUNDFONT_PATH = "TonicNet/audio/soundfont/UprightPianoKW-20220221.sf2"
