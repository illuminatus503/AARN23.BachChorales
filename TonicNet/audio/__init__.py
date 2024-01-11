from .load_dataset import (
    MAX_SEQ,
    N_PITCH,
    N_CHORD,
    N_TOKENS,
    TRAIN_BATCHES,
    TOTAL_BATCHES,
    CV_PHASES,
    TRAIN_ONLY_PHASES,
)

from pathlib import Path

INVPITCH_TOKENIZER_PATH = Path("TonicNet/audio/tokenisers/inverse_pitch_only.p")
PITCH_TOKENIZER_PATH = Path("TonicNet/audio/tokenisers/pitch_only.p")

DEFAULT_SOUNDFONT_PATH = Path("TonicNet/audio/soundfont/UprightPianoKW-20220221.sf2")
