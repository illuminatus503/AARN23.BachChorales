from midi2audio import FluidSynth
from pathlib import Path

from TonicNet.audio import DEFAULT_SOUNDFONT_PATH


class Synthesizer:
    def __init__(self, input_dir, output_dir, synth_path=DEFAULT_SOUNDFONT_PATH):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self._synth = FluidSynth(sound_font=Path(synth_path))

    def synth(self, in_file, filename):
        self._synth.midi_to_audio(self.input_dir / in_file, self.output_dir / filename)
