import os
import argparse
from pathlib import Path

from midi2audio import FluidSynth

from TonicNet.audio.load_dataset import bach_chorales_classic
from TonicNet.audio.utils import indices_to_stream, smooth_rhythm
from TonicNet.audio import DEFAULT_SOUNDFONT_PATH

from TonicNet.models.external import CrossEntropyTimeDistributedLoss
from TonicNet.models.train import (
    train_TonicNet,
    TonicNet_lr_finder,
    TonicNet_sanity_test,
)
from TonicNet.models.eval import eval_on_test_set, sample_TonicNet_random
from TonicNet.models import TonicNet

DEFAULT_LOAD_MODEL = Path(
    "TonicNet/models/pretrained/TonicNet_epoch-58_loss-0.317_acc-90.928.pt"
)

DEFAULT_OUTPUT_PATH = Path("samples")
DEFAULT_OUTPUT_PATH_MID = Path("samples/raw_mid")


def main():
    VERSION_STR = """
    TonicNet (Training on Ordered Notation Including Chords)
    Omar Peracha, 2019
    """

    USAGE_STR = """
    --jsf all                   prepare dataset with JS Fake Chorales with data augmentation
    --jsf only                  prepare dataset with JS Fake Chorales only
    --train [-t]                train model from scratch
    --eval_nn [-e]              evaluate pretrained model on test set
    --sample [-s]               sample from pretrained model
    --sanity_check [-scheck]    check if the model is correctly structured
    """

    parser = argparse.ArgumentParser(
        prog=VERSION_STR,
        usage=USAGE_STR,
    )

    parser.add_argument("-t", "--train", action="store_true")
    parser.add_argument("-lr", "--find_lr", action="store_true")
    parser.add_argument("-scheck", "--sanity_check", action="store_true")
    parser.add_argument("-s", "--sample", type=int)
    parser.add_argument("-e", "--eval_nn", action="store_true")
    parser.add_argument("-v", "--version", action="store_true")
    parser.add_argument("-m", "--model", type=str, default=DEFAULT_LOAD_MODEL)

    parser.add_argument("--jsf", default=None, choices=["all", "only", "fake", None])

    args = parser.parse_args()
    LOADED_MODEL_PATH = Path(args.model)

    if args.version:
        print(VERSION_STR)
        exit(0)

    # Parse the dataset generation
    if args.jsf is not None:
        match args.jsf:
            case "only":
                for x, y, p, i, c in bach_chorales_classic(
                    "save", transpose=True, jsf_aug="only"
                ):
                    continue
            case "fake":
                for x, y, p, i, c in bach_chorales_classic("save", transpose=True):
                    continue
            case "all":
                for x, y, p, i, c in bach_chorales_classic(
                    "save", transpose=True, jsf_aug="all"
                ):
                    continue

    if args.train:
        # TODO: add filename
        train_TonicNet(3000, shuffle_batches=1, train_emb_freq=1, load_path="")
    elif args.find_lr:
        TonicNet_lr_finder(train_emb_freq=1, load_path="")
    elif args.sanity_check:
        TonicNet_sanity_test(num_batches=1, train_emb_freq=1)
    elif args.sample > 0:
        os.makedirs(DEFAULT_OUTPUT_PATH, exist_ok=True)
        os.makedirs(DEFAULT_OUTPUT_PATH_MID, exist_ok=True)

        for n in range(args.sample):
            x = sample_TonicNet_random(load_path=LOADED_MODEL_PATH, temperature=1.0)

            stream = indices_to_stream(x)
            smooth_rhythm(
                stream,
                filename=DEFAULT_OUTPUT_PATH_MID / f"sample_{n}.mid",
            )

            # Convert to audio
            FluidSynth(sound_font=DEFAULT_SOUNDFONT_PATH).midi_to_audio(
                DEFAULT_OUTPUT_PATH_MID / f"sample_{n}.mid",
                DEFAULT_OUTPUT_PATH / f"sample_{n}.wav",
            )

    elif args.eval_nn:
        eval_on_test_set(
            DEFAULT_LOAD_MODEL,
            TonicNet(nb_tags=98, z_dim=32, nb_layers=3, nb_rnn_units=256, dropout=0.0),
            CrossEntropyTimeDistributedLoss(),
            set="test",
            notes_only=True,
        )

    else:
        print("[E] Invalid argument. See help with -h.")
        exit(0)


if __name__ == "__main__":
    main()
