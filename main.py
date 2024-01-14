import os
import argparse
from pathlib import Path

from midi2audio import FluidSynth

from preprocessing.nn_dataset import bach_chorales_classic
from preprocessing.nn_dataset import TRAIN_BATCHES

from train.train_nn import train_TonicNet
from train.train_nn import CrossEntropyTimeDistributedLoss
from train.models import TonicNet

from eval.utils import plot_loss_acc_curves, indices_to_stream, smooth_rhythm
from eval.eval import eval_on_test_set
from eval.sample import sample_TonicNet_random


def main():
    VERSION_STR = """
    TonicNet (Training on Ordered Notation Including Chords)
    Omar Peracha, 2019
    """

    USAGE_STR = """
    --jsf                       prepare dataset with JS Fake Chorales with data augmentation
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

    parser.add_argument("-t", "--train", action="store_true")  # train flag
    parser.add_argument("-m", "--model", type=str, default="")  # model path
    parser.add_argument("--save", action="store_true")  # save model flag
    parser.add_argument("-f", "--factorize", action="store_true")

    parser.add_argument("-lr", "--find_lr", action="store_true")
    parser.add_argument("-st", "--sanity_test", action="store_true")
    parser.add_argument("-s", "--sample", default=1, type=int)
    parser.add_argument("-e", "--eval_nn", action="store_true")

    parser.add_argument("-p", "--plot", action="store_true")

    parser.add_argument("-v", "--version", action="store_true")

    parser.add_argument("--jsf", default=None, choices=["all", "only", "fake", None])

    args = parser.parse_args()

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

    elif args.train:
        # Train, from pretrained or not
        train_TonicNet(
            epochs=3000,
            shuffle_batches=True,
            train_emb_freq=1,
            load_path=Path(args.model) if args.model else "",
            factorize=args.factorize,
            save_model=args.save,
        )

    elif args.plot:
        # Plot loss and accuracy curves
        plot_loss_acc_curves()

    elif args.find_lr:
        # Find optimal learning rate, from pretrained or not
        train_TonicNet(
            epochs=60,
            save_model=True,
            load_path=Path(args.model) if args.model else "",
            shuffle_batches=False,
            num_batches=1,
            val=True,
            train_emb_freq=60,
            lr_range_test=False,
            sanity_test=False,
            factorize=args.factorize,
        )

    elif args.sanity_test:
        # Sanity check, with one batch. Pretrained or not.
        train_TonicNet(
            epochs=60,
            save_model=False,
            load_path=Path(args.model) if args.model else "",
            shuffle_batches=False,
            num_batches=1,
            val=1,
            train_emb_freq=3000,
            lr_range_test=False,
            sanity_test=True,
            factorize=args.factorize,
        )

    elif args.sample > 0:
        os.makedirs(Path("eval/chorales/raw_mid"), exist_ok=True)
        os.makedirs(Path("eval/chorales/audio"), exist_ok=True)

        for n in range(args.sample):
            x = sample_TonicNet_random(
                load_path=Path("eval/TonicNet_epoch-56_loss-0.328_acc-90.750.pt"),
                temperature=1.0,
            )

            stream = indices_to_stream(x)
            smooth_rhythm(
                stream, filename=Path(f"eval/chorales/raw_mid/sample_{n}.mid")
            )

            # Convert to audio
            FluidSynth(
                sound_font=Path("soundfont/UprightPianoKW-20220221.sf2")
            ).midi_to_audio(
                Path(f"eval/chorales/raw_mid/sample_{n}.mid"),
                Path(f"eval/chorales/audio/sample_{n}.wav"),
            )

    elif args.eval_nn:
        # Evaluate a pretrained model
        eval_on_test_set(
            Path("eval/TonicNet_epoch-58_loss-0.317_acc-90.928.pt"),
            TonicNet(nb_tags=98, z_dim=32, nb_layers=3, nb_rnn_units=256, dropout=0.0),
            CrossEntropyTimeDistributedLoss(),
            set="test",
            notes_only=True,
        )

    elif args.version:
        print(VERSION_STR)


if __name__ == "__main__":
    main()
