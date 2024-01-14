import os
import argparse
from pathlib import Path

from TonicNet.audio.preprocessing import bach_chorales_classic
from TonicNet.audio.utils import indices_to_stream, smooth_rhythm
from TonicNet.audio.synthesizer import Synthesizer
from TonicNet.audio import N_TOKENS

from TonicNet.models import TonicNet
from TonicNet.models.utils import print_model_summary

from TonicNet.models.train import (
    train_TonicNet,
    TonicNet_lr_finder,
    TonicNet_sanity_test,
    train_Transformer,
    Transformer_lr_finder,
    Transformer_sanity_test,
)
from TonicNet.models.eval import sample_TonicNet_random

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
    --summary [-sum]            make a summary of the model
    """

    parser = argparse.ArgumentParser(
        prog=VERSION_STR,
        usage=USAGE_STR,
    )

    parser.add_argument("-t", "--train", action="store_true")
    parser.add_argument("-lr", "--find_lr", action="store_true")
    parser.add_argument("-scheck", "--sanity_check", action="store_true")
    parser.add_argument("-s", "--sample", type=int, default=0)
    parser.add_argument("-e", "--eval_nn", action="store_true")
    parser.add_argument("-v", "--version", action="store_true")
    parser.add_argument("-m", "--model", type=str, default=DEFAULT_LOAD_MODEL)
    parser.add_argument("-sum", "--summary", action="store_true")

    parser.add_argument("--jsf", default=None, choices=["all", "only", "fake", None])
    parser.add_argument(
        "--arch", type=str, choices=["tonicnet", "transformer"], default="tonicnet"
    )

    args = parser.parse_args()
    LOADED_MODEL_PATH = Path(args.model)

    if args.version:
        print(VERSION_STR)

    elif args.jsf is not None:
        match args.jsf:
            case "only":
                bach_chorales_classic(transpose=True, jsf_aug="only")
            case "fake":
                bach_chorales_classic(transpose=True)
            case "all":
                bach_chorales_classic(transpose=True, jsf_aug="all")

        print("Dataset preprocessed successfully.")

    elif args.summary:
        match args.arch:
            case "tonicnet":
                model = TonicNet(
                    nb_tags=N_TOKENS,
                    z_dim=32,
                    nb_layers=3,
                    nb_rnn_units=256,
                    dropout=0.3,
                )

            case "transformer":
                raise NotImplementedError

            case _:
                raise RuntimeError("NotReachable")
        
        print_model_summary(model)

    elif args.train:
        match args.arch:
            case "tonicnet":
                train_TonicNet(
                    traindir=Path("preprocessed_data/training_set"),
                    valdir=Path("preprocessed_data/validation_set"),
                )
                # train_TonicNet(3000, shuffle_batches=1, train_emb_freq=1, load_path="")
            case "transformer":
                # train_Transformer(3000, shuffle_batches=1, load_path="")
                print(NotImplemented)
            case _:
                raise RuntimeError("NotReachable")

    elif args.find_lr:
        match args.arch:
            case "tonicnet":
                TonicNet_lr_finder(train_emb_freq=1, load_path="")
            case "transformer":
                Transformer_lr_finder(load_path="")
            case _:
                raise RuntimeError("NotReachable")

    elif args.sanity_check:
        match args.arch:
            case "tonicnet":
                TonicNet_sanity_test(num_batches=1, train_emb_freq=1)
            case "transformer":
                Transformer_sanity_test(num_batches=1)
            case _:
                raise RuntimeError("NotReachable")

    elif args.sample > 0:
        synth = Synthesizer(DEFAULT_OUTPUT_PATH_MID, DEFAULT_OUTPUT_PATH)

        match args.arch:
            case "tonicnet":
                os.makedirs(DEFAULT_OUTPUT_PATH_MID / "tonicnet", exist_ok=True)
                os.makedirs(DEFAULT_OUTPUT_PATH / "tonicnet", exist_ok=True)

                for n in range(args.sample):
                    print(LOADED_MODEL_PATH)
                    x = sample_TonicNet_random(
                        load_path=LOADED_MODEL_PATH, temperature=1.0
                    )

                    stream = indices_to_stream(x)
                    smooth_rhythm(
                        stream,
                        filename=DEFAULT_OUTPUT_PATH_MID / f"tonicnet/sample_{n}.mid",
                    )

                    # Convert to audio
                    synth.synth(f"tonicnet/sample_{n}.mid", f"tonicnet/sample_{n}.wav")

            case "transformer":
                os.makedirs(DEFAULT_OUTPUT_PATH_MID / "transformer", exist_ok=True)
                os.makedirs(DEFAULT_OUTPUT_PATH / "transformer", exist_ok=True)

                for n in range(args.sample):
                    x = sample_TonicNet_random(
                        load_path=LOADED_MODEL_PATH, temperature=1.0
                    )

                    stream = indices_to_stream(x)
                    smooth_rhythm(
                        stream,
                        filename=DEFAULT_OUTPUT_PATH_MID
                        / f"transformer/sample_{n}.mid",
                    )

                    # Convert to audio
                    synth.synth(
                        f"transformer/sample_{n}.mid", f"transformer/sample_{n}.wav"
                    )

    elif args.eval_nn:
        raise NotImplementedError
        # eval_on_test_set(
        #     DEFAULT_LOAD_MODEL,
        #     TonicNet(nb_tags=98, z_dim=32, nb_layers=3, nb_rnn_units=256, dropout=0.0),
        #     CrossEntropyTimeDistributedLoss(),
        #     set="test",
        #     notes_only=True,
        # )

    else:
        print("[E] Invalid argument. See help with -h.")
        exit(0)


if __name__ == "__main__":
    main()
