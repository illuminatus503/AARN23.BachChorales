import os
import pickle

from pathlib import Path

import numpy as np
import torch

from tqdm import tqdm

from .utils.instruments import *
from .utils.preprocessing import *

from . import PITCH_TOKENIZER_PATH


def bach_chorales_classic(
    transpose=False,
    maj_min=False,
    jsf_aug=None,
    dir=Path("preprocessed_data"),
):
    if maj_min and jsf_aug is not None:
        raise ValueError("maj_min and jsf_aug can not both be true")

    if jsf_aug not in [
        "all",
        "only",
        "topk-all",
        "topk-only",
        "topk-skilled-all",
        "topk-skilled-only",
        None,
    ]:
        raise ValueError(
            "unrecognised value for jsf_aug parameter: can only be 'all', 'only', "
            "'topl-all', 'topk-only', 'topk-skilled-all', 'topk-skilled-only' or None"
        )

    # Load tokenizer
    with open(PITCH_TOKENIZER_PATH, "rb") as tokeniser_fp:
        tokeniser = pickle.load(tokeniser_fp)
    tokeniser["end"] = 0

    # Load raw datasets
    d = np.load(
        Path("TonicNet/audio/data/Jsb16thSeparated.npz"),
        allow_pickle=True,
        encoding="latin1",
    )

    if jsf_aug:
        jsf = np.load(
            Path("TonicNet/audio/data/js-fakes-16thSeparated.npz"),
            allow_pickle=True,
            encoding="latin1",
        )

    count = 0
    for folder_name in ["training_set", "validation_set", "test_set"]:
        os.makedirs(dir / f"{folder_name}/X", exist_ok=True)
        os.makedirs(dir / f"{folder_name}/Y", exist_ok=True)
        os.makedirs(dir / f"{folder_name}/P", exist_ok=True)
        os.makedirs(dir / f"{folder_name}/I", exist_ok=True)
        os.makedirs(dir / f"{folder_name}/C", exist_ok=True)

    for phase, folder_name in tqdm(
        zip(["train", "valid", "test"], ["training_set", "validation_set", "test_set"])
    ):
        k_count = 0
        train = d[phase]

        with open(f"TonicNet/audio/data/{phase}_keysigs.p", "rb") as keysigns_fp:
            ks = pickle.load(keysigns_fp)

        with open(f"TonicNet/audio/data/{phase}_chords.p", "rb") as chords_fp:
            crds = pickle.load(chords_fp)

        if phase == "train":
            with open(
                "TonicNet/audio/data/train_majmin_chords.p", "rb"
            ) as maj_min_chords_fp:
                crds_majmin = pickle.load(maj_min_chords_fp)

            if jsf_aug == "all":
                train = np.concatenate((train, jsf["pitches"]))
                crds = np.concatenate((crds, jsf["chords"]))
            elif jsf_aug == "only":
                train = jsf["pitches"]
                crds = jsf["chords"]

        for m in train:
            int_m = m.astype(int)

            if maj_min:
                tonic = ks[k_count][0]
                scale = ks[k_count][1]
                crd_majmin = crds_majmin[k_count]

            crd = crds[k_count]
            k_count += 1

            if not transpose or phase == "valid":
                transpositions = [int_m]
                crds_pieces = [crd]
            else:
                parts = [int_m[:, 0], int_m[:, 1], int_m[:, 2], int_m[:, 3]]
                transpositions, tonics, crds_pieces = np_perform_all_transpositions(
                    parts, 0, crd
                )

                if maj_min:
                    mode_switch = np_convert_major_minor(int_m, tonic, scale)
                    ms_parts = [
                        mode_switch[:, 0],
                        mode_switch[:, 1],
                        mode_switch[:, 2],
                        mode_switch[:, 3],
                    ]
                    ms_trans, ms_tons, ms_crds = np_perform_all_transpositions(
                        ms_parts, tonic, crd_majmin
                    )

                    transpositions += ms_trans
                    tonics += ms_tons
                    crds_pieces += ms_crds

            kc = 0

            for t in transpositions:
                crds_piece = crds_pieces[kc]

                _tokens = []
                inst_ids = []
                c_class = []

                current_s = ""
                s_count = 0

                current_a = ""
                a_count = 0

                current_t = ""
                t_count = 0

                current_b = ""
                b_count = 0

                current_c = ""
                c_count = 0

                timestep = 0

                for i in t:
                    s = "Rest" if i[0] < 36 else str(i[0])
                    b = "Rest" if i[3] < 36 else str(i[3])
                    a = "Rest" if i[1] < 36 else str(i[1])
                    t = "Rest" if i[2] < 36 else str(i[2])

                    c_val = crds_piece[timestep] + 48
                    timestep += 1

                    _tokens = _tokens + [c_val, s, b, a, t]
                    c_class = c_class + [c_val]

                    if c_val == current_c:
                        c_count += 1
                    else:
                        c_count = 0
                        current_c = c_val

                    if s == current_s:
                        s_count += 1
                    else:
                        s_count = 0
                        current_s = s

                    if b == current_b:
                        b_count += 1
                    else:
                        b_count = 0
                        current_b = b

                    if a == current_a:
                        a_count += 1
                    else:
                        a_count = 0
                        current_a = a

                    if t == current_t:
                        t_count += 1
                    else:
                        t_count = 0
                        current_t = t

                    inst_ids = inst_ids + [c_count, s_count, b_count, a_count, t_count]

                pos_ids = list(range(len(_tokens)))

                kc += 1
                _tokens.append("end")

                # TOKENIZATION !!
                tokens = []
                for x in _tokens:
                    try:
                        if isinstance(x, str):
                            tokens.append(tokeniser[x])
                        else:
                            tokens.append(x)
                    except:
                        continue
                SEQ_LEN = len(tokens) - 1

                count += 1

                data_x = []
                data_y = []

                pos_x = []

                for i in range(0, len(tokens) - SEQ_LEN, 1):
                    t_seq_in = tokens[i : i + SEQ_LEN]
                    t_seq_out = tokens[i + 1 : i + 1 + SEQ_LEN]
                    data_x.append(t_seq_in)
                    data_y.append(t_seq_out)

                    p_seq_in = pos_ids[i : i + SEQ_LEN]
                    pos_x.append(p_seq_in)

                # Generate data and save it for later.
                X = torch.tensor(data_x)
                X = torch.unsqueeze(X, 2)

                Y = torch.tensor(data_y)
                P = torch.tensor(pos_x)
                I = torch.tensor(inst_ids)
                C = torch.tensor(c_class)

                torch.save(X, dir / f"{folder_name}/X/{count}.pt")
                torch.save(Y, dir / f"{folder_name}/Y/{count}.pt")
                torch.save(P, dir / f"{folder_name}/P/{count}.pt")
                torch.save(I, dir / f"{folder_name}/I/{count}.pt")
                torch.save(C, dir / f"{folder_name}/C/{count}.pt")
