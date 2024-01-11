import os
import pickle

from pathlib import Path

from random import sample

import numpy as np
import torch

from .instruments import *
from .preprocessing import *

PITCH_TOKENIZER_PATH = Path("TonicNet/audio/tokenisers/pitch_only.p")

MAX_SEQ = 2880
N_PITCH = 48
N_CHORD = 50
N_TOKENS = N_PITCH + N_CHORD

CV_PHASES = ["train", "val"]
TRAIN_ONLY_PHASES = ["train"]

if torch.cuda.is_available():
    PATH = Path("train/training_set/X_cuda")
else:
    PATH = Path("train/training_set/X")

if os.path.exists(PATH):
    TRAIN_BATCHES = len(os.listdir(PATH))
else:
    TRAIN_BATCHES = 0

TOTAL_BATCHES = TRAIN_BATCHES + 76


def get_data_set(mode, shuffle_batches=True, return_I=False):
    match mode:
        case "train":
            parent_dir = "train/training_set"

        case "val":
            parent_dir = "train/val_set"

        case _:
            raise RuntimeError("Invalid model mode. Only TRAIN or VAL are valid")

    if torch.cuda.is_available():
        lst = os.listdir(f"{parent_dir}/X_cuda")
    else:
        lst = os.listdir(f"{parent_dir}/X")

    try:
        lst.remove(".DS_Store")
    except:
        pass

    if shuffle_batches:
        lst = sample(lst, len(lst))

    for file_name in lst:
        if torch.cuda.is_available():
            X = torch.load(f"{parent_dir}/X_cuda/{file_name}")
            Y = torch.load(f"{parent_dir}/Y_cuda/{file_name}")
            P = torch.load(f"{parent_dir}/P_cuda/{file_name}")
            if return_I:
                I = torch.load(f"{parent_dir}/I_cuda/{file_name}")
                C = torch.load(f"{parent_dir}/C_cuda/{file_name}")
        else:
            X = torch.load(f"{parent_dir}/X/{file_name}")
            Y = torch.load(f"{parent_dir}/Y/{file_name}")
            P = torch.load(f"{parent_dir}/P/{file_name}")
            if return_I:
                I = torch.load(f"{parent_dir}/I/{file_name}")
                C = torch.load(f"{parent_dir}/C/{file_name}")

        if return_I:
            yield X, Y, P, I, C
        else:
            yield X, Y, P


def bach_chorales_classic(mode, transpose=False, maj_min=False, jsf_aug=None):
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

    with open(PITCH_TOKENIZER_PATH, "rb") as tokeniser_fp:
        tokeniser = pickle.load(tokeniser_fp)
    tokeniser["end"] = 0
    count = 0

    for folder_name in ["training_set", "val_set"]:
        if torch.cuda.is_available():
            print("cuda:")
            try:
                os.makedirs(f"train/{folder_name}/X_cuda")
                os.makedirs(f"train/{folder_name}/Y_cuda")
                os.makedirs(f"train/{folder_name}/P_cuda")
                os.makedirs(f"train/{folder_name}/I_cuda")
                os.makedirs(f"train/{folder_name}/C_cuda")
            except:
                pass
        else:
            try:
                os.makedirs(f"train/{folder_name}/X")
                os.makedirs(f"train/{folder_name}/Y")
                os.makedirs(f"train/{folder_name}/P")
                os.makedirs(f"train/{folder_name}/I")
                os.makedirs(f"train/{folder_name}/C")
            except:
                pass

    for phase in ["train", "valid"]:
        d = np.load(
            Path("TonicNet/audio/data/Jsb16thSeparated.npz"),
            allow_pickle=True,
            encoding="latin1",
        )
        train = d[phase]

        with open(f"TonicNet/audio/data/{phase}_keysigs.p", "rb") as ks_fp:
            ks = pickle.load(ks_fp)
        
        with open(f"TonicNet/audio/data/{phase}_chords.p", "rb") as cords_fp:
            crds = pickle.load(cords_fp)
            
        with open("TonicNet/audio/data/train_majmin_chords.p", "rb") as cords_fp:
            crds_majmin = pickle.load(cords_fp)
            
        k_count = 0

        if jsf_aug is not None and phase == "train":
            if jsf_aug in ["all", "only"]:
                jsf_path = "TonicNet/audio/data/js-fakes-16thSeparated.npz"
            jsf = np.load(jsf_path, allow_pickle=True, encoding="latin1")
            js_chords = jsf["chords"]
            jsf = jsf["pitches"]

            if jsf_aug == "all":
                train = np.concatenate((train, jsf))
                crds = np.concatenate((crds, js_chords))
            elif jsf_aug == "only":
                train = jsf
                crds = js_chords

        for m in train:
            int_m = m.astype(int)

            if maj_min:
                tonic = ks[k_count][0]
                scale = ks[k_count][1]
                crd_majmin = crds_majmin[k_count]

            crd = crds[k_count]
            k_count += 1

            if transpose is False or phase == "valid":
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
                tokens = []
                try:
                    for x in _tokens:
                        if isinstance(x, str):
                            tokens.append(tokeniser[x])
                        else:
                            tokens.append(x)
                except:
                    print("ERROR: tokenisation")
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

                X = torch.tensor(data_x)
                X = torch.unsqueeze(X, 2)

                Y = torch.tensor(data_y)
                P = torch.tensor(pos_x)
                I = torch.tensor(inst_ids)
                C = torch.tensor(c_class)

                set_folder = "training_set"
                if phase == "valid":
                    set_folder = "val_set"

                if mode == "save":
                    if torch.cuda.is_available():
                        print("cuda:")
                        torch.save(X.cuda(), f"train/{set_folder}/X_cuda/{count}.pt")
                        torch.save(Y.cuda(), f"train/{set_folder}/Y_cuda/{count}.pt")
                        torch.save(P.cuda(), f"train/{set_folder}/P_cuda/{count}.pt")
                        torch.save(I.cuda(), f"train/{set_folder}/I_cuda/{count}.pt")
                        torch.save(C.cuda(), f"train/{set_folder}/C_cuda/{count}.pt")
                    else:
                        torch.save(X, f"train/{set_folder}/X/{count}.pt")
                        torch.save(Y, f"train/{set_folder}/Y/{count}.pt")
                        torch.save(P, f"train/{set_folder}/P/{count}.pt")
                        torch.save(I, f"train/{set_folder}/I/{count}.pt")
                        torch.save(C, f"train/{set_folder}/C/{count}.pt")
                    print("saved", count)
                else:
                    print("processed", count)
                    yield X, Y, P, I, C


def get_test_set_for_eval_classic(phase="test"):
    with open(PITCH_TOKENIZER_PATH, 'rb') as tokeniser_fp:
        tokeniser = pickle.load(tokeniser_fp)
    tokeniser["end"] = 0

    d = np.load("TonicNet/audio/data/Jsb16thSeparated.npz", allow_pickle=True, encoding="latin1")
    test = d[f"{phase}"]

    crds = pickle.load(open(f"TonicNet/audio/data/{phase}_chords.p", "rb"))
    crd_count = 0

    for m in test:
        int_m = m.astype(int)

        crds_piece = crds[crd_count]
        crd_count += 1

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

        for i in int_m:
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

        _tokens.append("end")

        tokens = []
        try:
            for x in _tokens:
                if isinstance(x, str):
                    tokens.append(tokeniser[x])
                else:
                    tokens.append(x)
        except:
            print("ERROR: tokenisation")
            continue

        SEQ_LEN = len(tokens) - 1

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

        X = torch.tensor(data_x)
        X = torch.unsqueeze(X, 2)

        Y = torch.tensor(data_y)
        P = torch.tensor(pos_x)
        C = torch.tensor(c_class)
        I = torch.tensor(inst_ids)

        yield X, Y, P, I, C
