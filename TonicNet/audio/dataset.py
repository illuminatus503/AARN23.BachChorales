import os
from pathlib import Path

import torch
from torch.utils.data import Dataset

from typing import *


class BachChoralesDataset(Dataset):
    def __init__(self, datadir: Union[str, Path], return_I=False, device=None):
        self.data_path = Path(datadir)
        self.return_I = return_I

        if device:
            self.device = device
        else:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")

        # Load all files for X, Y and P tracks
        self._Xdir = self.data_path / "X"
        self._Ydir = self.data_path / "Y"
        self._Pdir = self.data_path / "P"

        self.X_filenames = [
            file for file in os.listdir(self._Xdir) if file.endswith(".pt")
        ]
        self.Y_filenames = [
            file for file in os.listdir(self._Ydir) if file.endswith(".pt")
        ]
        self.P_filenames = [
            file for file in os.listdir(self._Pdir) if file.endswith(".pt")
        ]

        if not (
            len(self.X_filenames) == len(self.Y_filenames) == len(self.P_filenames)
        ):
            raise ValueError("Invalid data length for X, Y and P tracks")

        # If other tracks are needed,
        if return_I:
            self._Idir = self.data_path / "I"
            self._Cdir = self.data_path / "C"

            self.I_filenames = [
                file for file in os.listdir(self._Idir) if file.endswith(".pt")
            ]
            self.C_filenames = [
                file for file in os.listdir(self._Cdir) if file.endswith(".pt")
            ]

            if not (
                len(self.X_filenames) == len(self.I_filenames) == len(self.C_filenames)
            ):
                raise ValueError("Invalid data length for I and C tracks")

    def __len__(self):
        return len(self.X_filenames)

    def __getitem__(self, idx):
        X_file = self._Xdir / self.X_filenames[idx]
        Y_file = self._Ydir / self.Y_filenames[idx]
        P_file = self._Pdir / self.P_filenames[idx]

        X = torch.load(X_file, map_location=self.device)
        Y = torch.load(Y_file, map_location=self.device)
        P = torch.load(P_file, map_location=self.device)

        if self.return_I:
            I_file = self._Idir / self.I_filenames[idx]
            C_file = self._Cdir / self.C_filenames[idx]

            I = torch.load(I_file, map_location=self.device)
            C = torch.load(C_file, map_location=self.device)

            return X, Y, P, I, C

        return X, Y, P
