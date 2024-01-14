import os
from pathlib import Path

import torch
from torch.utils.data import Dataset

from typing import *


class BachChoralesDataset(Dataset):
    def __init__(
        self,
        datadir: Union[str, Path],
        return_I=False,
        lazy=False,
        device=None,
    ):
        if device:
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Folder containing all data to be loaded
        self.data_path = Path(datadir)

        # Select the tracks which will be used in this dset
        track_names = ["X", "Y", "P"]
        if return_I:
            track_names += ["I", "C"]

        # Retrieve all names of files to be loaded, per track
        self._track_filenames = {
            track: [
                self.data_path / track / file
                for file in os.listdir(self.data_path / track)
                if file.endswith(".pt")
            ]
            for track in track_names
        }

        # Check & load all data, if required
        self._track_data = None
        if not lazy:
            self._track_data = {
                track: [
                    torch.load(file, map_location=self.device)
                    for file in self._track_filenames[track]
                ]
                for track in self._track_filenames
            }

    def __len__(self):
        return len(self._track_filenames["X"])

    def __getitem__(self, idx):
        if not self._track_data:
            return tuple(
                [
                    torch.load(
                        self._track_filenames[track][idx], map_location=self.device
                    )
                    for track in self._track_filenames
                ]
            )

        return tuple([self._track_data[track][idx] for track in self._track_data])
