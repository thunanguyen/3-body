import glob
import random
import os
import numpy as np

import torch
from torch.utils.data import Dataset

class Data(Dataset):
    def __init__(self, root):
        self.files = sorted(glob.glob(root + "/*.npy"))

    def __getitem__(self, index):
        len_file = len(self.files) // 2
        final = np.load(self.files[index % len_file])
        init = np.load(self.files[len_file + index % len_file])

        return {"init": init, "final": final}

    def __len__(self):
        return len(self.files) // 2
