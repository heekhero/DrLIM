from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys
from torchvision import transforms
from torchvision.datasets import MNIST
import torch
from matplotlib import pyplot as plt

class MNIST_clip(MNIST):
    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(MNIST_clip, self).__init__(root, train, transform, target_transform, download)
        self._filter()

    # keep 4 and 9
    def _filter(self):
        four_idx = torch.where(self.targets == 4)[0]
        nine_idx = torch.where(self.targets == 9)[0]

        cat_idx = torch.cat([four_idx, nine_idx], dim=0)
        self.data = self.data[cat_idx]
        self.targets[four_idx] = 0
        self.targets[nine_idx] = 1
        self.targets = self.targets[cat_idx]



