### Make a dataloader for the MNIST dataset (including the digit, parity, and high-low labels along with the pixel values)
### Normalize the pixel values, and add noises
### This code script was completely written by us, who are anonymous authors of the submitted paper under review.

import torch
from torch.autograd import Variable
from torchvision import datasets, transforms, utils
import numpy as np
import os, sys, copy ; sys.path.append('..')

reseed = lambda: np.random.seed(seed=1) ; ms = torch.manual_seed(1) # for reproducibility
reseed()

if torch.cuda.is_available():
    useCuda = True
else:
    useCuda = False

class Dataloader():
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.modes = modes = ['train', 'test']
        trans = transforms.Compose([transforms.ToTensor(),]) 
        dsets = {k: datasets.MNIST('./data', train=k=='train', download=True, transform=trans) for k in modes}
        self.loaders = {k: torch.utils.data.DataLoader(dsets[k], batch_size=batch_size, shuffle=True) for k in modes}

    def next(self, mode='train',sigma=0.7):
        orig_X, orig_y = next(iter(self.loaders[mode]))
        X = Variable(orig_X + sigma*torch.rand(orig_X.shape)).view(self.batch_size, -1)
        y = Variable(orig_y) # y is an integer between 0 and 9 inclusively.
        p = Variable(orig_y%2 == 1) # p is 0 if even, 1 if odd
        hl = Variable(orig_y > 4) # hl is 0 if low (0~4), 1 if high (5~9)
        if useCuda:
            X = X.type(torch.cuda.FloatTensor)
            y = y.type(torch.cuda.LongTensor)
            p = p.type(torch.cuda.LongTensor)
            hl = hl.type(torch.cuda.LongTensor)
        else:
            X = X.type(torch.FloatTensor)
            y = y.type(torch.LongTensor)
            p = p.type(torch.LongTensor)
            hl = hl.type(torch.LongTensor)
        return X, y, p, hl
