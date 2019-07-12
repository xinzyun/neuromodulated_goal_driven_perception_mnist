### Define the classifier model for training and test with the MNIST pair datasets.
### This code script was completely written by us, who are anonymous authors of the submitted paper under review.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, sys, copy ; sys.path.append('..')

reseed = lambda: np.random.seed(seed=1) ; ms = torch.manual_seed(1) # for reproducibility
reseed()

if torch.cuda.is_available():
    useCuda = True
else:
    useCuda = False

class MnistClassifier(nn.Module):
    def __init__(self):
        super(MnistClassifier, self).__init__()
        self.fc1 = nn.Linear(28*56, 800) 
        self.fc2 = nn.Linear(800, 600)
        self.fc3 = nn.Linear(600, 400)
        self.fc4 = nn.Linear(600, 400)
        self.fc5 = nn.Linear(400, 20)
        self.fc6 = nn.Linear(400, 4)
        self.fc7 = nn.Linear(400, 20)
        self.fc8 = nn.Linear(400, 4)

    def forward(self, x):
        activations = []
        x = x.view(-1, 28*56)
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        h3 = F.relu(self.fc3(h2))
        h4 = F.relu(self.fc4(h2))
        h5 = self.fc5(h3)
        h6 = self.fc6(h3)
        h7 = self.fc7(h4)
        h8 = self.fc8(h4)
        return [h5, h6, h7, h8]