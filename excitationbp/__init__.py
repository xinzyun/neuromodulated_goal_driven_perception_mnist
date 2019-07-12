### excitation_bp: visualizing how deep networks make decisions
### This code script directly came from the PyTorch implementation of EB and c-EB by Sam Greydanus (https://github.com/greydanus/excitationbp/).
### But most code scripts in our submitted code folder were completely written by us, who are anonymous authors of the submitted paper under review.

from __future__ import absolute_import

from . import functions

from .functions.eb_linear import *

from .utils import *
import copy

__version__ = '0.1'

real_fs = []
real_fs.append(copy.deepcopy(torch.nn.functional.linear))

def use_eb(use_eb, verbose=False):
    global real_torch_funcs
    if use_eb:
        torch.use_pos_weights = True

        if verbose: print("using excitation backprop autograd mode:")

        if verbose: print("\t->replacing torch.nn.functional.linear with eb_linear...")
        torch.nn.functional.linear = EBLinear.apply
        
    else:
        if verbose: print("using regular backprop autograd mode:")

        if verbose: print("\t->restoring torch.nn.backends.thnn.backend.Linear...")
        torch.nn.functional.linear = real_fs[0]