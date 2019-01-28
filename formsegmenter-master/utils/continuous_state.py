
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch import nn

import sol
from sol.start_of_line_finder import StartOfLineFinder

from utils import safe_load

import numpy as np
import cv2
import json
import sys
import os
import time
import random

def init_model(config):
    base_0 = config['network']['sol']['base0']
    base_1 = config['network']['sol']['base1']

    sol = StartOfLineFinder(base_0, base_1)
    sol_state = safe_load.torch_state(os.path.join(config['pretraining']['snapshot_path'], "sol.pt"))
    sol.load_state_dict(sol_state)
    sol.cuda()

    return sol
