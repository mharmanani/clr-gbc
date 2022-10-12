import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision.io import read_image

class SimCLR():
    def __init__(self, model_backbone, optim, loss, batch_size):
        self.model = model_backbone
        self.optim = optim
        self.criterion = loss
        self.batch_size = batch_size

    def infoNCELoss(self, X):
        pass
