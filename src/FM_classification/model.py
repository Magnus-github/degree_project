import torch
import torch.nn as nn



class STTransformer(nn.Module):
    def __init__(self, test):
        self.test = test
        pass

    def forward(self, x):
        B, T, J, C = x.shape # Batch, num_Frames, Joints, Channels

