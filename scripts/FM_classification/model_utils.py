import torch.nn as nn
import torch

def AvgPool(kernel_size: int):
    return nn.AvgPool1d(kernel_size)

def MaxPool(kernel_size: int):
    return nn.MaxPool1d(kernel_size)

def Voting():
    def __init__(self, dummy):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def __call__(self, x):
        x = x.to('cpu')
        x = torch.mode(x, dim=-1).values
        return x.to(self.device)
    