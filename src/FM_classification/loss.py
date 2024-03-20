import torch
import torch.nn as nn


def WeightedCrossEntropyLoss(weights, device):
    weights = torch.tensor(weights)*len(weights)
    weights = weights.to(device)
    return nn.CrossEntropyLoss(weight=weights)

def CrossEntropyLoss():
    return nn.CrossEntropyLoss()
