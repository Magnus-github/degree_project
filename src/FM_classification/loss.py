import torch
import torch.nn as nn


def WeightedCrossEntropyLoss(weights, device):
    weights = torch.tensor(weights)*len(weights)
    weights = weights.to(device)
    return nn.CrossEntropyLoss(weight=weights)

def CrossEntropyLoss(weights=None, device=None):
    return nn.CrossEntropyLoss()


if __name__ == '__main__':
    weights = [1.0, 1.0, 1.0]
    device = 'cpu'
    criterion = WeightedCrossEntropyLoss(weights, device)
    input = torch.randn(5, 3, requires_grad=True)
    target = torch.empty(5, dtype=torch.long).random_(3)
    output = criterion(input, target)
    print('Test passed')