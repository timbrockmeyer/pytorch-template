import torch

def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert len(pred) == len(target)
        correct = torch.sum(pred == target).item()
    return correct / len(target)