import torch
import numpy as np

def compute_accuracy():
    raise NotImplementedError

def append_zero(input):
    input = np.append(input, 0)

def print_size(input):
    print(input.size())

def unsqueeze_n(input, n, dim=0):
    for _ in range(n):
        input = torch.unsqueeze(input, dim=0)
    return input
