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

def print_gpu_memory():
    # torch.cuda.empty_cache()
    print("GPU Memory {:.3f} MB / 2048 MB".format(
        torch.cuda.memory_allocated(0)/1024/1024))