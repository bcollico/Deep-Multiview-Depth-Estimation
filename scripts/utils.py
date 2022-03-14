import torch
import numpy as np
import psutil

def append_zero(input):
    input = np.append(input, 0)

def print_size(input):
    print(input.size())

def unsqueeze_n(input, n, dim=0):
    for _ in range(n):
        input = torch.unsqueeze(input, dim=0)
    return input

def print_gpu_memory(label=''):
    # torch.cuda.empty_cache()
    print(label+"RAM: {:.1f}mb/{:.1f}mb \t GPU: {:.1f}mb/10500mb".format(
        psutil.virtual_memory()[3] >> 20,
        psutil.virtual_memory()[0] >> 20,
        torch.cuda.memory_allocated(0)/1024/1024))