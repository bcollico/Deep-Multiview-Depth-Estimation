import torch
from config import DEVICE, ACC_THRESH

def loss_fcn(gt:torch.Tensor, initial:torch.Tensor, refined:torch.Tensor):
    """Compute loss from initial and refined depth maps."""

    # absolute difference between ground truth and estimated depth maps
    initial_diff = torch.abs(torch.subtract(gt, initial))
    refined_diff = torch.abs(torch.subtract(gt, refined))

    # ignore invalid points in the depth map
    mask = torch.eq(gt, torch.tensor(0.0).to(DEVICE)).float()
    p_valid = mask.sum((1,2,3))

    # compute mean absolute error for both depth maps
    loss = torch.multiply(mask, initial_diff).sum((1,2,3)).div(p_valid) + \
           torch.multiply(mask, refined_diff).sum((1,2,3)).div(p_valid)

    # percentage of valid point estimates within ACC_THRESH percent of true depth
    initial_acc = initial_diff.div(gt).le(ACC_THRESH).multiply(mask).sum((1,2,3)).div(p_valid)
    refined_acc = refined_diff.div(gt).le(ACC_THRESH).multiply(mask).sum((1,2,3)).div(p_valid)

    return loss, initial_acc, refined_acc
