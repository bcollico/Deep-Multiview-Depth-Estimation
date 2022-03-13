import torch
from config import DEVICE, ACC_THRESH

def loss_fcn(gt:torch.Tensor, initial:torch.Tensor, refined:torch.Tensor):
    """Compute loss from initial and refined depth maps."""

    # ignore invalid points in the depth map
    mask = torch.eq(gt, torch.tensor(0.0).to(DEVICE)).float()
    p_valid = mask.sum((1,2,3))

#     print(p_valid, mask.size())

    # absolute difference between ground truth and estimated depth maps
#     masked_gt    = torch.multiply(torch.abs(gt), mask)
    initial_diff = torch.abs(torch.subtract(gt, initial))
    refined_diff = torch.abs(torch.subtract(gt, refined))

    loss_0 = torch.multiply(mask, initial_diff).sum((1,2,3)).div(p_valid)
    loss_1 = torch.multiply(mask, refined_diff).sum((1,2,3)).div(p_valid)

    # compute mean absolute error for both depth maps
    loss = (loss_0 + loss_1).sum()

    initial_acc = loss_0.mean()
    refined_acc = loss_1.mean()

    return loss, initial_acc, refined_acc
