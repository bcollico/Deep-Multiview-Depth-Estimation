import torch
from config import DEVICE, ACC_THRESH, FEAT_H, FEAT_W

def loss_fcn(gt:torch.Tensor, initial:torch.Tensor, refined:torch.Tensor):
    """Compute loss from initial and refined depth maps."""

    # ignore invalid points in the depth map
    mask = torch.ne(gt, torch.tensor(0.0).to(DEVICE)).float()
    # p_valid = torch.neg(mask - torch.tensor(1.0).to(DEVICE)).sum((1,2,3))
    p_valid = mask.sum((1,2,3))

    # absolute difference between ground truth and estimated depth maps
#     masked_gt    = torch.multiply(torch.abs(gt), mask)

    # print(gt.min(), gt.max())
    # print(initial.min(), initial.max())
    # print(refined.min(), refined.max())   

    initial_diff = torch.abs(torch.subtract(gt, initial))
    refined_diff = torch.abs(torch.subtract(gt, refined))

    # print(initial_diff.min(), initial_diff.max())
    # print(refined_diff.min(), refined_diff.max())

    loss_0 = torch.multiply(mask, initial_diff).sum((1,2,3)).div(p_valid)
    loss_1 = torch.multiply(mask, refined_diff).sum((1,2,3)).div(p_valid)

    # print(torch.multiply(mask, initial_diff).min(), torch.multiply(mask, initial_diff).max())
    # print(torch.multiply(mask, refined_diff).min(), torch.multiply(mask, refined_diff).max())

    # print(loss_0, loss_1)
    # print(loss_0*p_valid, loss_1*p_valid)
    # print(torch.multiply(mask, initial_diff).mean(), torch.multiply(mask, refined_diff).mean())

    # compute mean absolute error for both depth maps
    loss = (loss_0 + loss_1).sum()

    initial_acc = loss_0.mean()
    refined_acc = loss_1.mean()

    return loss, initial_acc, refined_acc
