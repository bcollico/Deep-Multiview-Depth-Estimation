from config import D_NUM, D_SCALE, N_DEPTH_EST, DEVICE
import torch
import torch.nn.functional as f

def extract_depth_map(prob_volume:torch.Tensor, d_batch:torch.Tensor):
    """Extract the depth via the soft argmax operation. Take the sum of the
    product of each sampled depth with the pixel probabilities at each depth.
    
    Input  Shape: <batch_size, 1, D_NUM, h, w>
    Output Shape: <batch_size, 1, h, w>"""

    # print(prob_volume.max())
    # print(prob_volume.min())

    _, prob_mask = prob_volume.sort(2, descending=True)      # sort along depth direction, get indices
    prob_thresh  = torch.less(prob_mask, N_DEPTH_EST).float() # threshold to max number of depths

    # multiple to remove all probabilities outside the top N_DEPTH_EST
    filtered_prob_volume = torch.multiply(prob_volume, prob_thresh)

    # print("Filtered Prob Volume computed with shape:, ", filtered_prob_volume.size())

    depth_map = (d_batch.unsqueeze(1) * filtered_prob_volume).sum(2).squeeze(2)

    # print(depth_map.max())
    # print(depth_map.min())

    return depth_map
