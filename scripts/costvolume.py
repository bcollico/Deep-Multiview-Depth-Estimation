import torch

def assemble_cost_volume(warped_feature_maps, n_views):
    """Assemble the cost volume using a variance-based cost metric. There is
    ambiguity in the MVS paper whether the mean is obtained over all 32 
    feature volume channels or if it is per channel. The implementation here is 
    per channel."""
    b, c, w, h = warped_feature_maps.size()
    ref_idx    = np.arange(0,b*n_views,n_views)

    N = w * h * b # number of pixels for averaging

    print(N)

    mean = torch.sum(warped_feature_maps, (0, 2, 3)) / (N)

    print(torch.reshape(mean, (1, c, 1, 1)).size())
    print(warped_feature_maps.size())

    diff = warped_feature_maps - torch.reshape(mean, (1, c, 1, 1))

    var = torch.sum(diff**2, [2,3]) / N

    return var

if __name__ == "__main__":

    input = torch.randn(9,32,1000,1000)
    n_views = 3

    var = assemble_cost_volume(input, n_views)

    print(var)
