import torch

def assemble_cost_volume(warped_feature_maps:torch.Tensor, n_views:int):
    """Assemble the cost volume using a variance-based cost metric. There is
    ambiguity in the MVS paper about the mean feature volume, in this case
    it is taken as a sum in the n_view dimension divided by n_views."""
    bn, c, d, h, w = warped_feature_maps.size()

    # separate the views of each batch: new shape <b, n_views, c, d, h, w>
    warped_feature_maps = warped_feature_maps.reshape((int(bn/n_views), n_views, c, d, h, w))

    mean_feature_map = (warped_feature_maps.sum(1) / n_views).unsqueeze(1)

    cost_volume = (warped_feature_maps - mean_feature_map).pow(2).sum(1) / n_views

    return cost_volume

if __name__ == "__main__":

    """Test cost volume computation"""

    input = torch.randn(9, 32, 5, 128, 160).to("cuda:0")
    n_views = 3

    cost_volume = assemble_cost_volume(input, n_views)

    # print(var)
