import torch
from torch import Tensor

def count_non_nans(
    tracks: Tensor,
    gt_trajectories: Tensor
) -> int:
    """
    Args:
        tracks: Tensor of track positions with shape (M, N, 3)
        gt_trajectories: Tensor of ground truth positions with shape (M, N, 3)

    Returns:
        Total number of valid (track point, keypoint) pairs.
    """
    valid_mask = ~(torch.isnan(tracks).any(dim=-1) | torch.isnan(gt_trajectories).any(dim=-1))
    return int(torch.sum(valid_mask).item())

def measure_track_L2_loss(
    tracks: Tensor,
    gt_trajectories: Tensor
) -> float:
    """
    Args:
        tracks: Tensor of track positions with shape (M, N, 3)
        gt_trajectories: Tensor of ground truth positions with shape (M, N, 3)

    Returns:
        Total L2 loss between track and ground truth trajectory.
    """
    assert tracks.shape == gt_trajectories.shape, \
        "Tracks and ground truth trajectories must have the same shape"
    assert len(tracks.shape) == 3, \
        "Tracks must have shape (M, N, 3)."

    squared_diffs = torch.nansum((tracks - gt_trajectories) ** 2, dim=2)
    point_distances = torch.sqrt(squared_diffs)
    cum_sum = torch.sum(point_distances)
    valid_cnt = count_non_nans(tracks, gt_trajectories)
    l2_loss = float(cum_sum.item() / valid_cnt)

    return float(l2_loss)

def measure_track_L2_variance(
    tracks: Tensor,
    gt_trajectories: Tensor
) -> float:
    """
    Args:
        track: Tensor of track positions with shape (M, N, 3)
        gt_trajectory: Tensor of ground truth positions with shape (M, N, 3)

    Returns:
        Total Variance between track and ground truth trajectory.
    """
    assert tracks.shape == gt_trajectories.shape, \
        "Tracks and ground truth trajectories must have the same shape"
    assert len(tracks.shape) == 3, \
        "Tracks must have shape (M, N, 3)."

    # Ref: var = mean(abs(x - x.mean())**2)
    mean_loss = measure_track_L2_loss(tracks, gt_trajectories)
    squared_diffs = torch.nansum((tracks - gt_trajectories) ** 2, dim=2)
    per_point_loss = torch.sqrt(squared_diffs)
    variance = torch.mean(torch.abs(per_point_loss - mean_loss)**2)

    return float(variance.item())