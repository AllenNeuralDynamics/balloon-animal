import numpy as np
from numpy.typing import NDArray

def count_non_nans(
    tracks: NDArray[np.float64],
    gt_trajectories: NDArray[np.float64]
) -> int:
    """
    Args:
        tracks: Array of track positions with shape (M, N, 3)
        gt_trajectories: Array of ground truth positions with shape (M, N, 3)
    Returns:
        Total number of valid (track point, keypoint) pairs.
    """
    valid_mask = ~(np.isnan(tracks).any(axis=-1) | np.isnan(gt_trajectories).any(axis=-1))
    return int(np.sum(valid_mask))

def measure_track_L2_loss(
    tracks: NDArray[np.float64],
    gt_trajectories: NDArray[np.float64]
) -> float:
    """
    Args:
        tracks: Array of track positions with shape (M, N, 3)
        gt_trajectories: Array of ground truth positions with shape (M, N, 3)
    Returns:
        Total L2 loss between track and ground truth trajectory.
    """
    assert tracks.shape == gt_trajectories.shape, \
        "Tracks and ground truth trajectories must have the same shape"
    assert len(tracks.shape) == 3, \
        "Tracks must have shape (M, N, 3)."

    squared_diffs = np.nansum((tracks - gt_trajectories) ** 2, axis=2)
    point_distances = np.sqrt(squared_diffs)
    cum_sum = np.sum(point_distances)
    valid_cnt = count_non_nans(tracks, gt_trajectories)
    l2_loss = float(cum_sum / valid_cnt)

    return float(l2_loss)

def measure_track_L2_variance(
    tracks: NDArray[np.float64],
    gt_trajectories: NDArray[np.float64]
) -> float:
    """
    Args:
        track: Array of track positions with shape (M, N, 3)
        gt_trajectory: Array of ground truth positions with shape (M, N, 3)
    Returns:
        Total Variance between track and ground truth trajectory.
    """
    assert tracks.shape == gt_trajectories.shape, \
        "Tracks and ground truth trajectories must have the same shape"
    assert len(tracks.shape) == 3, \
        "Tracks must have shape (M, N, 3)."

    # Ref: var = mean(abs(x - x.mean())**2)
    mean_loss = measure_track_L2_loss(tracks, gt_trajectories)
    squared_diffs = np.nansum((tracks - gt_trajectories) ** 2, axis=2)
    per_point_loss = np.sqrt(squared_diffs)

    variance = np.mean(np.abs(per_point_loss - mean_loss)**2)
    return float(variance)
