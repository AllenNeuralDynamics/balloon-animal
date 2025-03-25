"""
Variety of utilities.
"""
from collections import defaultdict
from typing import TypeVar, Dict
import numpy as np
from numpy.typing import NDArray

T = TypeVar('T')
N = TypeVar('N')

def quaternions_to_rotation_matrices(
    q: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Convert batch of unnormalized quaternions [r, x, y, z]
    into batch of rotation matrices.

    Args:
        q: Array of quaternions with shape (N, 4)

    Returns:
        Array of rotation matrices with shape (N, 3, 3)
    """
    # Validate input shape
    if q.ndim != 2 or q.shape[1] != 4:
        raise ValueError(f"Expected quaternions with shape (N, 4), got {q.shape}")

    # Normalize quaternions
    norm = np.sqrt(q[:, 0] * q[:, 0] + q[:, 1] * q[:, 1] + q[:, 2] * q[:, 2] + q[:, 3] * q[:, 3])
    q = q / norm[:, None]

    # Initialize rotation matrix array
    rot = np.zeros((q.shape[0], 3, 3))

    # Extract quaternion components
    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    # Fill rotation matrix
    rot[:, 0, 0] = 1 - 2 * (y * y + z * z)
    rot[:, 0, 1] = 2 * (x * y - r * z)
    rot[:, 0, 2] = 2 * (x * z + r * y)
    rot[:, 1, 0] = 2 * (x * y + r * z)
    rot[:, 1, 1] = 1 - 2 * (x * x + z * z)
    rot[:, 1, 2] = 2 * (y * z - r * x)
    rot[:, 2, 0] = 2 * (x * z - r * y)
    rot[:, 2, 1] = 2 * (y * z + r * x)
    rot[:, 2, 2] = 1 - 2 * (x * x + y * y)

    return rot

def build_homogeneous_matrices(
    means: NDArray[np.float64],
    semi_axes: NDArray[np.float64],
    quaternions: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Convert gaussian means, semi_axes, and quaternions into basis frames.

    Args:
        means: Array of mean positions with shape (N, 3)
        semi_axes: Array of semi-axes with shape (N, 3)
        quaternions: Array of quaternions with shape (N, 4)

    Returns:
        Array of homogeneous transformation matrices with shape (N, 4, 4)
    """
    # Validate input shapes
    if means.ndim != 2 or means.shape[1] != 3:
        raise ValueError(f"Expected means with shape (N, 3), got {means.shape}")
    if semi_axes.ndim != 2 or semi_axes.shape[1] != 3:
        raise ValueError(f"Expected semi_axes with shape (N, 3), got {semi_axes.shape}")
    if quaternions.ndim != 2 or quaternions.shape[1] != 4:
        raise ValueError(f"Expected quaternions with shape (N, 4), got {quaternions.shape}")

    # Check that all arrays have the same first dimension
    N = means.shape[0]
    if semi_axes.shape[0] != N or quaternions.shape[0] != N:
        raise ValueError(f"Inconsistent batch sizes: means {means.shape[0]}, "
                         f"semi_axes {semi_axes.shape[0]}, quaternions {quaternions.shape[0]}")

    rots = quaternions_to_rotation_matrices(quaternions)

    mats = []
    num_matrices = len(means)
    for i in range(num_matrices):
        scale_matrix = np.diag(semi_axes[i])
        rot_matrix = rots[i]
        mean = means[i]

        H = np.eye(4)
        H[:3, :3] = scale_matrix @ rot_matrix
        H[:3, 3] = mean

        mats.append(H)

    mats = np.array(mats)
    return mats

def sample_ellipsoid_union_surface(
    means: NDArray[np.float64],
    semi_axes: NDArray[np.float64],
    quaternions: NDArray[np.float64],
    num_samples: int
) -> Dict[int, NDArray[np.float64]]:
    """
    Samples surface of ellipsoid union with rejection sampling.
    Returns dictionary mapping gaussian id to buffer of 3D points.

    Args:
        means: Array of mean positions with shape (N, 3)
        semi_axes: Array of semi-axes with shape (N, 3)
        quaternions: Array of quaternions with shape (N, 4)
        num_samples: Number of points to sample

    Returns:
        Dictionary mapping ellipsoid ID to array of points with shape (M, 3)
    """
    # Validate input shapes
    if means.ndim != 2 or means.shape[1] != 3:
        raise ValueError(f"Expected means with shape (N, 3), got {means.shape}")
    if semi_axes.ndim != 2 or semi_axes.shape[1] != 3:
        raise ValueError(f"Expected semi_axes with shape (N, 3), got {semi_axes.shape}")
    if quaternions.ndim != 2 or quaternions.shape[1] != 4:
        raise ValueError(f"Expected quaternions with shape (N, 4), got {quaternions.shape}")

    # Check that all arrays have the same first dimension
    N = means.shape[0]
    if semi_axes.shape[0] != N or quaternions.shape[0] != N:
        raise ValueError(f"Inconsistent batch sizes: means {means.shape[0]}, "
                         f"semi_axes {semi_axes.shape[0]}, quaternions {quaternions.shape[0]}")

    # Init output buffer, map of ellipsoid id -> samples
    surface_union_samples = defaultdict(list)

    # Calculate samples per ellipsoid based on relative surface area.
    # Ensures more even distribution of points across union.
    avg_semi_axes = np.mean(semi_axes, axis=1)
    surface_areas = 4 * np.pi * avg_semi_axes**2
    total_area = np.sum(surface_areas)
    samples_per_ellipsoid = \
        np.maximum(1, np.floor(num_samples * surface_areas / total_area)).astype(int)
    remaining = num_samples - np.sum(samples_per_ellipsoid)
    if remaining > 0:
        # Add remaining samples to ellipsoids with largest surface areas
        indices = np.argsort(-surface_areas)[:remaining]
        samples_per_ellipsoid[indices] += 1

    # Convert gaussian info into basis frames
    g_mats = build_homogeneous_matrices(means, semi_axes, quaternions)

    # Surface sampling per ellipsoid
    n_ellipsoids = len(means)
    for i in range(n_ellipsoids):
        # Current local basis
        G = g_mats[i]

        # Collect true surface points
        num_collected = 0
        num_target = samples_per_ellipsoid[i]
        batch_size = 1000
        while num_collected < num_target:
            # Sample unit sphere
            rng = np.random.RandomState(42)
            xyz = rng.normal(0, 1, size=(batch_size, 3))
            xyz = xyz / np.linalg.norm(xyz, axis=1, keepdims=True)

            # Apply gaussian/ellipsoid basis
            # Ref: (G @ xyz.T).T = xyzw @ G.T
            candidates = (xyz @ G[:3, :3].T) + G[:3, 3]

            # Change of basis into ALL other ellipsoids
            # for intersection check.
            valid_mask = np.ones(batch_size, dtype=bool)
            for j in range(n_ellipsoids):
                if i == j:
                    continue
                if num_collected >= num_target:
                    break

                N = np.linalg.inv(g_mats[j])
                tmp = (candidates @ N[:3, :3].T) + N[:3, 3]
                inside_mask = np.sum(tmp**2, axis=1) <= 1
                valid_mask = valid_mask & ~inside_mask

            # These samples passed inner loop check
            valid_candidates = candidates[valid_mask]
            if len(valid_candidates) > 0:
                # Add diff/fill target container up to the top
                to_add = min(len(valid_candidates), num_target - num_collected)
                surface_union_samples[i].extend(valid_candidates[:to_add])
                num_collected += to_add

    return {k: np.array(v) for k, v in surface_union_samples.items()}

def sample_gaussian_tracks(
    means: NDArray[np.float64],
    semi_axes: NDArray[np.float64],
    quaternions: NDArray[np.float64],
    num_samples: int
) -> NDArray[np.float64]:
    """
    Samples ellipsoid union surface
    and produces tracks by applying local semi_axes
    and quaternion transformations overtime.

    Args:
        means: Array of mean positions with shape (T, N, 3) where T is time and N is number of ellipsoids
        semi_axes: Array of semi-axes with shape (T, N, 3)
        quaternions: Array of quaternions with shape (T, N, 4)
        num_samples: Number of points to sample

    Returns:
        Array of point trajectories with shape (T, M, 3) where M is the total number of points
    """
    # Validate input shapes
    if means.ndim != 3 or means.shape[2] != 3:
        raise ValueError(f"Expected means with shape (T, N, 3), got {means.shape}")
    if semi_axes.ndim != 3 or semi_axes.shape[2] != 3:
        raise ValueError(f"Expected semi_axes with shape (T, N, 3), got {semi_axes.shape}")
    if quaternions.ndim != 3 or quaternions.shape[2] != 4:
        raise ValueError(f"Expected quaternions with shape (T, N, 4), got {quaternions.shape}")

    # Check that all arrays have the same dimensions
    T, N = means.shape[0], means.shape[1]
    if semi_axes.shape[0] != T or semi_axes.shape[1] != N:
        raise ValueError(f"Inconsistent shape for semi_axes: expected ({T}, {N}, 3), got {semi_axes.shape}")
    if quaternions.shape[0] != T or quaternions.shape[1] != N:
        raise ValueError(f"Inconsistent shape for quaternions: expected ({T}, {N}, 4), got {quaternions.shape}")

    # Init buffer of output tracks
    output_tracks = []

    # Init starting points of output tracks
    means_0 = means[0, ...]
    semi_axes_0 = semi_axes[0, ...]
    quaternions_0 = quaternions[0, ...]
    gaussian_pts = sample_ellipsoid_union_surface(means_0,
                                                  semi_axes_0,
                                                  quaternions_0,
                                                  num_samples)
    current_timestep = []
    for g_id, global_pts in gaussian_pts.items():
        current_timestep.extend(global_pts)
    output_tracks.append(current_timestep)

    # Resolve local coordinates for rollout
    local_gaussian_pts = {}
    g_mats = build_homogeneous_matrices(means_0,
                                        semi_axes_0,
                                        quaternions_0)
    for g_id, global_pts in gaussian_pts.items():
        to_local = np.linalg.inv(g_mats[g_id])
        local_pts = (global_pts @ to_local[:3, :3].T) + to_local[:3, 3]
        local_gaussian_pts[g_id] = local_pts

    # Rollout the track
    timesteps = len(means)
    for i in range(1, timesteps):
        current_timestep = []
        g_mats = build_homogeneous_matrices(means[i, ...],
                                            semi_axes[i, ...],
                                            quaternions[i, ...])
        for g_id, local_pts in local_gaussian_pts.items():
            to_global = g_mats[g_id]
            global_pts = (local_pts @ to_global[:3, :3].T) + to_global[:3, 3]
            current_timestep.extend(global_pts)

        output_tracks.append(current_timestep)

    # (DONE) First, add gaussian_pts to output_tracks in ascending gaussian id order.
    # (DONE) Next, use homogeneous utility to get all g_mat transforms.
    # (DONE) Next, use inverse(g_mats[0, ...]) to get local_gaussian_pts.
    # (DONE) Finally, for g_mats[t >= 1, ...], apply g_mat[t] to get track at timestep t.
    #   Add to output_tracks in ascending gaussian id order.

    # NOTE:
    # Rolling this out almost certainly causes self-intersection.
    # In that case, I will likely have to 'rectify tracks'.
    # I can densely sample the ellipsoid union again, and map the tracked point to the nearest surface point.
    # ^TODO, fix at later time.

    output_tracks = np.array(output_tracks)
    return output_tracks

def log_image():
    pass

def log_video():
    pass

def log_gaussian_sequence():
    pass

def log_tracks():
    pass