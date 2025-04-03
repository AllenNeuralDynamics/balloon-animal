"""
Variety of utilities.
"""
from collections import defaultdict
from typing import TypeVar, Dict
import torch
from torch import Tensor

T = TypeVar('T')
N = TypeVar('N')

def quaternions_to_rotation_matrices(q: Tensor) -> Tensor:
    """
    Convert batch of unnormalized quaternions [r, x, y, z]
    into batch of rotation matrices.

    Args:
        q: Tensor of quaternions with shape (N, 4)

    Returns:
        Tensor of rotation matrices with shape (N, 3, 3)
    """
    # Validate input shape
    if q.dim() != 2 or q.shape[1] != 4:
        raise ValueError(f"Expected quaternions with shape (N, 4), got {q.shape}")

    # Normalize quaternions
    norm = torch.sqrt(q[:, 0]**2 + q[:, 1]**2 + q[:, 2]**2 + q[:, 3]**2)
    q = q / norm.unsqueeze(1)

    # Initialize rotation matrix tensor
    rot = torch.zeros((q.shape[0], 3, 3), device=q.device, dtype=q.dtype)

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
    means: Tensor,
    semi_axes: Tensor,
    quaternions: Tensor,
) -> Tensor:
    """
    Convert gaussian means, semi_axes, and quaternions into basis frames.

    Args:
        means: Tensor of mean positions with shape (N, 3)
        semi_axes: Tensor of semi-axes with shape (N, 3)
        quaternions: Tensor of quaternions with shape (N, 4)

    Returns:
        Tensor of homogeneous transformation matrices with shape (N, 4, 4)
    """
    # Validate input shapes
    if means.dim() != 2 or means.shape[1] != 3:
        raise ValueError(f"Expected means with shape (N, 3), got {means.shape}")
    if semi_axes.dim() != 2 or semi_axes.shape[1] != 3:
        raise ValueError(f"Expected semi_axes with shape (N, 3), got {semi_axes.shape}")
    if quaternions.dim() != 2 or quaternions.shape[1] != 4:
        raise ValueError(f"Expected quaternions with shape (N, 4), got {quaternions.shape}")

    # Check that all arrays have the same first dimension
    N = means.shape[0]
    if semi_axes.shape[0] != N or quaternions.shape[0] != N:
        raise ValueError(f"Inconsistent batch sizes: means {means.shape[0]}, "
                         f"semi_axes {semi_axes.shape[0]}, quaternions {quaternions.shape[0]}")

    rots = quaternions_to_rotation_matrices(quaternions)

    # Create batch of homogeneous matrices
    mats = torch.zeros((N, 4, 4), device=means.device, dtype=means.dtype)

    # Set diagonal to 1 (equivalent to eye for each matrix)
    mats[:, torch.arange(4), torch.arange(4)] = 1.0

    # Apply scale and rotation to the top-left 3x3 block
    for i in range(N):
        scale_matrix = torch.diag(semi_axes[i])
        mats[i, :3, :3] = scale_matrix @ rots[i]
        mats[i, :3, 3] = means[i]

    return mats

def sample_ellipsoid_union_surface(
    means: Tensor,
    semi_axes: Tensor,
    quaternions: Tensor,
    num_samples: int
) -> Dict[int, Tensor]:
    """
    Samples surface of ellipsoid union with rejection sampling.
    Returns dictionary mapping gaussian id to buffer of 3D points.
    Uses uniform spherical coordinate sampling for better distribution.

    Args:
        means: Tensor of mean positions with shape (N, 3)
        semi_axes: Tensor of semi-axes with shape (N, 3)
        quaternions: Tensor of quaternions with shape (N, 4)
        num_samples: Number of points to sample

    Returns:
        Dictionary mapping ellipsoid ID to tensor of points with shape (M, 3)
    """
    # Validate input shapes
    if means.dim() != 2 or means.shape[1] != 3:
        raise ValueError(f"Expected means with shape (N, 3), got {means.shape}")
    if semi_axes.dim() != 2 or semi_axes.shape[1] != 3:
        raise ValueError(f"Expected semi_axes with shape (N, 3), got {semi_axes.shape}")
    if quaternions.dim() != 2 or quaternions.shape[1] != 4:
        raise ValueError(f"Expected quaternions with shape (N, 4), got {quaternions.shape}")

    # Check that all arrays have the same first dimension
    N = means.shape[0]
    if semi_axes.shape[0] != N or quaternions.shape[0] != N:
        raise ValueError(f"Inconsistent batch sizes: means {means.shape[0]}, "
                         f"semi_axes {semi_axes.shape[0]}, quaternions {quaternions.shape[0]}")

    # Init output buffer, map of ellipsoid id -> samples
    surface_union_samples = defaultdict(list)
    device = means.device

    # Calculate samples per ellipsoid based on relative surface area.
    # Ensures more even distribution of points across union.
    avg_semi_axes = torch.mean(semi_axes, dim=1)
    surface_areas = 4 * torch.pi * avg_semi_axes**2
    total_area = torch.sum(surface_areas)
    samples_per_ellipsoid = torch.maximum(
        torch.tensor(1.0, device=device),
        torch.floor(num_samples * surface_areas / total_area)
    ).int()

    remaining = num_samples - torch.sum(samples_per_ellipsoid)
    if remaining > 0:
        # Add remaining samples to ellipsoids with largest surface areas
        _, indices = torch.sort(surface_areas, descending=True)
        indices = indices[:remaining]
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
        num_target = int(samples_per_ellipsoid[i].item())
        batch_size = 10000

        print(f'Sampling points for ellipsoid {i}')
        print(f"{num_target=}")

        while num_collected < num_target:
            # Sample unit sphere uniformly using spherical coordinates
            # phi: (0 to 2π)
            # theta: (0 to π)
            # Set seed for reproducibility
            generator = torch.Generator(device=device)
            generator.manual_seed(42)

            phi = torch.rand(batch_size, device=device, generator=generator) * 2 * torch.pi
            theta = torch.arccos(2 * torch.rand(batch_size, device=device, generator=generator) - 1)

            # Convert spherical to Cartesian coordinates
            x = torch.sin(theta) * torch.cos(phi)
            y = torch.sin(theta) * torch.sin(phi)
            z = torch.cos(theta)
            xyz = torch.stack((x, y, z), dim=1)

            # Apply gaussian/ellipsoid basis
            # Ref: (G @ xyz.T).T = xyzw @ G.T
            candidates = torch.matmul(xyz, G[:3, :3].T) + G[:3, 3]

            # Change of basis into ALL other ellipsoids
            # for intersection check.
            valid_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
            for j in range(n_ellipsoids):
                if i == j:
                    continue
                if num_collected >= num_target:
                    break

                N = torch.inverse(g_mats[j])
                tmp = torch.matmul(candidates, N[:3, :3].T) + N[:3, 3]
                inside_mask = torch.sum(tmp**2, dim=1) <= 1
                valid_mask = valid_mask & ~inside_mask

            # These samples passed inner loop check
            valid_candidates = candidates[valid_mask]
            if len(valid_candidates) > 0:
                print(f'entered, {len(valid_candidates)=}')
                # Add diff/fill target container up to the top
                to_add = min(len(valid_candidates), num_target - num_collected)
                surface_union_samples[i].append(valid_candidates[:to_add])
                num_collected += to_add

    # Convert lists of tensors to single tensors for each ellipsoid
    return {k: torch.cat(v, dim=0) for k, v in surface_union_samples.items()}

def sample_gaussian_tracks(
    means: Tensor,
    semi_axes: Tensor,
    quaternions: Tensor,
    num_samples: int
) -> Tensor:
    """
    Samples ellipsoid union surface
    and produces tracks by applying local semi_axes
    and quaternion transformations overtime.

    Args:
        means: Tensor of mean positions with shape (T, N, 3) where T is time and N is number of ellipsoids
        semi_axes: Tensor of semi-axes with shape (T, N, 3)
        quaternions: Tensor of quaternions with shape (T, N, 4)
        num_samples: Number of points to sample

    Returns:
        Tensor of point trajectories with shape (T, M, 3) where M is the total number of points
    """
    # Validate input shapes
    if means.dim() != 3 or means.shape[2] != 3:
        raise ValueError(f"Expected means with shape (T, N, 3), got {means.shape}")
    if semi_axes.dim() != 3 or semi_axes.shape[2] != 3:
        raise ValueError(f"Expected semi_axes with shape (T, N, 3), got {semi_axes.shape}")
    if quaternions.dim() != 3 or quaternions.shape[2] != 4:
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
        current_timestep.append(global_pts)
    output_tracks.append(torch.cat(current_timestep, dim=0))

    # Resolve local coordinates for rollout
    local_gaussian_pts = {}
    g_mats = build_homogeneous_matrices(means_0,
                                      semi_axes_0,
                                      quaternions_0)
    for g_id, global_pts in gaussian_pts.items():
        to_local = torch.inverse(g_mats[g_id])
        local_pts = torch.matmul(global_pts, to_local[:3, :3].T) + to_local[:3, 3]
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
            global_pts = torch.matmul(local_pts, to_global[:3, :3].T) + to_global[:3, 3]
            current_timestep.append(global_pts)
        output_tracks.append(torch.cat(current_timestep, dim=0))

    # Stack all timesteps
    output_tracks = torch.stack(output_tracks)
    return output_tracks

def log_image():
    pass

def log_video():
    pass

def log_gaussian_sequence():
    pass

def log_tracks():
    pass