"""
Personal Utilities for 3DGS Tracking
"""
import torch
from torch import Tensor

def sample_points_in_sphere(N, r=1.0, seed=42, device='cuda'):
    """
    Uniformly samples N points inside a sphere of radius r, natively on GPU.

    Parameters:
    N (int): Number of points to sample.
    r (float): Radius of the sphere.
    seed (int or None): Optional seed for reproducibility.
    device (str or torch.device): Device to sample on (default: 'cuda').

    Returns:
    torch.Tensor: Tensor of shape (N, 3) with sampled 3D points on specified device.
    """
    if seed is not None:
        torch.manual_seed(seed)

    # Sample points from normal distribution and normalize to unit sphere
    directions = torch.randn(N, 3, device=device)
    directions /= torch.norm(directions, dim=1, keepdim=True)

    # Sample radius with cube root to ensure uniform volume distribution
    radii = torch.rand(N, device=device) ** (1/3)
    radii *= r

    # Scale directions by radii
    points = directions * radii.unsqueeze(1)

    return points

def sample_ellipsoid_surfaces(means, semi_axes, num_samples):
    """
    Sample surface points from a union of axis-aligned ellipsoids (no intersection check).
    Args:
        means: (N, 3) tensor of ellipsoid centers
        semi_axes: (N, 3) tensor of semi-axes (a, b, c)
        num_samples: total number of surface samples to generate
    Returns:
        (num_samples, 3) tensor of sampled 3D points
    """
    # Ensure inputs are PyTorch tensors
    means = torch.as_tensor(means)
    semi_axes = torch.as_tensor(semi_axes)

    # Get device and dtype from input tensors
    device = means.device
    dtype = means.dtype

    # Estimate surface area using Knud Thomsen's approximation
    a, b, c = semi_axes.T
    p = 1.6075
    areas = 4 * torch.pi * ((a**p * b**p + a**p * c**p + b**p * c**p) / 3)**(1/p)
    weights = areas / areas.sum()

    # Allocate samples proportionally
    samples_per_ellipsoid = torch.floor(weights * num_samples).long()
    remainder = num_samples - samples_per_ellipsoid.sum()

    # Fill remainder by adding to ellipsoids with largest weights
    _, indices = torch.sort(weights, descending=True)
    samples_per_ellipsoid[indices[:remainder]] += 1

    # Sample each ellipsoid
    all_samples = []
    for mean, axes, n_samples in zip(means, semi_axes, samples_per_ellipsoid):
        # Create random tensors on the same device
        phi = torch.rand(n_samples, device=device, dtype=dtype) * 2 * torch.pi
        costheta = torch.rand(n_samples, device=device, dtype=dtype) * 2 - 1  # Uniform between -1 and 1
        sintheta = torch.sqrt(1 - costheta**2)

        # Unit sphere to ellipsoid surface
        directions = torch.stack([
            sintheta * torch.cos(phi),
            sintheta * torch.sin(phi),
            costheta
        ], dim=1)

        points = directions * axes + mean
        all_samples.append(points)

    return torch.cat(all_samples, dim=0)

def calc_ellipsoid_union_sdf(
    query_points: torch.Tensor,
    ellipsoid_centers: torch.Tensor,
    ellipsoid_scales: torch.Tensor
) -> tuple:
    """
    Calculates ellipsoid union sdf.

    Args:
        query_points (torch.Tensor): (N, 3) xyz queries.
        ellipsoid_centers (torch.Tensor): (M, 3) xyz centers.
        ellipsoid_scales (torch.Tensor): (M, 3) xyz scales.
    Ellipsoid centers + ellipsoid scales having corresponding indices.
    Assuming the ellipsoids have no rotation.

    Returns:
        torch.Tensor: SDF of query points.
        -: inside union, 0: on union surface, +: outside union
        torch.Tensor: Indices of closest ellipsoid parent.
    """
    # Ensure inputs are PyTorch tensors
    query_points = torch.as_tensor(query_points)
    ellipsoid_centers = torch.as_tensor(ellipsoid_centers)
    ellipsoid_scales = torch.as_tensor(ellipsoid_scales)

    # Vectorized implementation for all query points at once
    # Reshape for broadcasting: (N, 1, 3) - (1, M, 3) = (N, M, 3)
    normalized_dists = (query_points.unsqueeze(1) - ellipsoid_centers.unsqueeze(0)) / ellipsoid_scales.unsqueeze(0)

    # Compute SDF for each query point to each ellipsoid: (N, M)
    sdfs = torch.norm(normalized_dists, dim=2) - 1.0

    # Find minimum SDF and corresponding parent index for each query point
    union_sdf, union_parents = torch.min(sdfs, dim=1)

    return union_sdf, union_parents

def quaternions_to_rotation_matrices(q: torch.Tensor) -> torch.Tensor:
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
    means: torch.Tensor,
    semi_axes: torch.Tensor,
    quaternions: torch.Tensor,
) -> torch.Tensor:
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

def filter_ellipsoids_by_overlap(
    ellipsoid_centers: torch.Tensor,
    ellipsoid_scales: torch.Tensor,
    overlap_threshold: float = 0.9
) -> torch.Tensor:
    """
    Filters ellipsoids by progressively growing a keep set.
    The filtering first sorts input ellipsoids by volume, then
    iterates through the sorted buffer checking if each candidate
    ellipsoid actually adds detail to the reconstruction, determined by
    an overlap threshold.

    Input:
        ellipsoid_centers (torch.Tensor): (N, 3) xyz centers.
        ellipsoid_scales (torch.Tensor): (N, 3) xyz scales.
        overlap_threshold (float): Amount of overlap between two ellipsoids that
            that will trigger filter. Between two overlapping ellipsoids, the
            smaller ellipsoid in volume is filtered out.

    Returns:
        torch.Tensor: integer tensor of keep indices.
    """
    # Ensure inputs are PyTorch tensors
    ellipsoid_centers = torch.as_tensor(ellipsoid_centers)
    ellipsoid_scales = torch.as_tensor(ellipsoid_scales)

    # Sort ellipsoids by volume, descending order
    volumes = (4/3) * torch.pi * torch.prod(ellipsoid_scales, dim=1)
    sorted_indices = torch.argsort(volumes, descending=True)

    # Initialize keep_indices with the largest ellipsoid
    keep_indices = [sorted_indices[0].item()]

    # Check each remaining ellipsoid
    for i in sorted_indices[1:]:
        i = i.item()
        # Collect query points
        query_pts = sample_points_in_sphere(500)
        query_pts = (query_pts * ellipsoid_scales[i]) + ellipsoid_centers[i]

        # Evaluate query points against current kept ellipsoids
        sdf, _ = calc_ellipsoid_union_sdf(
            query_pts,
            ellipsoid_centers[keep_indices],
            ellipsoid_scales[keep_indices]
        )

        # Conditionally add to keep_indices based on overlap
        overlap = torch.sum(sdf < 0).float() / len(query_pts)
        if overlap < overlap_threshold:
            keep_indices.append(i)

    return torch.tensor(keep_indices, dtype=torch.long)

def sample_ellipsoid_union_surface(
    ellipsoid_centers: torch.Tensor,
    ellipsoid_scales: torch.Tensor,
    num_samples: int = 50000
) -> dict:
    """
    Produces samples of ellipsoid union.

    Args:
        ellipsoid_centers (torch.Tensor): (N, 3) xyz centers.
        ellipsoid_scales (torch.Tensor): (N, 3) xyz scales.
        num_samples (int): Number of surface samples to generate.

    Returns:
       dict: gaussian_id -> global 3D coordinate tensor.
    """
    # Ensure inputs are PyTorch tensors
    ellipsoid_centers = torch.as_tensor(ellipsoid_centers)
    ellipsoid_scales = torch.as_tensor(ellipsoid_scales)

    # Sample initial points on ellipsoid surfaces
    query_points = sample_ellipsoid_surfaces(ellipsoid_centers, ellipsoid_scales, num_samples)

    # Calculate SDF and parent ellipsoid for each point
    sdf, parent = calc_ellipsoid_union_sdf(
        query_points, ellipsoid_centers, ellipsoid_scales
    )

    # Filter points that are approximately on the surface
    tolerance = 1e-2  # Within 1/100 of the surface
    surface_filter = torch.abs(sdf) < tolerance

    # Create output map keyed by ellipsoid index
    output_map = {}
    for i in range(len(ellipsoid_centers)):
        parent_filter = (parent == i)
        full_condition = surface_filter & parent_filter
        output_map[i] = query_points[full_condition]

    return output_map

def _quaternion_multiply_batch(q1, q2):
    """
    Multiply quaternions in batch.

    Args:
        q1 (torch.Tensor): First quaternions, shape (N, 4) in (r, x, y, z) format
        q2 (torch.Tensor): Second quaternions, shape (N, 4) in (r, x, y, z) format

    Returns:
        torch.Tensor: Product quaternions, shape (N, 4)
    """
    r1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    r2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]

    # Quaternion multiplication formula
    r = r1 * r2 - x1 * x2 - y1 * y2 - z1 * z2
    x = r1 * x2 + x1 * r2 + y1 * z2 - z1 * y2
    y = r1 * y2 - x1 * z2 + y1 * r2 + z1 * x2
    z = r1 * z2 + x1 * y2 - y1 * x2 + z1 * r2

    return torch.stack([r, x, y, z], dim=1)

def quaternion_rotate_points(quaternion, points):
    """
    Apply quaternion rotation to a buffer of 3D points.

    Args:
        quaternion (torch.Tensor): Quaternion in (r, x, y, z) format, shape (4,)
                                 where r is the real/scalar part
        points (torch.Tensor): Points to rotate, shape (N, 3)

    Returns:
        torch.Tensor: Rotated points, shape (N, 3)
    """
    # Ensure inputs are float tensors
    quaternion = quaternion.float()
    points = points.float()

    # Normalize quaternion to unit quaternion
    quaternion = quaternion / torch.norm(quaternion)

    # Extract quaternion components (r, x, y, z)
    r, x, y, z = quaternion[0], quaternion[1], quaternion[2], quaternion[3]

    # Convert points to homogeneous quaternions (0, x, y, z)
    # Shape: (N, 4) where first column is 0
    point_quats = torch.cat([torch.zeros(points.shape[0], 1, device=points.device), points], dim=1)

    # Quaternion conjugate for q* = (r, -x, -y, -z)
    q_conj = torch.tensor([r, -x, -y, -z], device=quaternion.device)

    # Apply rotation: q * p * q*
    # First: q * p
    rotated = _quaternion_multiply_batch(quaternion.unsqueeze(0).expand(points.shape[0], -1), point_quats)

    # Second: (q * p) * q*
    rotated = _quaternion_multiply_batch(rotated, q_conj.unsqueeze(0).expand(points.shape[0], -1))

    # Extract the rotated 3D points (ignore the scalar part)
    return rotated[:, 1:]

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
