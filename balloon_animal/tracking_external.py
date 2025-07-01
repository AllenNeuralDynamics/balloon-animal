"""
External Utilities for 3DGS Tracking
"""
import numpy as np
import torch
from scipy.spatial import cKDTree

from diff_gaussian_rasterization import GaussianRasterizationSettings as Camera


def setup_camera(w, h, k, w2c, near=0.01, far=100):
    fx, fy, cx, cy = k[0][0], k[1][1], k[0][2], k[1][2]
    w2c = torch.tensor(w2c).cuda().float()
    cam_center = torch.inverse(w2c)[:3, 3]
    w2c = w2c.unsqueeze(0).transpose(1, 2)
    opengl_proj = torch.tensor([[2 * fx / w, 0.0, -(w - 2 * cx) / w, 0.0],
                                [0.0, 2 * fy / h, -(h - 2 * cy) / h, 0.0],
                                [0.0, 0.0, far / (far - near), -(far * near) / (far - near)],
                                [0.0, 0.0, 1.0, 0.0]]).cuda().float().unsqueeze(0).transpose(1, 2)
    full_proj = w2c.bmm(opengl_proj)
    cam = Camera(
        image_height=h,
        image_width=w,
        tanfovx=w / (2 * fx),
        tanfovy=h / (2 * fy),
        bg=torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda"),
        scale_modifier=1.0,
        viewmatrix=w2c,
        projmatrix=full_proj,
        sh_degree=0,
        campos=cam_center,
        prefiltered=False
    )
    return cam

def weighted_l2_loss_v2(x, y, w=1.0):
    return torch.sqrt(((x - y) ** 2).sum(-1) * w + 1e-20).mean()

def o3d_knn(pts, num_knn):
    # Not o3d anymore, but keeping signature for minimal maintainence
    # Converted by ChatGPT.

    # Create a KD-tree from the points
    tree = cKDTree(pts)
    
    # Preallocate arrays for indices and squared distances
    sq_dists = np.zeros((len(pts), num_knn), dtype=float)
    indices = np.zeros((len(pts), num_knn), dtype=int)
    
    # Search for k-nearest neighbors for each point
    for i, p in enumerate(pts):
        # Query returns distances and indices
        # We use the 2nd element onwards to exclude the point itself
        dists, idx = tree.query(p, k=num_knn + 1)
        
        sq_dists[i] = dists[1:]**2  # Convert to squared distances
        indices[i] = idx[1:]  # Exclude the point itself
    
    return sq_dists, indices

def params2cpu(params):
    """
    Transfer parameters to CPU, handling both PyTorch tensors and other data types.
    
    Args:
        params: Dictionary of parameters
        
    Returns:
        Dictionary with all parameters converted to numpy arrays on CPU
    """
    result = {}
    for k, v in params.items():
        if isinstance(v, torch.Tensor):
            # Handle PyTorch tensors
            result[k] = v.detach().cpu().contiguous().numpy()
        else:
            # Handle non-tensor values (floats, ints, numpy arrays, etc.)
            result[k] = np.array(v)
    return result

def save_params(output_params, output_path):
    """
    Save parameters to a file, truncating buffers to minimum size for each key.
    
    Args:
        output_params: List of parameter dictionaries where each dict contains
                      arrays of potentially different sizes
        output_path: Path to save the parameters
    """
    to_save = {}
    
    # Get all unique keys from all timesteps
    all_keys = set()
    for params in output_params:
        all_keys.update(params.keys())
    
    for k in all_keys:
        # Check if this key exists in all timesteps
        if all(k in params for params in output_params):
            # Get all arrays for this key
            arrays = [params[k] for params in output_params]
            
            # Find minimum size along the first dimension (buffer size)
            min_size = min(arr.shape[0] for arr in arrays)
            
            # Truncate all arrays to minimum size
            truncated_arrays = [arr[:min_size] for arr in arrays]
            
            # Stack the truncated arrays
            to_save[k] = np.stack(truncated_arrays)
        else:
            # If it only exists in some timesteps, save from the first occurrence
            for params in output_params:
                if k in params:
                    to_save[k] = params[k]
                    break
    
    np.savez(output_path, **to_save)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = func.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = func.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = func.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = func.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = func.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def calc_ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def calc_psnr(img1, img2):
    mse = ((img1 - img2) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))