"""
Core Tracking Algorithm. 

"""

import torch
from torchvision import transforms

from pathlib import Path
from collections import defaultdict
import copy
import time
from tqdm import tqdm
import yaml

import numpy as np
import open3d as o3d
from PIL import Image

from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from tracking_external import setup_camera, weighted_l2_loss_v2, \
    o3d_knn, params2cpu, save_params, calc_ssim, calc_psnr

from init_utils import (
    build_homogeneous_matrices,
    filter_ellipsoids_by_overlap,
    sample_ellipsoid_union_surface,
    quaternion_rotate_points
)

"""
TODO: 
- Precomputed Capsule needs: 
  - Point clouds (generated faster please)
  - Metadata yml (Yes, the cameras are in different spots)

- Client script: 
  - Write App Panel
  - Write Stream to aind-scratch-data for immediate availability + easy sharing 
  (The dataset should not live in CO considering training is on HPC/remote cluster)
  - Write Parallel capsule submission

- Integration Test
- Wandb logging? Ctrl + F to remove after testing. 
"""

def read_config_yaml(yaml_path: str) -> dict:
    with open(yaml_path, "r") as f:
        yaml_dict = yaml.safe_load(f)
    return yaml_dict

def get_dataset(t: int, dataset_path: Path, metadata: dict, camera_subset: list[str]=[]):
    """
    Dataset is a list of dictionaries containing:
    - camera object
    - camera_id
    - ground truth image
    - ground truth seg mask
    """
    assert (list(metadata['extrinsic_matrices'].keys()) ==
            list(metadata['intrinsic_matrices'].keys())), 'metadata must contain same number of extrinsic and intrinsic matrices.'
    assert all([cam_id in list(metadata['extrinsic_matrices'].keys()) 
                for cam_id in camera_subset]), \
        'Camera subset must be a subset of available cameras.'

    h = metadata['camera_height']
    w = metadata['camera_width']

    # Calculate focal_pts, scene center to determine far plane of camera frustrum.
    cam_centers = []
    for cam in metadata['extrinsic_matrices'].keys():
        ext = np.array(metadata['extrinsic_matrices'][cam])
        cam_centers.append(np.linalg.inv(ext)[:3, 3])
    scene_center = np.mean(np.array(cam_centers), axis=0)

    dataset = []
    for i, cam_id in enumerate(camera_subset):
        # far_plane_z is important for gaussian visibility!
        w2c = np.array(metadata['extrinsic_matrices'][cam_id])
        k = np.array(metadata['intrinsic_matrices'][cam_id])
        far_plane_z = np.linalg.norm(scene_center - cam_centers[i]) * 2.
        cam = setup_camera(w, h, k, w2c, near=1.0, far=far_plane_z)

        # Searching directory for image/seg mask corresponding
        # to specific timestamp 't'.
        # Image is normalized [0, 1]
        # Seg is also normalized [0, 1]
        found_imgs = []
        img_cam_dir = dataset_path / 'img' / f'cam_{cam_id}'
        found_imgs.extend(img_cam_dir.glob(f'{t}.jpg'))
        found_imgs.extend(img_cam_dir.glob(f'{t}.png'))
        im = np.array(copy.deepcopy(Image.open(str(found_imgs[0]))))
        im = torch.tensor(im).float().cuda().permute(2, 0, 1) / 255

        found_segs = []
        seg_cam_dir = dataset_path / 'seg' / f'cam_{cam_id}'
        found_segs.extend(seg_cam_dir.glob(f'{t}.jpg'))
        found_segs.extend(seg_cam_dir.glob(f'{t}.png'))
        seg = np.array(copy.deepcopy(Image.open(str(found_segs[0])))).astype(np.float32)
        seg = torch.tensor(seg).float().cuda() / 255

        dataset.append({'cam': cam,
                        'im': im,
                        'seg': seg,
                        'id': i,
                        'cam_near_plane': 1.0,
                        'cam_far_plane': far_plane_z})
    return dataset

def initialize_params(t: int, dataset_path: Path, pc_path: str, metadata: dict, downsample_lvl: int):
    """
    Initalize application params and vars.
    """

    # Downsampling input PC by factor (mean distance between pts)
    mouse_pc = np.load(pc_path)
    sq_dist, _ = o3d_knn(mouse_pc, num_knn=3)
    mean_sq_dist = np.mean(sq_dist)
    factor = 1 + (downsample_lvl * 0.25)
    voxel_size = mean_sq_dist * factor

    # Starting point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(mouse_pc)
    downsampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    means3d = np.asarray(downsampled_pcd.points)

    # Metadata
    sq_dist, _ = o3d_knn(means3d, 3)
    mean3_sq_dist = sq_dist.mean(-1).clip(min=0.0000001)
    total_pts = len(means3d)

    # Starting Color, white
    rgb_colors = np.ones((total_pts, 3))

    # Starting parameter buffer
    params = {
        'means3D': means3d,
        'rgb_colors': rgb_colors,
        'unnorm_rotations': np.tile([1, 0, 0, 0], (total_pts, 1)),
        'logit_opacities': np.ones((total_pts, 1)),  # Fully opaque
        'log_scale': np.log(np.sqrt(mean3_sq_dist))[..., None]
    }
    params = {k: torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True)) for k, v in
              params.items()}

    cam_exts = []
    for cam in metadata['extrinsic_matrices'].keys():
        cam_exts.append(np.array(metadata['extrinsic_matrices'][cam]))
    cam_exts = np.stack(cam_exts)
    cam_centers = np.linalg.inv(cam_exts)[:, :3, 3]  # Get scene radius
    scene_radius = 1.1 * np.max(np.linalg.norm(cam_centers - np.mean(cam_centers, 0)[None], axis=-1))
    print('scene radius', scene_radius)
    print(f"{params['means3D'].shape=}")

    return params, scene_radius

# ------
# Training Stages
def initialize_recon_optimizer(params, scene_radius):
    """
    Optimize position and scale.
    """
    param_groups = \
        [
            {'params': [params['means3D']],
             'name': 'means3D',
             'lr': 0.001 * scene_radius
            },
            {'params': [params['log_scale']],
              'name': 'log_scale',
              'lr': 0.001
            }
        ]

    return torch.optim.Adam(param_groups)

def initialize_recon_optimizer_2(params, scene_radius):
    """
    Optimize covariance shape.
    """

    param_groups = \
        [
            {'params': [params['log_scales']],
             'name': 'log_scales',
             'lr': 0.001
            }
        ]
    return torch.optim.Adam(param_groups)

def initialize_recon_optimizer_3a(params):
    """
    Optimize base gaussian average color.
    """

    param_groups = \
        [
            {'params': [params['rgb_colors']],
             'name': 'rgb_colors',
             'lr': 0.01
            }
        ]
    return torch.optim.Adam(param_groups)

def initialize_recon_optimizer_3b(params):
    """
    Optimize marble color.
    """
    param_groups = \
        [
            {'params': [params['surface_rgb']],
             'name': 'surface_rgb',
             'lr': 0.01
            }
        ]
    return torch.optim.Adam(param_groups)

def initialize_track_optimizer(params, scene_radius):
    """
    Position + Rotation optimizer
    """

    param_groups = \
        [
            {'params': [params['means3D']],
             'name': 'means3D',
             'lr': 0.00016 * scene_radius
            },
            {'params': [params['unnorm_rotations']],
             'name': 'unnorm_rotations',
             'lr': 0.001
            }
        ]

    return torch.optim.Adam(param_groups)

def get_recon_loss(params, curr_data):
    """
    L1/SSIM Loss on base gaussians.
    Free parameters defined in recon_optimizer_1
    """

    obj_img = curr_data['seg'].unsqueeze(0).expand(3, *curr_data['seg'].shape)
    rendervar = {
        'means3D': params['means3D'],
        'colors_precomp': params['rgb_colors'],
        'rotations': torch.nn.functional.normalize(params['unnorm_rotations']),
        'opacities': torch.sigmoid(params['logit_opacities']),
        'scales': torch.tile(torch.exp(params['log_scale']), (1, 3)),
        'means2D': torch.zeros_like(params['means3D'], requires_grad=True, device="cuda") + 0
    }
    im, radius, depth = Renderer(raster_settings=curr_data['cam'])(**rendervar)

    near_z = curr_data['cam_near_plane']
    far_z = curr_data['cam_far_plane']
    norm_depth = (depth - near_z) / (far_z - near_z)
    loss = 0.5 * weighted_l2_loss_v2(im, obj_img) + \
            0.5 * (1.0 - calc_ssim(im, obj_img))

    return loss, im, norm_depth

def get_recon_loss_2(params, curr_data):
    """
    L1/SSIM Loss on base gaussians.
    Free parameters defined in recon_optimizer_2
    """

    obj_img = curr_data['seg'].unsqueeze(0).expand(3, *curr_data['seg'].shape)
    rendervar = {
        'means3D': params['means3D'],
        'colors_precomp': params['rgb_colors'],
        'rotations': torch.nn.functional.normalize(params['unnorm_rotations']),
        'opacities': torch.sigmoid(params['logit_opacities']),
        'scales': torch.exp(params['log_scales']),
        'means2D': torch.zeros_like(params['means3D'], requires_grad=True, device="cuda") + 0
    }
    im, radius, depth = Renderer(raster_settings=curr_data['cam'])(**rendervar)

    near_z = curr_data['cam_near_plane']
    far_z = curr_data['cam_far_plane']
    norm_depth = (depth - near_z) / (far_z - near_z)
    loss = 0.5 * weighted_l2_loss_v2(im, obj_img) + \
            0.5 * (1.0 - calc_ssim(im, obj_img))

    return loss, im, norm_depth

def get_recon_loss_3a(params, curr_data):
    """
    Temp loss for testing.
    Supervize on masked curr im.
    """
    obj_img = curr_data['im'] * curr_data['seg']

    rendervar = {
        'means3D': params['means3D'],
        'colors_precomp': params['rgb_colors'],
        'rotations': torch.nn.functional.normalize(params['unnorm_rotations']),
        'opacities': torch.sigmoid(params['logit_opacities']),
        'scales': torch.exp(params['log_scales']),
        'means2D': torch.zeros_like(params['means3D'], requires_grad=True, device="cuda") + 0
    }
    im, radius, depth = Renderer(raster_settings=curr_data['cam'])(**rendervar)

    near_z = curr_data['cam_near_plane']
    far_z = curr_data['cam_far_plane']
    norm_depth = (depth - near_z) / (far_z - near_z)
    loss = 0.5 * weighted_l2_loss_v2(im, obj_img) + \
            0.5 * (1.0 - calc_ssim(im, obj_img))

    return loss, im, norm_depth

def get_recon_loss_3b(params, curr_data):
    """
    L1/SSIM Loss on surface gaussians.
    Free parameters defined in recon_optimizer_3
    """

    if not (
    'surface_means3D' in params.keys() or
    'surface_rgb' in params.keys() or
    'surface_log_scale' in params.keys()
    ):
        raise ValueError('Missing surface buffer(s) in 3rd stage reconstruction.')

    obj_img = curr_data['im'] * curr_data['seg']

    rendervar = {
        'means3D': params['surface_means3D'],
        'colors_precomp': params['surface_rgb'],
        'rotations': torch.nn.functional.normalize(params['surface_unnorm_rotations']),
        'opacities': torch.sigmoid(params['surface_logit_opacities']),
        'scales': torch.tile(torch.exp(params['surface_log_scale']), (1, 3)),
        'means2D': torch.zeros_like(params['surface_means3D'], requires_grad=True, device="cuda") + 0
    }
    im, radius, depth = Renderer(raster_settings=curr_data['cam'])(**rendervar)

    near_z = curr_data['cam_near_plane']
    far_z = curr_data['cam_far_plane']
    norm_depth = (depth - near_z) / (far_z - near_z)
    loss = 0.5 * weighted_l2_loss_v2(im, obj_img) + \
            0.5 * (1.0 - calc_ssim(im, obj_img))

    return loss, im, norm_depth

def get_tracking_loss(curr_params, curr_data, local_gaussian_pts):
    """
    L1/SSIM Loss on surface gaussians.
    Free parameters defined in track_optimzer
    """
    log_render_imgs = {}

    curr_surface_means = []
    for g_id, local_pts in local_gaussian_pts.items():
        # Will try Rotate -> Scale -> Translate
        local_quat = curr_params['unnorm_rotations'][g_id]
        local_scale = torch.exp(curr_params['log_scales'])[g_id]
        local_displacement = curr_params['means3D'][g_id]

        local_pts = quaternion_rotate_points(local_quat, local_pts)
        local_pts = (local_pts * local_scale) + local_displacement
        curr_surface_means.append(local_pts)
    curr_params['surface_means3D'] = torch.cat(curr_surface_means, dim=0)

    # Base Color Loss (implicitly segmented)
    base_color_loss = 0
    for d in curr_data:
        obj_img = d['im'] * d['seg']

        rendervar = {
            'means3D': curr_params['means3D'],
            'colors_precomp': curr_params['rgb_colors'],
            'rotations': torch.nn.functional.normalize(curr_params['unnorm_rotations']),
            'opacities': torch.sigmoid(curr_params['logit_opacities']),
            'scales': torch.exp(curr_params['log_scales']),
            'means2D': torch.zeros_like(curr_params['means3D'], requires_grad=True, device="cuda") + 0
        }
        im, radius, depth = Renderer(raster_settings=d['cam'])(**rendervar)
        near_z = d['cam_near_plane']
        far_z = d['cam_far_plane']
        base_color_loss += 0.5 * weighted_l2_loss_v2(im, obj_img) + \
                           0.5 * (1.0 - calc_ssim(im, obj_img))

        log_render_imgs[d["id"]] = im

    # Texture Loss (implicitly segmented)
    texture_loss = 0

    # Defining Blur transform for regularization
    kernel_size = 5
    sigma = 1.0
    blur_transform = transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)

    for d in curr_data:
        obj_img = d['im'] * d['seg']
        rendervar = {
            'means3D': curr_params['surface_means3D'],
            'colors_precomp': curr_params['surface_rgb'],
            'rotations': torch.nn.functional.normalize(curr_params['surface_unnorm_rotations']),
            'opacities': torch.sigmoid(curr_params['surface_logit_opacities']),
            'scales': torch.tile(torch.exp(curr_params['surface_log_scale']), (1, 3)),
            'means2D': torch.zeros_like(curr_params['surface_means3D'], requires_grad=True, device="cuda") + 0
        }
        im, radius, depth = Renderer(raster_settings=d['cam'])(**rendervar)

        near_z = d['cam_near_plane']
        far_z = d['cam_far_plane']
        texture_loss += 0.5 * weighted_l2_loss_v2(im, obj_img) + \
                        0.5 * (1.0 - calc_ssim(im, obj_img))

        im_low_pass = blur_transform(im)
        obj_img_lowpass = blur_transform(obj_img)
        lowpass_loss = 0.5 * weighted_l2_loss_v2(im_low_pass, obj_img_lowpass) + \
                        0.5 * (1.0 - calc_ssim(im_low_pass, obj_img_lowpass))
        texture_loss += lowpass_loss

        another_low_pass = blur_transform(im_low_pass)
        another_obj_img_lowpass = blur_transform(obj_img_lowpass)
        another_low_pass_loss = 0.5 * weighted_l2_loss_v2(another_low_pass, another_obj_img_lowpass) + \
                                0.5 * (1.0 - calc_ssim(another_low_pass, another_obj_img_lowpass))
        texture_loss += another_low_pass_loss

        log_render_imgs[d["id"]] = im

    loss = seg_loss + base_color_loss + texture_loss

    wandb.log({f"seg_loss": seg_loss})
    wandb.log({f"base_color_loss": base_color_loss})
    wandb.log({f"texture_loss": texture_loss})

    return loss, log_render_imgs
# -------

# -------
# Training Recipe 
def reconstruct_timestep(t, dataset_path, pc_path, metadata, camera_subset, recon_iters):
    """
    Runs reconstruction optimization loop
    """

    # Fetch data at current time
    dataset = get_dataset(t, dataset_path, metadata, camera_subset)
    params, scene_radius = initialize_params(t, dataset_path, pc_path, metadata, downsample_lvl=0)

    # Stage 1 Reconstruction
    progress_bar = tqdm(range(recon_iters), desc=f"timestep {t}")
    optimizer = initialize_recon_optimizer(params, scene_radius)
    for i in progress_bar:
        renders = []
        for d in dataset:
            loss, im, _ = get_recon_loss(params, d)
            loss.backward()
        wandb.log({f"recon_loss/lvl_{lvl}": loss})

        with torch.no_grad():
            if i % 10 == 0:
                psnr = calc_psnr(im, dataset[0]['im']).mean()
                progress_bar.set_postfix({"Cam 1, PSNR": f"{psnr:.{7}f}"})

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    # Log 1st stage reconstruction point cloud
    positions = params['means3D'].detach().cpu().numpy()
    colors = params['rgb_colors'].detach().cpu().numpy()
    colors = np.clip(colors, 0, 1) * 255
    log_point_cloud = np.concatenate((positions, colors), axis=1)
    wandb.log({f"1st_stage_reconstruction/lvl_{lvl}": wandb.Object3D(log_point_cloud)}, commit=True)

    # Stage 2 Reconstruction
    log_scale = params['log_scale'].detach().cpu().contiguous().numpy()
    del params['log_scale']
    log_scales = np.tile(log_scale, (1, 3))
    params['log_scales'] = torch.nn.Parameter(torch.tensor(log_scales).cuda().float().contiguous().requires_grad_(True))

    progress_bar = tqdm(range(recon_iters), desc=f"timestep {t} stage 2")
    optimizer_2 = initialize_recon_optimizer_2(params, scene_radius)
    for i in progress_bar:
        for d in dataset:
            loss, im, _ = get_recon_loss_2(params, d)
            loss.backward()
        wandb.log({f"recon_loss/lvl_{lvl}": loss})

        with torch.no_grad():
            if i % 10 == 0:
                psnr = calc_psnr(im, dataset[0]['im']).mean()
                progress_bar.set_postfix({"Cam 1, PSNR": f"{psnr:.{7}f}"})
        optimizer_2.step()
        optimizer_2.zero_grad(set_to_none=True)

    # Log 2nd stage reconstruction point cloud
    positions = params['means3D'].detach().cpu().numpy()
    colors = params['rgb_colors'].detach().cpu().numpy()
    colors = np.clip(colors, 0, 1) * 255
    log_point_cloud = np.concatenate((positions, colors), axis=1)
    wandb.log({f"2nd_stage_reconstruction/lvl_{lvl}": wandb.Object3D(log_point_cloud)}, commit=True)

    # Stage 3 Reconstruction Prep: Filter the gaussians
    # Here, reconstructing parameter buffer following filtering operations.
    means = params['means3D']
    semi_axes = torch.exp(params['log_scales']) * 2
    keep_indices = filter_ellipsoids_by_overlap(
        ellipsoid_centers=means,
        ellipsoid_scales=semi_axes
    )
    params = {k: torch.nn.Parameter(v[keep_indices].detach().clone().requires_grad_(True))
                for k, v in params.items()}

    # Stage 3 Reconstruction Prep: Initalize Skin Gaussians
    means = params['means3D']
    semi_axes = torch.exp(params['log_scales']) * 2
    surface_map = sample_ellipsoid_union_surface(ellipsoid_centers=means, ellipsoid_scales=semi_axes)
    num_samples = sum(len(pts) for pts in surface_map.values())

    surface_pts = {}
    surface_means_list = [pts.detach().cpu().numpy() for pts in surface_map.values()]
    surface_means = np.concatenate(surface_means_list, axis=0)
    surface_pts['surface_means3D'] = surface_means
    surface_pts['surface_rgb'] = np.ones((num_samples, 3))
    surface_pts['surface_unnorm_rotations'] = np.tile([1, 0, 0, 0], (num_samples, 1))
    surface_pts['surface_logit_opacities'] = np.ones((num_samples, 1))

    sq_dist, _ = o3d_knn(surface_means, 3)
    mean3_sq_dist = sq_dist.mean(-1).clip(min=0.0000001)  # NOTE: Original size.
    surface_pts['surface_log_scale'] = np.log(np.sqrt(mean3_sq_dist))[..., None]

    params.update({k: torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True))
              for k, v in surface_pts.items()})
    
    # Stage 3 Reconstruction: Optimizing color of base gaussians
    progress_bar = tqdm(range(recon_iters), desc=f"timestep {t} stage 3a")
    optimizer_3a = initialize_recon_optimizer_3a(params)
    for i in progress_bar:
        for d in dataset:
            loss, im, _ = get_recon_loss_3a(params, d)
            loss.backward()
        wandb.log({f"recon_loss/stage_3a": loss})

        with torch.no_grad():
            if i % 10 == 0:
                psnr = calc_psnr(im, dataset[0]['im']).mean()
                progress_bar.set_postfix({"Cam 1, PSNR": f"{psnr:.{7}f}"})
        optimizer_3a.step()
        optimizer_3a.zero_grad(set_to_none=True)

    # Stage 3 Reconstruction: Optimizing color of skin gaussians
    progress_bar = tqdm(range(recon_iters), desc=f"timestep {t} stage 3b")
    optimizer_3b = initialize_recon_optimizer_3b(params)
    for i in progress_bar:
        for d in dataset:
            loss, im, _ = get_recon_loss_3b(params, d)
            loss.backward()
        wandb.log({f"recon_loss/stage_3": loss})

        with torch.no_grad():
            if i % 10 == 0:
                psnr = calc_psnr(im, dataset[0]['im']).mean()
                progress_bar.set_postfix({"Cam 1, PSNR": f"{psnr:.{7}f}"})
        optimizer_3b.step()
        optimizer_3b.zero_grad(set_to_none=True)

    # Log 3rd stage reconstruction point cloud
    positions = params['surface_means3D'].detach().cpu().numpy()
    colors = params['surface_rgb'].detach().cpu().numpy()
    colors = np.clip(colors, 0, 1) * 255
    log_point_cloud = np.concatenate((positions, colors), axis=1)
    wandb.log({"final_reconstruction": wandb.Object3D(log_point_cloud)}, commit=True)

    # Parent surface gaussians to nearest volume
    means = params['means3D']
    semi_axes = torch.exp(params['log_scales'])  # NOTE: Don't x2 here. We will apply regular forward scaling later in tracking.
    quat = params['unnorm_rotations']
    local_surface_map = {}
    g_mats = build_homogeneous_matrices(
        means=means,
        semi_axes=semi_axes,
        quaternions=quat,
    )
    for g_id, global_pts in surface_map.items():
        to_local = torch.inverse(g_mats[g_id])
        local_pts = torch.matmul(global_pts, to_local[:3, :3].T) + to_local[:3, 3]
        local_surface_map[g_id] = local_pts.detach()

    return params, local_surface_map, scene_radius

def track_gaussians(t, curr_params, curr_dataset, scene_radius, local_gaussian_pts, track_iters):
    """
    Runs tracking optimization loop.
    """

    # Tracking Loop
    progress_bar = tqdm(range(track_iters), desc=f"tracking to timestep {t}")
    log_render_imgs = {}
    optimizer = initialize_track_optimizer(curr_params, scene_radius)

    for i in progress_bar:
        # Using tracking loss
        loss, log_render_imgs = get_tracking_loss(
            curr_params, curr_dataset, local_gaussian_pts,
            left_overlap, right_overlap
        )
        loss.backward(retain_graph=True)
        wandb.log({"tracking_loss": loss})

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    # Final result logging
    if t % 1 == 0:   # Swap to 100 for the full run.
        for d in curr_dataset:
            log_gt = (d['im'].detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            log_mask = (d['seg'].detach().cpu().numpy() * 255).astype(np.uint8)
            log_masked_gt = ((d['im'] * d['seg']).detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)

            img = torch.clamp(log_render_imgs[d["id"]], 0, 1)
            log_render = (img.detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)

            image_log_dict = {
                f"track_cam_{d['id']}/images": [
                    wandb.Image(log_gt, caption="ground_truth"),
                    wandb.Image(log_mask, caption="mask"),
                    wandb.Image(log_masked_gt, caption="masked_ground_truth"),
                    wandb.Image(log_render, caption="render"),
                ]
            }
            wandb.log(image_log_dict, commit=False)

        # Log final point cloud
        positions = curr_params['means3D'].detach().cpu().numpy()
        colors = curr_params['rgb_colors'].detach().cpu().numpy() * 255
        log_point_cloud = np.concatenate((positions, colors), axis=1)
        wandb.log({"track_point_cloud": wandb.Object3D(log_point_cloud)}, commit=True)

    return curr_params

def train(
    dataset_path: str, 
    metadata_path: str, 
    pc_path: str, 
    camera_subset: list[str],
    start_time: int, 
    num_timesteps: int, 
    stride: int, 
    output_timeseries_path: str
):
    """
    Edited to accept custom dataset directory.
    """

    wandb.init(project='3dgs-sweep')
    wandb.define_metric("recon_iter")
    wandb.define_metric("track_iter")

    # Optimization Hyperparameters
    RECONSTRUCTION_ITERS = 200
    # RECONSTRUCTION_ITERS = 50   # For quick testing
    TRACKING_ITERS = 100

    # Central Timeseries Buffer
    gaussian_timeseries = []

    # Initial Reconstruction
    metadata = read_config_yaml(metadata_path)
    curr_params, local_gaussian_pts, scene_radius = reconstruct_timestep(start_time,
                                                                        dataset_path,
                                                                        pc_path,
                                                                        metadata,
                                                                        camera_subset,
                                                                        recon_iters=RECONSTRUCTION_ITERS)
    gaussian_timeseries.append(params2cpu(curr_params))

    # Tracking Loop
    for t in range(start_time + 1, start_time + num_timesteps, stride):
        curr_dataset = get_dataset(t, dataset_path, metadata, camera_subset)
        curr_params = \
            track_gaussians(t, curr_params, curr_dataset, scene_radius,
                            local_gaussian_pts, track_iters=TRACKING_ITERS)
        gaussian_timeseries.append(params2cpu(curr_params))

    # End of tracking, save pc timeseries
    save_params(gaussian_timeseries, output_timeseries_path)

# -------

if __name__ == "__main__":
    # Edit these at runtime
    # I am doing a parallel capsule run which is more flexible than parallel pipeline run. 
    DATASET_PATH = '/root/capsule/data/rat7m-3dgs'
    camera_subset = ['1', '2', '3', '4', '5', '6']

    # TODO: 
    # Read these from an app panel
    start_time = 0
    num_timesteps = 50
    stride = 1
    metadata_path = '...'
    pc_path = '...'
    OUTPUT_TIMESERIES_PATH = "/results/tmp.npz"

    # Start training
    start_time = time.time()
    train(
        DATASET_PATH,
        metadata_path, 
        pc_path, 
        camera_subset,
        start_time,
        num_timesteps, 
        stride, 
        output_timeseries_path
    )
    print(f'Took {time.time() - start_time}')

    # TODO: Export to S3
    # ...