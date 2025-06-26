from pathlib import Path
from PIL import Image
import shutil
import numpy as np

from balloon_animal.precompute import MolmoInference, SAM2Inference, monte_carlo_sample_visible_points
from balloon_animal.precompute_utils import get_ordered_paths, visualize_point_prediction, create_video_from_images, visualize_scene

class Skeleton:
    def __init__(self, points=None, edges=None):
        self.points = points
        self.edges = edges

def generate_validation(
    results_dir: str, 
    initial_detections: dict,
    video_extrinsic_matrices: list[np.ndarray],
    point_cloud: np.ndarray,
    skeleton: Skeleton
):
    """
    Parameters:
        results_dir (str): Output directory of generate_precomputed_assets()
        initial_detections (dict): Detections from generate_precomputed_assets()
        video_extrinsic_matrices (list[np.ndarray]): Calibration from source dataset. 
        point_cloud (np.ndarray): Point cloud from generate_precomputed_assets()
        skeleton (Skeleton): Graph GT inital pose. Can be blank.

    Generates preview of precomputed assets in the following format:     
    detection:
        |-- Dir_1.jpg
        |-- Dir_2.jpg
    segmentation:
        |-- dir_1.mp4
        |-- dir_2.mp4
    scene_0.html
    """

    results_dir = Path(results_dir)
    val_path = results_dir / 'validation'
    val_path.mkdir(exist_ok=True, parents=True)
    det_path = val_path / 'detection'
    det_path.mkdir(exist_ok=True, parents=True)
    seg_path = val_path / 'segmentation'
    seg_path.mkdir(exist_ok=True, parents=True)

    print('Generating Detection Preview...')
    img_results = results_dir / 'img'
    for vid_dir in img_results.iterdir():
        vid_name = str(Path(vid_dir).name)

        first_image = str(Path(vid_dir) / '0.jpg')
        detections = initial_detections[vid_name]
        out_path = det_path / f'{vid_name}.jpg'

        # visualize_detections(first_image, detections, out_path)
        visualize_point_prediction(first_image, detections, out_path)
    
    print('Generating Segmentation Preview...')
    seg_results = results_dir / 'seg'
    for vid_dir in seg_results.iterdir():
        vid_name = str(Path(vid_dir).name)

        ordered_paths = get_ordered_paths(vid_dir)
        out_path = seg_path / f'{vid_name}.mp4'

        create_video_from_images(ordered_paths, out_path)

    print('Generating Scene Preview...')
    visualize_scene(video_extrinsic_matrices, 
                    point_cloud,
                    output_path=str(val_path / 'scene.html'),
                    skeleton_points=skeleton.points,
                    skeleton_edges=skeleton.edges)

def generate_precomputed_assets(
    video_dataset: list[str], 
    video_extrinsic_matrices: list[np.ndarray], 
    video_intrinsic_matrices: list[np.ndarray], 
    output_dir: str, 
    prompt: str,
    pc_size: int = 5000,
    pc_visibility: float = 0.8,
    include_validation: bool = True,
    skeleton: Skeleton = Skeleton()
) -> None:
    """
    Generates precomputed assets in the following format: 
    img:
        |-- Dir_1
            |-- 0.jpg
            |-- ...
        |-- Dir_2
    seg:
        |-- Dir_1
            |-- 0.jpg
            |-- ...
        |-- Dir_2
    pcs:
        |-- Dir_1
            |-- 0.ply
            |-- ...
        |-- Dir_2
    validation: 
        (optional, calls generate_validation)

    Images, segmentation masks, point clouds are renamed {number}.{ext} in ascending order.
    Directory names are preserved. 

    Inputs:
        video_dataset: list of synced image sequence directory paths
        output_dir: output directory to place assets
        prompt: prompt to open vocabulary model
    """
    
    # Create output directory format
    img_dir = (Path(output_dir) / 'img')
    img_dir.mkdir(parents=True, exist_ok=True)
    seg_dir = (Path(output_dir) / 'seg')
    seg_dir.mkdir(parents=True, exist_ok=True)
    pc_dir = (Path(output_dir) / 'pcs')
    pc_dir.mkdir(parents=True, exist_ok=True)  # Will make this a directory even if there is only one pc.

    # Copy img contents over, renaming in ascending number order. 
    print('Copying images to output directory...')
    for vid_dir in video_dataset:
        vid_name = str(Path(vid_dir).name)
        curr_img_dir = (Path(img_dir) / vid_name)
        curr_img_dir.mkdir(parents=True, exist_ok=True)

        print(f'Copying {vid_dir}...')
        ordered_img_paths = get_ordered_paths(str(vid_dir))
        for i, img_path in enumerate(ordered_img_paths):
            source_path = str(img_path)
            dest_path = str(curr_img_dir / f'{i}.jpg')
            shutil.copy2(source_path, dest_path)

    # Alternative OmDet, faster, but more imprecise. 
    # print('Running initial detection...')
    # initial_detections = {}
    # with OmDetInference(cache_dir="/scratch/hub") as model: 
    #     for vid_dir in video_dataset:
    #         vid_name = str(Path(vid_dir).name)
    #         first_image = str(get_ordered_paths(vid_dir)[0])

    #         print(f'Detecting on {vid_name}...')
    #         detections = model(first_image, [prompt])
    #         initial_detections[vid_name] = detections

    print('Running initial detection...')
    initial_detections = {}
    with MolmoInference(cache_dir="/scratch/hub") as molmo:
        for vid_dir in video_dataset:
            vid_name = str(Path(vid_dir).name)
            first_image = str(get_ordered_paths(vid_dir)[0])

            print(f'Pointing on {vid_name}...')
            coordinates = molmo(first_image, prompt)
            initial_detections[vid_name] = coordinates

    # Open Vocabulary Detection -> Segmentation
    print('Running segmentation...')
    with SAM2Inference(option='video', cache_dir="/scratch/hub") as model:
        for vid_dir in video_dataset:
            vid_name = str(Path(vid_dir).name)
            # input_box = initial_detections[vid_name][0]['box']
            input_coords = initial_detections[vid_name]

            print(f'Segmenting {vid_name}...')
            model.process_directory(image_dir=vid_dir, 
                                    output_dir=seg_dir / vid_name,
                                    input_points=[input_coords])

    # Segmentation -> Point Cloud
    # print('Running Point Cloud Initalization...')
    # num_images = len(list(video_dataset[0].iterdir()))
    # for i in range(num_images): 
    #     seg_masks = [
    #         np.array(Image.open(vid_dir / f'{i}.jpg')) / 255.
    #         for vid_dir in seg_dir.iterdir()
    #     ]

    #     point_cloud = monte_carlo_sample_visible_points(
    #         video_extrinsic_matrices,
    #         video_intrinsic_matrices,
    #         seg_masks,
    #         target_points=pc_size,
    #         visibility_percentage=pc_visibility
    #     )
    #     point_cloud_path = str(pc_dir / f'{i}.npy')
    #     np.save(point_cloud_path, point_cloud)
    # ^ If multiple point cloud initalization is needed

    # Segmentation -> Point Cloud Init
    print('Running Point Cloud Initalization...')
    initial_seg_masks = []
    for cam_dir in get_ordered_paths(seg_dir):
        first_img_path = str(cam_dir / '0.jpg')
        seg = np.array(Image.open(first_img_path)) / 255.
        initial_seg_masks.append(seg)
    point_cloud = monte_carlo_sample_visible_points(
        video_extrinsic_matrices,
        video_intrinsic_matrices,
        initial_seg_masks,
        target_points=pc_size,
        visibility_percentage=pc_visibility
    )
    point_cloud_path = str(pc_dir / '0.npy')
    np.save(point_cloud_path, point_cloud)

    if include_validation:
        point_cloud = np.load(point_cloud_path)
        generate_validation(output_dir, 
                            initial_detections, 
                            video_extrinsic_matrices,
                            point_cloud,
                            skeleton)
