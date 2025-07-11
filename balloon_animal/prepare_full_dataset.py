from pathlib import Path
from PIL import Image
import shutil
import numpy as np
import yaml

from balloon_animal.precompute import MolmoInference, SAM2Inference
from balloon_animal.precompute_utils import get_ordered_paths, visualize_point_prediction, create_video_from_images

def format_camera_yaml(
    dataset_name: str, 
    camera_width: int,
    camera_height: int,
    num_cameras: int,
    num_frames: int,
    extrinsic_matrices: dict[str, np.ndarray],
    intrinsic_matrices: dict[str, np.ndarray],
    output_yaml_path: str
):
    """
    Format camera configuration data into YAML file using the yaml library.
    
    Args:
        dataset_name (str): Name of the dataset
        camera_width (int): Width of camera images
        camera_height (int): Height of camera images
        num_cameras (int): Number of cameras
        num_frames (int): Number of frames
        extrinsic_matrices (list[np.ndarray]): List of 4x4 extrinsic matrices
        intrinsic_matrices (list[np.ndarray]): List of 3x3 intrinsic matrices
        output_yaml_path (str): Path to output YAML file
    """
    
    # Create the data structure using regular dict
    data = {
        'dataset_name': dataset_name,
        'camera_height': camera_height,
        'camera_width': camera_width,
        'num_cameras': num_cameras,
        'num_frames': num_frames,
        'extrinsic_matrices': {},
        'intrinsic_matrices': {}
    }
    
    for name, matrix in extrinsic_matrices.items():
        data['extrinsic_matrices'][name] = matrix.tolist()
    for name, matrix in intrinsic_matrices.items():
        data['intrinsic_matrices'][name] = matrix.tolist()

    # Write to YAML file using safe_dump
    with open(output_yaml_path, 'w') as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)

def generate_validation(
    results_dir: str,
    initial_detections: dict,
):
    """
    Parameters:
        results_dir (str): Output directory of generate_precomputed_assets()
        initial_detections (dict): Detections from generate_precomputed_assets()

    Generates preview of precomputed assets in the following format:
    detection:
        |-- Dir_1.jpg
        |-- Dir_2.jpg
    segmentation:
        |-- dir_1.mp4
        |-- dir_2.mp4
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
    
def generate_precomputed_assets(
    dataset_name: str,
    video_dataset: list[str],
    video_extrinsic_matrices: dict[str, np.ndarray],
    video_intrinsic_matrices: dict[str, np.ndarray],
    output_dir: str,
    prompt: str,
    batch_size: int = 10,  # Low number to prevent CUDA OOM. 
    batch_dir: str = '/results/tmp',
    include_validation: bool = True
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
    validation:
        (optional, calls generate_validation)
    metadata.yml

    Images, segmentation masks, point clouds are renamed {number}.{ext} in ascending order.
    Directory names are preserved.

    Inputs:
        dataset_name: name of the dataset for future reference
        video_dataset: list of synced image sequence directory paths
        output_dir: output directory to place assets
        prompt: prompt to open vocabulary model
    """

    # Create output directory format
    img_dir = (Path(output_dir) / 'img')
    img_dir.mkdir(parents=True, exist_ok=True)
    seg_dir = (Path(output_dir) / 'seg')
    seg_dir.mkdir(parents=True, exist_ok=True)

    # Create metadata file
    img_seq = get_ordered_paths(video_dataset[0])
    camera_width, camera_height = Image.open(img_seq[0]).size
    num_cameras = len(video_dataset)
    num_frames = len(img_seq)
    format_camera_yaml(
        dataset_name=dataset_name, 
        camera_width=camera_width,
        camera_height=camera_height,
        num_cameras=num_cameras,
        num_frames=num_frames,
        extrinsic_matrices=video_extrinsic_matrices,
        intrinsic_matrices=video_intrinsic_matrices,
        output_yaml_path=str(Path(output_dir) / 'metadata.yml')
    )

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
                                    input_points=[input_coords],
                                    batch_size=batch_size,
                                    batch_dir=batch_dir)

    if include_validation:
        generate_validation(output_dir,
                            initial_detections)
