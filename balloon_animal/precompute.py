"""
Precomputed inputs to tracking: 
- Detection
- Segmentation
- Point Clouds
"""

import os
from pathlib import Path
from PIL import Image
import shutil
from typing import Union, List, Optional

import numpy as np
import torch

from balloon_animal.precompute_utils import get_ordered_paths

def configure_hf_cache(cache_dir):
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        os.environ['HF_HOME'] = cache_dir
        os.environ['HUGGINGFACE_HUB_CACHE'] = cache_dir

class OmDetInference:
    """
    A class for performing inference with the OmDet Turbo object detection model.
    Provides functionality to mount/unmount the model from GPU as needed.
    """
    
    def __init__(self, model_name="omlab/omdet-turbo-swin-tiny-hf", cache_dir=None, device=None):
        """
        Initialize the OmDet inference class.
        
        Args:
            model_name (str): HuggingFace model identifier
            cache_dir (str, optional): Directory to store model files
            device (str, optional): Device to load the model on ('cuda', 'cpu', or None for auto-detection)
        """
        configure_hf_cache(cache_dir)  # Set cache before HF Import
        from transformers import AutoProcessor, OmDetTurboForObjectDetection
        self.AutoProcessor = AutoProcessor
        self.OmDetTurboForObjectDetection = OmDetTurboForObjectDetection
        
        # Determine device
        self.device = device
        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Initialize model and processor
        self.model_name = model_name
        self.processor = self.AutoProcessor.from_pretrained(self.model_name)
        
        # Start with model not loaded to save memory
        self.model = None
        self.is_mounted = False
    
    def mount(self):
        """
        Load the model to the specified device (GPU/CPU).
        """
        if not self.is_mounted:
            self.model = self.OmDetTurboForObjectDetection.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            self.is_mounted = True
            
    def unmount(self):
        """
        Unload the model from GPU/memory.
        """
        if self.is_mounted:
            if self.device == 'cuda':
                self.model.to('cpu')
                torch.cuda.empty_cache()
            self.model = None
            self.is_mounted = False
    
    def __call__(self, image, prompt, score_threshold=0.3, nms_threshold=0.3):
        """
        Perform inference on an image with a given text prompt.
        
        Args:
            image: PIL Image or path to image file
            prompt (str or list): Class name(s) to detect
            score_threshold (float): Confidence threshold for detections
            nms_threshold (float): Non-maximum suppression threshold
            
        Returns:
            list: List of dictionaries containing bounding boxes, scores, and class names
        """
        # Make sure model is mounted
        if not self.is_mounted:
            self.mount()
        
        # Convert string prompt to list if necessary
        if isinstance(prompt, str):
            classes = [prompt]
        else:
            classes = prompt
            
        # Load image if path is provided
        if isinstance(image, str):
            image = Image.open(image)
        
        # Process image and prepare for model
        inputs = self.processor(image, text=classes, return_tensors="pt")
        
        # Move inputs to the same device as model
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Post-process the outputs
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            classes=classes,
            target_sizes=[image.size[::-1]],  # Convert (width, height) to (height, width)
            score_threshold=score_threshold,
            nms_threshold=nms_threshold,
        )[0]
        
        # Format results
        detections = []
        for score, class_name, box in zip(
            results["scores"], results["classes"], results["boxes"]
        ):
            detections.append({
                "box": box.tolist(),  # [x1, y1, x2, y2]
                "score": score.item(),
                "class": class_name
            })
            
        return detections
    
    def __enter__(self):
        """
        Context manager entry - ensures model is mounted.
        """
        self.mount()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit - unmounts model from GPU.
        """
        self.unmount()

class MolmoInference:
    """
    A class for performing inference with the Molmo vision-language model.
    Provides functionality to mount/unmount the model from GPU as needed.
    """
    
    def __init__(self, model_name="allenai/Molmo-7B-D-0924", cache_dir=None, device=None):
        """
        Initialize the Molmo inference class.
        
        Args:
            model_name (str): HuggingFace model identifier for Molmo
            cache_dir (str, optional): Directory to store model files
            device (str, optional): Device to load the model on ('cuda', 'cpu', or None for auto-detection)
        """
        configure_hf_cache(cache_dir)  # Set cache before HF Import
        from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
        self.AutoModelForCausalLM = AutoModelForCausalLM
        self.AutoProcessor = AutoProcessor
        self.GenerationConfig = GenerationConfig
        
        # Determine device
        self.device = device
        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Initialize model name and processor
        self.model_name = model_name
        self.processor = self.AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        # Start with model not loaded to save memory
        self.model = None
        self.is_mounted = False
    
    def mount(self):
        """
        Load the model to the specified device (GPU/CPU).
        """
        if not self.is_mounted:
            # Load model with device_map='auto' for automatic offloading
            self.model = self.AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype='auto',
                device_map='auto'  # Let transformers handle device placement
            )
            # self.model = self.model.to(dtype=torch.bfloat16)
            self.model.eval()  # Set to evaluation mode
            self.is_mounted = True
            print(f"Molmo model mounted with automatic device placement")
            
    def unmount(self):
        """
        Unload the model from GPU/memory.
        """
        if self.is_mounted:
            # When using device_map='auto', we can't manually move the model
            # Just delete the reference and clear cache
            del self.model
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.is_mounted = False
            print("Molmo model unmounted")
    
    def __call__(self, image: Union[str, Image.Image], subject: str) -> Optional[tuple]:
        """
        Point to a subject in the image and return the coordinates.
        
        Args:
            image: PIL Image or path to image file
            subject (str): Subject/object to point to in the image
            
        Returns:
            tuple: (x, y) coordinates of the pointed location, or None if not found
        """
        # Make sure model is mounted
        if not self.is_mounted:
            self.mount()
        
        # Load image if path is provided
        if isinstance(image, str):
            image = Image.open(image)
        
        # Ensure image is in RGB format
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Format the pointing prompt
        prompt = f"Point to {subject} in the image"
        
        # Process inputs
        inputs = self.processor.process(
            images=[image],
            text=prompt
        )
        
        # Move inputs to the correct device and make a batch of size 1
        inputs = {k: v.to(self.model.device).unsqueeze(0) if isinstance(v, torch.Tensor) else v 
                 for k, v in inputs.items()}
        
        # Generate response with lower temperature for more consistent pointing
        with torch.no_grad():
            # with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            output = self.model.generate_from_batch(
                inputs,
                self.GenerationConfig(
                    max_new_tokens=100,  # Shorter since we just need coordinates
                    stop_strings=["<|endoftext|>"]
                ),
                tokenizer=self.processor.tokenizer
            )
        
        # Decode the generated text
        generated_tokens = output[0, inputs['input_ids'].size(1):]
        response = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # Parse coordinates from the response
        coordinates = self._parse_coordinates(response.strip())

        print('Predicted coordinates', coordinates)

        # Convert to image scale
        coordinates = self._convert_molmo_to_pixel_coords(coordinates, image.size)

        return coordinates
    
    def _parse_coordinates(self, response: str) -> Optional[tuple]:
        """
        Parse coordinates from Molmo's pointing response.
        
        Args:
            response (str): Raw response from Molmo
            
        Returns:
            tuple: (x, y) coordinates, or None if parsing fails
        """
        import re
        
        # Common patterns Molmo might use for coordinates
        patterns = [
            r'<point x="([0-9.]+)" y="([0-9.]+)"[^>]*>',  # <point x="28.0" y="77.3" alt="...">
            r'<click>([0-9.]+),\s*([0-9.]+)</click>',     # <click>x, y</click>
            r'<point x="([0-9.]+)" y="([0-9.]+)"/>',      # <point x="x" y="y"/>
            r'\(([0-9.]+),\s*([0-9.]+)\)',                # (x, y)
            r'x[=:]?\s*([0-9.]+).*?y[=:]?\s*([0-9.]+)',   # x=123, y=456 or similar
            r'([0-9.]+),\s*([0-9.]+)',                    # Simple x, y format
        ]
        
        for pattern in patterns:
            matches = re.search(pattern, response, re.IGNORECASE)
            if matches:
                try:
                    x, y = float(matches.group(1)), float(matches.group(2))
                    # Round to integers for pixel coordinates
                    return (int(round(x)), int(round(y)))
                except (ValueError, IndexError):
                    continue
        
        # If no pattern matches, return None
        print(f"Warning: Could not parse coordinates from response: {response}")
        return None

    def _convert_molmo_to_pixel_coords(self, normalized_coords: tuple, image_size: tuple) -> tuple:
        """
        Convert Molmo's normalized [0,100] coordinates to pixel coordinates.
        
        Args:
            normalized_coords: (norm_x, norm_y) coordinates in [0,100] range
            image_size: (width, height) of the image in pixels
            
        Returns:
            tuple: (pixel_x, pixel_y) coordinates
        """
        norm_x, norm_y = normalized_coords
        width, height = image_size
        
        pixel_x = int(round((norm_x / 100.0) * width))
        pixel_y = int(round((norm_y / 100.0) * height))
        
        # Ensure coordinates are within bounds
        pixel_x = max(0, min(pixel_x, width - 1))
        pixel_y = max(0, min(pixel_y, height - 1))
        
        return (pixel_x, pixel_y)
        
    def __enter__(self):
        """
        Context manager entry - ensures model is mounted.
        """
        self.mount()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit - unmounts model from GPU.
        """
        self.unmount()

class SAM2Inference:
    """
    A class for performing inference with SAM2 (Segment Anything Model 2).
    Provides functionality to mount/unmount the model from GPU as needed
    and process directories of images with propagation between frames.
    """
    
    def __init__(self, option: str, cache_dir=None, device=None):
        """
        Initialize the SAM2 inference class.
        
        Args:
            option (str): Select between 'image' or 'video' SAM model. 
            cache_dir (str, optional): Directory to store model files
            device (str, optional): Device to load the model on ('cuda', 'cpu', or None for auto-detection)
        """
        # Configure cache directory before importing SAM2 libraries
        configure_hf_cache(cache_dir)
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from sam2.sam2_video_predictor import SAM2VideoPredictor
        self.Sam2ImagePredictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-small")
        self.Sam2VideoPredictor = SAM2VideoPredictor.from_pretrained("facebook/sam2-hiera-small") 

        # Determine device
        self.device = device
        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Model State
        self.model = None
        self.mount(option)

    def mount(self, option: str):
        """
        Args: 
            option (str): Select between 'image' or 'video' SAM model. 
        """
        if option == 'image':
            self.model = self.Sam2ImagePredictor
        elif option == 'video':
            self.model = self.Sam2VideoPredictor
        else: 
            raise ValueError(f"Mount option '{option}' is not supported.")
        
    def unmount(self):
        """
        Unload the model from GPU/memory.
        """
        if self.model:
            self.model = None

    def _write_seg_image(self, mask, output_path):
        """
        Write segmentation mask to output path location.

        Args:
            mask (torch.Tensor | np.array): Predicted segmentation mask
            output_path: (str): Output image path
        """
        if isinstance(mask, np.ndarray):
            out_mask = (mask > 0.0).squeeze()
        elif isinstance(mask, torch.Tensor): 
            out_mask = (mask > 0.0).cpu().squeeze().numpy()
        uint8_array = (out_mask * 255).astype(np.uint8)
        img = Image.fromarray(uint8_array, mode='L')
        img.save(output_path)

    def _validate_prompts(self, input_points, input_box):
        """
        Validate that at least one prompt (points or box) is provided.
        
        Args:
            input_points: Points prompt
            input_box: Bounding box prompt
            
        Raises:
            ValueError: If neither points nor box are provided
        """
        if input_points is None and input_box is None:
            raise ValueError("At least one prompt is required: either input_points or input_box (or both)")

    def process_single_image(self, image, output_path, input_points=None, input_box=None, input_mask=None):
        """
        Process a single image with the SAM2 model.
        NOTE: Single subject segmentation for now.
        
        Args:
            image: PIL Image or path to image
            output_path: Output image path to write segmentation mask
            input_points: Optional points to prompt the model. SX2 array. 
            input_box: Optional bounding box to prompt the model. [x1, y1, x2, y2] array.
            input_mask: Optional masks to prompt the model. (image_dims) array.
            
        Raises:
            ValueError: If neither input_points nor input_box are provided
        """
        # Mount model
        self.mount(option='image')

        # Validate that at least one prompt is provided
        self._validate_prompts(input_points, input_box)
        
        # Load image
        if isinstance(image, str):
            image = Image.open(image)
        image = np.array(image.convert("RGB"))

        # Embed image
        self.model.set_image(image)

        # Define prediction inputs
        point_coords = None
        point_labels = None
        mask_input = None
        if input_points: 
            point_coords = input_points
            point_labels = np.ones(len(input_points))
        if input_mask:
            mask_input = input_mask

        # Inference
        masks, scores, _ = self.model.predict(
            point_coords=point_coords, 
            point_labels=point_labels, 
            box=input_box,
            mask_input=mask_input,
            multimask_output = False
        )

        self._write_seg_image(masks, output_path)

    def _create_batches(self, source_dir, output_dir, batch_size=100):
        """
        Organize images from source directory into batched subdirectories.
        Each image must be renamed into {number}.jpg to be compatible with SAM2 Video Predictor. 

        Args:
            source_dir (str): Path to source directory containing images
            output_dir (str): Path to output directory where images will be organized
            batch_size (int): Number of images per subdirectory
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        print('Splitting source directory into batches...')
        ordered_src_imgs = get_ordered_paths(source_dir)
        for i, filename in enumerate(ordered_src_imgs):
            # Calculate batch number (0-based)
            batch_num = i // batch_size
            
            # Create batch subdirectory
            batch_dir = os.path.join(output_dir, f"batch_{batch_num}")
            os.makedirs(batch_dir, exist_ok=True)
            
            # Copy file to batch directory
            source_path = os.path.join(source_dir, filename)
            dest_path = os.path.join(batch_dir, f"{i}.jpg")
            shutil.copy2(source_path, dest_path)
            
        print(f"Organized {len(ordered_src_imgs)} images into {(len(ordered_src_imgs) + batch_size - 1) // batch_size} batches")

    def process_directory(self, image_dir, output_dir, batch_size=100,
                          input_points=None, input_box=None, input_mask=None,
                          mask_threshold=0.5):
        """
        Process a directory of images with the SAM2 model using video propagation.
        NOTE: Single subject segmentation for now.
        
        Args:
            image_dir: Directory containing sequential images
            output_dir: Output directory for segmentation masks
            batch_size: Number of images to process in each batch
            input_points: Optional points to prompt the model. SX2 array. 
            input_box: Optional bounding box to prompt the model. [x1, y1, x2, y2] array.
            input_mask: Optional masks to prompt the model. (image_dims) array.
            mask_threshold: Threshold for binary mask conversion
            
        Returns:
            List of dictionaries with segmentation results
            
        Raises:
            ValueError: If neither input_points nor input_box are provided
        """
        # Mount model
        self.mount(option='video')
        
        # Validate that at least one prompt is provided
        self._validate_prompts(input_points, input_box)
        
        # Divide input directory into subdirectory batches inside scratch folder
        batch_dirs = Path(f'/scratch/{Path(image_dir).stem}_batches')
        self._create_batches(str(image_dir), str(batch_dirs))

        # Make the output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Inference Loop
        carryover_seg_mask = None
        ordered_batch_dirs = get_ordered_paths(batch_dirs)
        for i, sub_dir in enumerate(ordered_batch_dirs): 
            # Start of loop, embed prompts
            inference_state = self.model.init_state(video_path=str(sub_dir))
            
            # Idx to filename map
            resolve_filename = {j: p for j, p in 
                                enumerate(get_ordered_paths(sub_dir))}

            if i == 0: 
                # Set input prompts to input arguments
                ann_frame_idx = 0
                ann_obj_id = 1

                # Add bounding box prompt if provided
                if input_box is not None:
                    _, out_obj_ids, out_mask_logits = self.model.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=ann_frame_idx,
                        obj_id=ann_obj_id,
                        box=input_box,
                    )

                # Add point prompts if provided
                if input_points is not None:
                    _, out_obj_ids, out_mask_logits = self.model.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=ann_frame_idx,
                        obj_id=ann_obj_id,
                        points=input_points,
                        labels=np.ones(len(input_points)).astype(np.int32),
                    )
                
                # Add mask prompt if provided
                if input_mask is not None:
                    _, out_obj_ids, out_mask_logits = self.model.add_new_mask(
                        inference_state,
                        ann_frame_idx,
                        ann_obj_id,
                        input_mask,
                    )

                # Video Inference
                for out_frame_idx, out_obj_ids, out_mask_logits in self.model.propagate_in_video(inference_state):
                    if out_frame_idx == batch_size - 1:
                        carryover_seg_mask = (out_mask_logits > 0).squeeze()

                    output_img = Path(output_dir) / f'{resolve_filename[out_frame_idx].stem}.jpg'
                    self._write_seg_image(out_mask_logits, str(output_img))

            else:
                # Set input prompt to previous frame final segmentation
                ann_frame_idx = 0
                ann_obj_id = 1
                
                _, out_obj_ids, out_mask_logits = self.model.add_new_mask(
                    inference_state,
                    ann_frame_idx,
                    ann_obj_id,
                    carryover_seg_mask,
                )

                for out_frame_idx, out_obj_ids, out_mask_logits in self.model.propagate_in_video(inference_state):
                    if out_frame_idx == batch_size - 1:
                        carryover_seg_mask = (out_mask_logits > 0).squeeze()

                    output_img = Path(output_dir) / f'{resolve_filename[out_frame_idx].stem}.jpg'
                    self._write_seg_image(out_mask_logits, str(output_img))

    def __enter__(self):
        """
        Context manager entry
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit - unmounts model from GPU.
        """
        self.unmount()

def monte_carlo_sample_visible_points(
    extrinsic_matrices: list,
    intrinsic_matrices: list,
    seg_masks: list,
    target_points: int = 50000,
    visibility_percentage: float = 0.6,
    batch_size: int = 1000000,
    exploration_points: int = 1000000,
    region_expansion_factor: float = 1.5
) -> np.ndarray:
    """
    Optimized Monte Carlo sampling with adaptive region focusing.
    
    Args:
        extrinsic_matrices: List of camera extrinsic matrices (camera-to-world transform)
        intrinsic_matrices: List of camera intrinsic matrices corresponding to extrinsic_matrices
        seg_masks: List of segmentation masks corresponding to each camera
        target_points: Target number of points to return in the final point cloud
        visibility_percentage: Percentage of cameras a point must be visible from (0.0 to 1.0)
        batch_size: Number of points to sample in each batch during Monte Carlo sampling
        exploration_points: Number of points to find in exploration phase
        region_expansion_factor: Factor to expand the bounding region for focused sampling
        
    Returns:
        np.ndarray: Filtered point cloud containing points visible from the required percentage of views
    """
    # Input validation
    num_cameras = len(extrinsic_matrices)
    if not (len(intrinsic_matrices) == num_cameras and len(seg_masks) == num_cameras):
        raise ValueError(
            f"Mismatch in input sizes: extrinsic_matrices ({len(extrinsic_matrices)}), "
            f"intrinsic_matrices ({len(intrinsic_matrices)}), "
            f"seg_masks ({len(seg_masks)})"
        )
    
    min_visible_views = max(1, round(visibility_percentage * num_cameras))
    
    # Pre-compute camera information
    cam_centers = []
    projection_matrices = []
    
    for i, (ext, k, seg) in enumerate(zip(extrinsic_matrices, intrinsic_matrices, seg_masks)):
        # Camera center
        cam_centers.append(np.linalg.inv(ext)[:3, 3])
        
        # Pre-compute projection matrix
        h, w = seg.shape
        fx, fy, cx, cy = k[0][0], k[1][1], k[0][2], k[1][2]
        z_near = 1.0
        z_far = 1000.0  # Conservative estimate, will be refined later
        
        proj = np.array([
            [2 * fx / w, 0.0, -(w - 2 * cx) / w, 0.0],
            [0.0, 2 * fy / h, -(h - 2 * cy) / h, 0.0],
            [0.0, 0.0, z_far / (z_far - z_near), -(z_far * z_near) / (z_far - z_near)],
            [0.0, 0.0, 1.0, 0.0]
        ])
        
        projection_matrices.append((ext, proj, seg, w, h))
    
    cam_centers = np.array(cam_centers)
    scene_center = np.mean(cam_centers, axis=0)
    scene_radius = 1.1 * np.max(np.linalg.norm(cam_centers - scene_center, axis=1))
    
    print("Phase 1: Exploration - Finding initial visible region...")
    
    # Phase 1: Exploration sampling to find the visible region
    exploration_samples = _sample_sphere_uniform(scene_center, scene_radius, exploration_points * 20)  # Oversample
    visible_points = _filter_visible_points(exploration_samples, projection_matrices, min_visible_views)
    
    if len(visible_points) < 10:
        print("Warning: Very few visible points found in exploration. Using original full-sphere sampling.")
        return _sample_full_sphere(
            projection_matrices, scene_center, scene_radius, 
            target_points, min_visible_views, batch_size
        )
    
    # Take first exploration_points for region estimation
    visible_points = visible_points[:exploration_points]
    print(f"Found {len(visible_points)} visible points in exploration phase")
    
    # Phase 2: Define focused sampling region
    focused_region = _compute_focused_region(visible_points, region_expansion_factor)
    print(f"Focused region: center={focused_region['center']}, radius={focused_region['bounds']}")
    
    # Phase 3: Focused sampling in the identified region
    print("Phase 2: Focused sampling in identified region...")
    result_points = np.copy(visible_points)  # Start with exploration points
    
    # Adjust batch size for focused region (smaller region = smaller batches can be more efficient)
    focused_batch_size = min(batch_size, batch_size // 4)
    
    while len(result_points) < target_points:
        # Sample in focused region
        remaining_points = target_points - len(result_points)
        current_batch_size = min(focused_batch_size, remaining_points * 10)  # Oversample factor
        
        samples = _sample_focused_region(focused_region, current_batch_size)
        visible_batch = _filter_visible_points(samples, projection_matrices, min_visible_views)
        
        if len(visible_batch) > 0:
            result_points = np.vstack((result_points, visible_batch))
            print(f"Progress: {len(result_points)}/{target_points} points")
        
    return result_points[:target_points]

def _sample_sphere_uniform(center: np.ndarray, radius: float, n_samples: int) -> np.ndarray:
    """Sample points uniformly within a sphere."""
    theta = np.random.uniform(0, 2*np.pi, n_samples)
    phi = np.arccos(np.random.uniform(-1, 1, n_samples))
    r = radius * np.cbrt(np.random.uniform(0, 1, n_samples))
    
    x = center[0] + r * np.sin(phi) * np.cos(theta)
    y = center[1] + r * np.sin(phi) * np.sin(theta)
    z = center[2] + r * np.cos(phi)
    
    return np.column_stack((x, y, z))

def _sample_focused_region(region_info: dict, n_samples: int) -> np.ndarray:
    # Sample within axis-aligned bounding box
    mins, maxs = region_info['bounds']
    samples = np.random.uniform(mins, maxs, (n_samples, 3))
    return samples

def _compute_focused_region(visible_points: np.ndarray, expansion_factor: float) -> dict:
    """Compute focused sampling region based on visible points."""
    # Use bounding box approach for better efficiency
    mins = np.min(visible_points, axis=0)
    maxs = np.max(visible_points, axis=0)
    
    # Expand the bounding box
    center = (mins + maxs) / 2
    extent = (maxs - mins) / 2 * expansion_factor
    
    expanded_mins = center - extent
    expanded_maxs = center + extent
    
    return {
        'type': 'box',
        'bounds': (expanded_mins, expanded_maxs),
        'center': center
    }

def _filter_visible_points(samples: np.ndarray, projection_matrices: list, min_visible_views: int) -> np.ndarray:
    """Filter points based on visibility criteria."""
    if len(samples) == 0:
        return np.zeros((0, 3))
    
    batch_size = len(samples)
    global_visible_mask = np.zeros(batch_size)
    
    # Homogeneous coordinates
    samples_homo = np.hstack((samples, np.ones((batch_size, 1))))
    
    for ext, proj, seg, w, h in projection_matrices:
        # Apply transformations
        transformed = samples_homo @ ext.T @ proj.T
        
        # Perspective divide
        ndc = transformed[:, :3] / transformed[:, 3:4]
        
        # Convert to pixel coordinates
        pixel_coords = (ndc[:, :2] + 1) / 2 * np.array([w, h])
        pixel_coords = np.round(pixel_coords).astype(int)
        
        # Check bounds
        valid_mask = (
            (pixel_coords[:, 0] >= 0) & (pixel_coords[:, 0] < w) &
            (pixel_coords[:, 1] >= 0) & (pixel_coords[:, 1] < h)
        )
        
        # Check segmentation mask for valid points
        local_visible_mask = np.zeros(batch_size)
        if np.any(valid_mask):
            valid_indices = np.where(valid_mask)[0]
            valid_pixels = pixel_coords[valid_mask]
            local_visible_mask[valid_indices] = seg[valid_pixels[:, 1], valid_pixels[:, 0]]
        
        global_visible_mask += local_visible_mask
    
    # Return points visible from enough cameras
    valid_indices = global_visible_mask >= min_visible_views
    return samples[valid_indices]

def _sample_full_sphere(projection_matrices, scene_center, scene_radius, target_points, min_visible_views, batch_size):
    """Fallback to original full-sphere sampling method."""
    result_points = np.zeros((0, 3))
    
    while len(result_points) < target_points:
        samples = _sample_sphere_uniform(scene_center, scene_radius, batch_size)
        visible_batch = _filter_visible_points(samples, projection_matrices, min_visible_views)
        
        if len(visible_batch) > 0:
            result_points = np.vstack((result_points, visible_batch))
            print(f"Fallback sampling: {len(result_points)}/{target_points} points")
        
        if len(result_points) > target_points:
            result_points = result_points[:target_points]
            break
    
    return result_points
