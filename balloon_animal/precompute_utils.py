import re
import os
from pathlib import Path
import shutil
import random
from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple, Optional

import cv2
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def extract_number(filename: str) -> int:
    """
    Input: 
        filename: input filename
    Returns:
        integer contained in filename
    """

    match = re.search(r'\d+', filename)
    if match:
        return int(match.group())
    raise ValueError(f"No number found in filename: {filename}")

def get_ordered_paths(input_dir: str) -> list[Path]:
    """
    Input: 
        input_dir
    Returns:
        list of ordered files by number
    """
    files = sorted(Path(input_dir).iterdir(), 
            key=lambda x: extract_number(str(x.stem)))
    return files

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
from pathlib import Path

def visualize_point_prediction(image, predicted_point, output_path, subject=None, 
                         point_color='red', point_size=100, cross_size=20,
                         figsize=(12, 8), dpi=150):
    """
    Plot the input image with the predicted point overlaid and save to output path.
    
    Args:
        image: PIL Image or path to image file
        predicted_point: Tuple of (x, y) coordinates for the predicted point
        output_path: Path where the visualization will be saved
        subject: Optional string describing what was pointed to (for title)
        point_color: Color of the point marker (default: 'red')
        point_size: Size of the point marker (default: 100)
        cross_size: Size of the cross marker arms (default: 20)
        figsize: Figure size as (width, height) tuple (default: (12, 8))
        dpi: DPI for the saved image (default: 150)
    """
    # Load image if path is provided
    if isinstance(image, str):
        image = Image.open(image)
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Create figure and axis
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    
    # Display the image
    ax.imshow(np.array(image))
    
    # Extract coordinates
    if predicted_point is not None:
        x, y = predicted_point
        
        # Plot the point as a circle
        ax.scatter(x, y, c=point_color, s=point_size, alpha=0.8, 
                  edgecolors='white', linewidth=2, zorder=5)
        
        # Add a cross marker for better visibility
        ax.plot([x - cross_size, x + cross_size], [y, y], 
               color=point_color, linewidth=3, alpha=0.9, zorder=6)
        ax.plot([x, x], [y - cross_size, y + cross_size], 
               color=point_color, linewidth=3, alpha=0.9, zorder=6)
        
        # Add coordinate text
        coord_text = f"({x}, {y})"
        ax.annotate(coord_text, (x, y), xytext=(10, 10), 
                   textcoords='offset points', fontsize=12, 
                   color=point_color, weight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Set title
    if subject:
        title = f"Molmo Prediction: Pointing to '{subject}'"
        if predicted_point is not None:
            title += f" at ({predicted_point[0]}, {predicted_point[1]})"
    else:
        title = "Molmo Prediction Result"
        if predicted_point is not None:
            title += f" at ({predicted_point[0]}, {predicted_point[1]})"
    
    ax.set_title(title, fontsize=14, weight='bold', pad=20)
    
    # Remove axis ticks and labels for cleaner look
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Ensure the plot fits tightly
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save the plot
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"Visualization saved to: {output_path}")

def visualize_detections(image, detections, output_path=None, 
                         color_map=None, font_path=None, font_size=15, 
                         thickness=3, show_scores=True):
    """
    Draw bounding boxes on an image based on detection results.
    
    Args:
        image: PIL Image object or path to image
        detections: List of detection dictionaries with 'box', 'score', and 'class' keys
        output_path: Path to save the output image (if None, won't save)
        color_map: Dictionary mapping class names to RGB tuples (will generate if None)
        font_path: Path to TrueType font file (will use default if None)
        font_size: Font size for labels
        thickness: Thickness of bounding box lines
        show_scores: Whether to display confidence scores
        
    Returns:
        PIL Image with the drawn bounding boxes
    """
    # Load image if path is provided
    if isinstance(image, str):
        image = Image.open(image)
    
    # Create a copy of the image to draw on
    image_with_boxes = image.copy()
    draw = ImageDraw.Draw(image_with_boxes)
    
    # Create or use provided color mapping
    if color_map is None:
        color_map = {}
        classes = set(det["class"] for det in detections)
        for cls in classes:
            # Generate vibrant colors with good contrast
            color_map[cls] = (
                random.randint(100, 255),
                random.randint(100, 255),
                random.randint(100, 255)
            )
    
    # Try to load the specified font, fall back to default if needed
    try:
        if font_path:
            font = ImageFont.truetype(font_path, font_size)
        else:
            # Try some common system fonts before falling back to default
            for system_font in ["arial.ttf", "Arial.ttf", "DejaVuSans.ttf", "Verdana.ttf"]:
                try:
                    font = ImageFont.truetype(system_font, font_size)
                    break
                except IOError:
                    continue
            else:
                font = ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()
    
    # Draw each detection
    for det in detections:
        box = det["box"]
        score = det["score"]
        class_name = det["class"]
        
        # Get color for this class
        color = color_map.get(class_name, (255, 0, 0))  # Default to red if not found
        
        # Draw rectangle (box)
        draw.rectangle(
            [(box[0], box[1]), (box[2], box[3])], 
            outline=color, 
            width=thickness
        )
        
        # Create label
        if show_scores:
            label = f"{class_name}: {round(score, 2)}"
        else:
            label = class_name
        
        # Calculate text size and position
        # Handle different versions of PIL
        if hasattr(draw, 'textbbox'):
            # For newer PIL versions
            bbox = draw.textbbox((box[0], box[1]), label, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        elif hasattr(font, 'getbbox'):
            # For PIL 9.0.0+
            bbox = font.getbbox(label)
            text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        else:
            # For older PIL versions
            text_width, text_height = draw.textsize(label, font=font)
        
        # Draw label background
        draw.rectangle(
            [(box[0], box[1] - text_height - 4), (box[0] + text_width + 4, box[1])],
            fill=color
        )
        
        # Draw label text
        draw.text((box[0] + 2, box[1] - text_height - 2), label, fill="white", font=font)
    
    # Save the image if output path is provided
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        image_with_boxes.save(output_path)
        print(f"Visualization saved to: {output_path}")
    
    return image_with_boxes

def create_video_from_images(image_paths: List[str], output_path: str, fps: int = 30, codec: str = 'mp4v') -> bool:
    """
    Create a video from a list of image paths.
    
    Args:
        image_paths: List of paths to the images in the desired order
        output_path: Path where the output video will be saved
        fps: Frames per second for the output video
        codec: FourCC codec code (default: mp4v for .mp4 files)
    
    Returns:
        bool: True if video creation was successful, False otherwise
    """
    if not image_paths:
        print("Error: No images provided")
        return False
    
    # Read the first image to get dimensions
    first_image = cv2.imread(image_paths[0])
    if first_image is None:
        print(f"Error: Could not read the first image: {image_paths[0]}")
        return False
    
    height, width, layers = first_image.shape
    size = (width, height)
    
    # Get the FourCC code
    fourcc = cv2.VideoWriter_fourcc(*codec)
    
    # Create the VideoWriter object
    out = cv2.VideoWriter(output_path, fourcc, fps, size)
    
    # Check if the VideoWriter was successfully created
    if not out.isOpened():
        print(f"Error: Could not create video writer with codec {codec}")
        return False
    
    print(f"Processing {len(image_paths)} images...")
    
    # Process each image
    for i, img_path in enumerate(image_paths):
        if i % 10 == 0:  # Print progress every 10 frames
            print(f"Processing image {i+1}/{len(image_paths)}: {img_path}")
        
        img = cv2.imread(img_path)
        
        # Check if the image was successfully loaded
        if img is None:
            print(f"Warning: Could not read image {img_path}, skipping...")
            continue
        
        # Check if the image has the same dimensions as the first image
        if img.shape[:2] != (height, width):
            print(f"Warning: Image {img_path} has different dimensions, resizing...")
            img = cv2.resize(img, size)
        
        # Write the image to the video
        out.write(img)
    
    # Release the VideoWriter
    out.release()
    print(f"Video saved to {output_path}")
    return True

def visualize_scene(extrinsic_matrices: List[np.ndarray], point_cloud: np.ndarray, 
                   output_path: str, skeleton_points: Optional[np.ndarray] = None, 
                   skeleton_edges: Optional[List[Tuple[int, int]]] = None):
    """
    Plot camera array, point cloud, and optional skeleton in plotly.
    
    Args:
        extrinsic_matrices: List of 4x4 extrinsic matrices
        point_cloud: (N, 3) array of points
        output_path: HTML destination
        skeleton_points: Optional (M, 3) array of points representing skeleton joints
        skeleton_edges: Optional list of tuples (i, j) representing edges between skeleton_points
    """

    # Initialize the 3D plot
    fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scatter3d'}]])    
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow']
    arrow_length = 30
    camera_size = 100
    
    # Store min and max coordinates for setting axis ranges
    all_points = []
    
    # Plot each camera
    for i, extrinsic in enumerate(extrinsic_matrices):
        # Get the color for this camera
        color = colors[i % len(colors)]
        
        # Extract rotation matrix and translation vector
        R = extrinsic[:3, :3]
        t = extrinsic[:3, 3]
        
        # Calculate camera center C = -R^T * t
        # (The camera center is the position of the camera in world coordinates)
        C = -np.dot(R.T, t)
        
        # Get camera orientation vectors (columns of R^T)
        RT = R.T
        x_axis = RT[:, 0] * camera_size
        y_axis = RT[:, 1] * camera_size
        z_axis = RT[:, 2] * camera_size
        
        # Calculate the direction the camera is pointing (z-axis of camera frame)
        # The negative z-axis points forward in a right-handed camera coordinate system
        direction = z_axis * arrow_length
        
        # Store all points for axis range determination
        all_points.append(C)
        all_points.append(C + direction)
        
        # Add camera center as a marker
        fig.add_trace(go.Scatter3d(
            x=[C[0]],
            y=[C[1]],
            z=[C[2]],
            mode='markers',
            marker=dict(size=8, color=color),
            name=f'Camera {i+1} Center'
        ))
        
        # Add camera viewing direction as an arrow
        fig.add_trace(go.Scatter3d(
            x=[C[0], C[0] + direction[0]],
            y=[C[1], C[1] + direction[1]],
            z=[C[2], C[2] + direction[2]],
            mode='lines',
            line=dict(width=5, color=color),
            name=f'Camera {i+1} Direction'
        ))
        
        # Plot camera coordinate axes
        # X-axis (red)
        fig.add_trace(go.Scatter3d(
            x=[C[0], C[0] + x_axis[0]],
            y=[C[1], C[1] + x_axis[1]],
            z=[C[2], C[2] + x_axis[2]],
            mode='lines',
            line=dict(width=2, color='red'),
            showlegend=False if i > 0 else True,
            name='X-axis'
        ))
        
        # Y-axis (green)
        fig.add_trace(go.Scatter3d(
            x=[C[0], C[0] + y_axis[0]],
            y=[C[1], C[1] + y_axis[1]],
            z=[C[2], C[2] + y_axis[2]],
            mode='lines',
            line=dict(width=2, color='green'),
            showlegend=False if i > 0 else True,
            name='Y-axis'
        ))
        
        # Z-axis (blue)
        fig.add_trace(go.Scatter3d(
            x=[C[0], C[0] + z_axis[0]],
            y=[C[1], C[1] + z_axis[1]],
            z=[C[2], C[2] + z_axis[2]],
            mode='lines',
            line=dict(width=2, color='blue'),
            showlegend=False if i > 0 else True,
            name='Z-axis'
        ))
    
    # Plot the point cloud if provided
    if point_cloud is not None and len(point_cloud) > 0:
        # Add point cloud as scatter points
        fig.add_trace(go.Scatter3d(
            x=point_cloud[:, 0],
            y=point_cloud[:, 1],
            z=point_cloud[:, 2],
            mode='markers',
            marker=dict(
                size=2,
                color='purple',  # Default color
                opacity=0.7,
                line=dict(width=0)
            ),
            name='Point Cloud'
        ))
        
        # Include point cloud in axis range calculation
        all_points.extend(point_cloud)
    
    # Plot the skeleton if provided
    if skeleton_points is not None and skeleton_edges is not None:
        # Include skeleton points in axis range calculation
        all_points.extend(skeleton_points)
        
        # Add skeleton joints as markers
        fig.add_trace(go.Scatter3d(
            x=skeleton_points[:, 0],
            y=skeleton_points[:, 1],
            z=skeleton_points[:, 2],
            mode='markers',
            marker=dict(
                size=6,
                color='darkred',  # Joint color
                symbol='circle',
                line=dict(width=1, color='black')
            ),
            name='Skeleton Joints'
        ))
        
        # Add skeleton bones as lines
        for edge in skeleton_edges:
            # Check if edge indices are valid
            if edge[0] < len(skeleton_points) and edge[1] < len(skeleton_points):
                p1 = skeleton_points[edge[0]]
                p2 = skeleton_points[edge[1]]
                
                fig.add_trace(go.Scatter3d(
                    x=[p1[0], p2[0]],
                    y=[p1[1], p2[1]],
                    z=[p1[2], p2[2]],
                    mode='lines',
                    line=dict(width=4, color='black'),
                    showlegend=False if edge != skeleton_edges[0] else True,
                    name='Skeleton Bones' if edge == skeleton_edges[0] else None
                ))
    
    # Calculate appropriate axis ranges
    all_points = np.array(all_points)
    min_vals = np.min(all_points, axis=0)
    max_vals = np.max(all_points, axis=0)
    center = (min_vals + max_vals) / 2
    
    # Find the maximum range across all dimensions
    max_range = np.max(max_vals - min_vals)
    
    # Set axis ranges to be equal
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[center[0] - max_range/2, center[0] + max_range/2]),
            yaxis=dict(range=[center[1] - max_range/2, center[1] + max_range/2]),
            zaxis=dict(range=[center[2] - max_range/2, center[2] + max_range/2]),
            aspectmode='cube'
        ),
        title='Camera Positions and Orientations with Point Cloud and Skeleton',
        scene_camera=dict(
            eye=dict(x=1.5, y=1.5, z=1.5)
        )
    )
    
    # Export to HTML
    fig.write_html(output_path)
    print(f"Visualization saved to {output_path}")
    
    return fig
