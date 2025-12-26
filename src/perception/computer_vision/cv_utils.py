"""
Computer vision utility functions for the perception system.

Provides helper functions for image processing, geometric transformations,
coordinate conversions, and depth estimation.
"""
from typing import Tuple, List, Optional, Union
import numpy as np
import cv2
from geometry_msgs.msg import Point


def resize_image(image: np.ndarray, width: Optional[int] = None,
                height: Optional[int] = None,
                interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
    """
    Resize an image while maintaining aspect ratio.

    Args:
        image: Input image
        width: Target width (if None, calculated from height)
        height: Target height (if None, calculated from width)
        interpolation: Interpolation method

    Returns:
        Resized image
    """
    h, w = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        # Calculate width to maintain aspect ratio
        ratio = height / float(h)
        width = int(w * ratio)
    elif height is None:
        # Calculate height to maintain aspect ratio
        ratio = width / float(w)
        height = int(h * ratio)

    resized = cv2.resize(image, (width, height), interpolation=interpolation)
    return resized


def crop_image(image: np.ndarray, bbox: Union[Tuple, np.ndarray]) -> np.ndarray:
    """
    Crop image using bounding box.

    Args:
        image: Input image
        bbox: Bounding box as (x, y, width, height) or [x, y, width, height]

    Returns:
        Cropped image
    """
    if isinstance(bbox, np.ndarray):
        bbox = bbox.astype(int)
    else:
        bbox = tuple(int(x) for x in bbox)

    x, y, w, h = bbox

    # Ensure coordinates are within image bounds
    height, width = image.shape[:2]
    x = max(0, min(x, width))
    y = max(0, min(y, height))
    w = max(0, min(w, width - x))
    h = max(0, min(h, height - y))

    cropped = image[y:y+h, x:x+w]
    return cropped


def enhance_image(image: np.ndarray, method: str = 'clahe') -> np.ndarray:
    """
    Enhance image contrast and brightness.

    Args:
        image: Input image (grayscale or BGR)
        method: Enhancement method ('clahe', 'hist_eq', 'gamma')

    Returns:
        Enhanced image
    """
    if len(image.shape) == 3:
        # Convert to LAB color space for better enhancement
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        if method == 'clahe':
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_enhanced = clahe.apply(l)
        elif method == 'hist_eq':
            # Apply histogram equalization to L channel
            l_enhanced = cv2.equalizeHist(l)
        else:
            l_enhanced = l

        # Merge channels and convert back to BGR
        enhanced_lab = cv2.merge([l_enhanced, a, b])
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    else:
        # Grayscale image
        if method == 'clahe':
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(image)
        elif method == 'hist_eq':
            enhanced = cv2.equalizeHist(image)
        else:
            enhanced = image

    return enhanced


def denoise_image(image: np.ndarray, method: str = 'bilateral',
                 strength: int = 10) -> np.ndarray:
    """
    Denoise an image.

    Args:
        image: Input image
        method: Denoising method ('bilateral', 'gaussian', 'median', 'nlm')
        strength: Denoising strength

    Returns:
        Denoised image
    """
    if method == 'bilateral':
        denoised = cv2.bilateralFilter(image, 9, strength * 2, strength * 2)
    elif method == 'gaussian':
        denoised = cv2.GaussianBlur(image, (5, 5), strength / 10.0)
    elif method == 'median':
        denoised = cv2.medianBlur(image, 5)
    elif method == 'nlm':
        if len(image.shape) == 3:
            denoised = cv2.fastNlMeansDenoisingColored(image, None, strength, strength, 7, 21)
        else:
            denoised = cv2.fastNlMeansDenoising(image, None, strength, 7, 21)
    else:
        denoised = image

    return denoised


def compute_iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    """
    Compute Intersection over Union (IoU) between two bounding boxes.

    Args:
        bbox1: First bounding box [x, y, width, height]
        bbox2: Second bounding box [x, y, width, height]

    Returns:
        IoU value between 0 and 1
    """
    x1_1, y1_1, w1, h1 = bbox1
    x1_2, y1_2, w2, h2 = bbox2

    # Convert to (x1, y1, x2, y2) format
    x2_1, y2_1 = x1_1 + w1, y1_1 + h1
    x2_2, y2_2 = x1_2 + w2, y1_2 + h2

    # Calculate intersection
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Calculate union
    bbox1_area = w1 * h1
    bbox2_area = w2 * h2
    union_area = bbox1_area + bbox2_area - intersection_area

    if union_area == 0:
        return 0.0

    iou = intersection_area / union_area
    return float(iou)


def non_max_suppression(bboxes: List[np.ndarray], scores: List[float],
                       threshold: float = 0.5) -> List[int]:
    """
    Apply Non-Maximum Suppression to bounding boxes.

    Args:
        bboxes: List of bounding boxes [x, y, width, height]
        scores: List of confidence scores
        threshold: IoU threshold for suppression

    Returns:
        Indices of boxes to keep
    """
    if len(bboxes) == 0:
        return []

    # Convert to numpy arrays
    bboxes = np.array(bboxes)
    scores = np.array(scores)

    # Get coordinates
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 0] + bboxes[:, 2]
    y2 = bboxes[:, 1] + bboxes[:, 3]

    # Calculate areas
    areas = bboxes[:, 2] * bboxes[:, 3]

    # Sort by score
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        # Calculate IoU with remaining boxes
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        intersection = w * h

        iou = intersection / (areas[i] + areas[order[1:]] - intersection)

        # Keep boxes with IoU less than threshold
        indices = np.where(iou <= threshold)[0]
        order = order[indices + 1]

    return keep


def estimate_depth_from_bbox(bbox: np.ndarray, image_shape: Tuple,
                             focal_length: float = 525.0,
                             real_object_height: float = 1.7) -> float:
    """
    Estimate depth of object from bounding box size (simplified pinhole model).

    Args:
        bbox: Bounding box [x, y, width, height]
        image_shape: Shape of image (height, width)
        focal_length: Camera focal length in pixels
        real_object_height: Real-world height of object in meters

    Returns:
        Estimated depth in meters
    """
    # Extract bbox height in pixels
    bbox_height = bbox[3]

    # Depth = (focal_length * real_height) / pixel_height
    if bbox_height == 0:
        return float('inf')

    depth = (focal_length * real_object_height) / bbox_height

    return float(depth)


def pixel_to_camera_coordinates(pixel_x: float, pixel_y: float, depth: float,
                                fx: float = 525.0, fy: float = 525.0,
                                cx: float = 319.5, cy: float = 239.5) -> np.ndarray:
    """
    Convert pixel coordinates to 3D camera coordinates.

    Args:
        pixel_x: X coordinate in pixels
        pixel_y: Y coordinate in pixels
        depth: Depth in meters
        fx: Focal length in x (pixels)
        fy: Focal length in y (pixels)
        cx: Principal point x (pixels)
        cy: Principal point y (pixels)

    Returns:
        3D point in camera frame [x, y, z]
    """
    x = (pixel_x - cx) * depth / fx
    y = (pixel_y - cy) * depth / fy
    z = depth

    return np.array([x, y, z])


def camera_to_robot_coordinates(camera_point: np.ndarray,
                                camera_transform: np.ndarray) -> np.ndarray:
    """
    Transform point from camera frame to robot base frame.

    Args:
        camera_point: Point in camera frame [x, y, z]
        camera_transform: 4x4 transformation matrix from camera to robot base

    Returns:
        Point in robot frame [x, y, z]
    """
    # Convert to homogeneous coordinates
    point_homogeneous = np.append(camera_point, 1.0)

    # Apply transformation
    robot_point_homogeneous = camera_transform @ point_homogeneous

    # Convert back to 3D
    robot_point = robot_point_homogeneous[:3]

    return robot_point


def estimate_object_position_3d(bbox: np.ndarray, depth_image: Optional[np.ndarray],
                                image_shape: Tuple,
                                camera_params: dict) -> np.ndarray:
    """
    Estimate 3D position of detected object.

    Args:
        bbox: Bounding box [x, y, width, height]
        depth_image: Depth image (if available)
        image_shape: Shape of image (height, width)
        camera_params: Camera parameters dict with 'fx', 'fy', 'cx', 'cy'

    Returns:
        3D position [x, y, z] in camera frame
    """
    # Calculate center of bounding box
    center_x = bbox[0] + bbox[2] / 2
    center_y = bbox[1] + bbox[3] / 2

    # Get depth
    if depth_image is not None:
        # Use actual depth from depth image
        cx_int, cy_int = int(center_x), int(center_y)
        # Average depth over bounding box region
        x1, y1 = int(bbox[0]), int(bbox[1])
        x2, y2 = int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])
        depth_region = depth_image[y1:y2, x1:x2]
        depth = np.median(depth_region[depth_region > 0]) if depth_region.size > 0 else 1.0
    else:
        # Estimate depth from bbox size (assuming human-sized object)
        depth = estimate_depth_from_bbox(bbox, image_shape)

    # Convert to 3D coordinates
    position_3d = pixel_to_camera_coordinates(
        center_x, center_y, depth,
        camera_params.get('fx', 525.0),
        camera_params.get('fy', 525.0),
        camera_params.get('cx', image_shape[1] / 2),
        camera_params.get('cy', image_shape[0] / 2)
    )

    return position_3d


def draw_3d_box(image: np.ndarray, bbox_3d: np.ndarray,
               camera_matrix: np.ndarray,
               color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    """
    Draw 3D bounding box on image.

    Args:
        image: Input image
        bbox_3d: 3D bounding box corners (8x3 array)
        camera_matrix: 3x3 camera intrinsic matrix
        color: Color for drawing

    Returns:
        Image with 3D box drawn
    """
    vis_image = image.copy()

    # Project 3D points to 2D
    points_2d = []
    for point_3d in bbox_3d:
        if len(point_3d) == 3:
            point_2d = camera_matrix @ point_3d
            point_2d = point_2d[:2] / point_2d[2]
            points_2d.append(point_2d.astype(int))

    points_2d = np.array(points_2d)

    # Draw edges of the 3D box
    # Define edges (connecting vertices)
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
        (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
        (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical edges
    ]

    for edge in edges:
        pt1 = tuple(points_2d[edge[0]])
        pt2 = tuple(points_2d[edge[1]])
        cv2.line(vis_image, pt1, pt2, color, 2)

    return vis_image


def compute_distance_between_points(point1: np.ndarray,
                                    point2: np.ndarray) -> float:
    """
    Compute Euclidean distance between two 3D points.

    Args:
        point1: First point [x, y, z]
        point2: Second point [x, y, z]

    Returns:
        Distance in meters
    """
    distance = np.linalg.norm(point2 - point1)
    return float(distance)


def filter_by_distance(positions: List[np.ndarray],
                      max_distance: float = 10.0) -> List[int]:
    """
    Filter positions by maximum distance from origin.

    Args:
        positions: List of 3D positions
        max_distance: Maximum distance threshold

    Returns:
        Indices of positions within threshold
    """
    keep_indices = []

    for i, pos in enumerate(positions):
        distance = np.linalg.norm(pos)
        if distance <= max_distance:
            keep_indices.append(i)

    return keep_indices


def convert_to_ros_point(position: np.ndarray) -> Point:
    """
    Convert numpy array to ROS geometry_msgs/Point.

    Args:
        position: 3D position [x, y, z]

    Returns:
        ROS Point message
    """
    point = Point()
    point.x = float(position[0])
    point.y = float(position[1])
    point.z = float(position[2]) if len(position) > 2 else 0.0

    return point


def rotate_image(image: np.ndarray, angle: float,
                center: Optional[Tuple[int, int]] = None,
                scale: float = 1.0) -> np.ndarray:
    """
    Rotate image by specified angle.

    Args:
        image: Input image
        angle: Rotation angle in degrees
        center: Center of rotation (if None, uses image center)
        scale: Scale factor

    Returns:
        Rotated image
    """
    height, width = image.shape[:2]

    if center is None:
        center = (width // 2, height // 2)

    # Get rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)

    # Rotate image
    rotated = cv2.warpAffine(image, rotation_matrix, (width, height))

    return rotated


def validate_bbox(bbox: np.ndarray, image_shape: Tuple) -> bool:
    """
    Validate that bounding box is within image bounds.

    Args:
        bbox: Bounding box [x, y, width, height]
        image_shape: Image shape (height, width)

    Returns:
        True if valid, False otherwise
    """
    x, y, w, h = bbox
    img_h, img_w = image_shape[:2]

    # Check if bbox is within bounds
    if x < 0 or y < 0 or x + w > img_w or y + h > img_h:
        return False

    # Check if bbox has valid size
    if w <= 0 or h <= 0:
        return False

    return True
