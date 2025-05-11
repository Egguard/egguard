import cv2
import numpy as np
from math import atan2, cos, sin, sqrt

def is_valid_egg(image, center, radius):
    """
    Verify if the detected circle is actually an egg by analyzing color and shape.
    
    Args:
        image (numpy.ndarray): Original image in BGR format
        center (tuple): (x, y) center coordinates of the potential egg
        radius (int): Radius of the detected potential egg
    
    Returns:
        bool: True if the object appears to be an egg, False otherwise
    """
    # Extract the region of interest (ROI) containing the potential egg
    x, y = center
    x, y = int(x), int(y)
    r = int(radius * 1.2)  # Slightly larger ROI to ensure full egg coverage
    
    # Define ROI boundaries ensuring they're within image bounds
    height, width = image.shape[:2]
    x_min = max(0, x - r)
    y_min = max(0, y - r)
    x_max = min(width, x + r)
    y_max = min(height, y + r)
    
    # Extract the ROI
    roi = image[y_min:y_max, x_min:x_max]
    
    if roi.size == 0:
        return False  # ROI is empty, not an egg
        
    # Create a circular mask for the egg region
    mask = np.zeros((y_max - y_min, x_max - x_min), dtype=np.uint8)
    cv2.circle(
        mask, 
        (x - x_min, y - y_min),  # Adjust center to ROI coordinates
        int(radius * 0.9),  # Slightly smaller to focus on egg surface
        255, 
        -1
    )
    
    # Convert ROI to HSV for color analysis
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # Mask the ROI to focus on the egg region
    masked_hsv = cv2.bitwise_and(hsv_roi, hsv_roi, mask=mask)
    
    # Skip if no pixels are masked
    non_zero = cv2.countNonZero(mask)
    if non_zero == 0:
        return False
    
    # Calculate color statistics in the egg region
    mean_hsv = cv2.mean(masked_hsv, mask=mask)
    
    # Egg color heuristics (adjust these based on your eggs)
    # Brown/white eggs typically have:
    # - Low saturation
    # - Medium to high value/brightness
    # - Hue in specific ranges (brown eggs: 0-30, white eggs: broad range, low saturation)
    saturation = mean_hsv[1]
    value = mean_hsv[2]
    
    # Check if color profile matches egg
    # Eggs typically have medium-high brightness and low-medium saturation
    is_egg_color = (saturation < 100) and (value > 100)
    
    # Check shape compactness (eggs are roughly circular/oval)
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary = cv2.bitwise_and(binary, mask)
    
    # Find contours in the masked binary image
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return False
    
    # Find the largest contour (should be the egg)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Calculate contour features
    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, True)
    
    # Calculate circularity/compactness (1.0 is a perfect circle)
    # Eggs should have reasonably high circularity
    circularity = 0
    if perimeter > 0:
        circularity = 4 * np.pi * area / (perimeter * perimeter)
    
    # Eggs have high circularity but not perfect (they're slightly oval)
    is_egg_shape = 0.6 < circularity < 0.95
    
    # Combine color and shape evidence
    return is_egg_color and is_egg_shape

def detect_broken_egg(image, center, radius):
    """
    Detect if an egg is broken by analyzing its contour regularity and texture.
    First validates if the object is truly an egg.
    
    Args:
        image (numpy.ndarray): Original image in BGR format
        center (tuple): (x, y) center coordinates of the egg
        radius (int): Radius of the detected egg
    
    Returns:
        bool: True if the egg appears to be broken, False otherwise
    """
    # First, verify this is actually an egg
    if not is_valid_egg(image, center, radius):
        return False  # Not an egg, so not a broken egg
    
    # Extract the region of interest (ROI) containing the egg
    x, y = center
    x, y = int(x), int(y)
    r = int(radius * 1.2)  # Slightly larger ROI to ensure full egg coverage
    
    # Define ROI boundaries ensuring they're within image bounds
    height, width = image.shape[:2]
    x_min = max(0, x - r)
    y_min = max(0, y - r)
    x_max = min(width, x + r)
    y_max = min(height, y + r)
    
    # Extract the ROI
    roi = image[y_min:y_max, x_min:x_max]
    
    if roi.size == 0:
        return False  # ROI is empty, can't analyze
    
    # Convert ROI to grayscale
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Apply edge detection to find cracks
    edges = cv2.Canny(gray_roi, 50, 150)
    
    # Count edge pixels within the egg region
    # Create a circular mask for the egg
    mask = np.zeros_like(gray_roi)
    cv2.circle(
        mask, 
        (x - x_min, y - y_min),  # Adjust center to ROI coordinates
        int(radius * 0.8),  # Focus on central area where cracks are more visible
        255, 
        -1
    )
    
    # Count edge pixels within the egg mask
    edge_count = cv2.countNonZero(cv2.bitwise_and(edges, edges, mask=mask))
    
    # Normalize by the egg area
    egg_area = np.pi * (radius * 0.8) ** 2
    edge_density = edge_count / egg_area if egg_area > 0 else 0
    
    # Calculate texture features
    texture_score = calculate_texture_features(gray_roi, mask)
    
    # More precise threshold for edge density - eggs naturally have some edges
    # but broken eggs have significantly more edge pixels
    is_broken = edge_density > 0.08 and texture_score > 0.2
    
    return is_broken

def calculate_texture_features(gray_roi, mask):
    """
    Calculate texture features to help identify broken eggs.
    
    Args:
        gray_roi (numpy.ndarray): Grayscale ROI image
        mask (numpy.ndarray): Binary mask of the egg region
    
    Returns:
        float: Texture irregularity score
    """
    # Apply the mask to the ROI
    masked_roi = cv2.bitwise_and(gray_roi, gray_roi, mask=mask)
    
    # Skip if no pixels are masked
    non_zero = cv2.countNonZero(mask)
    if non_zero == 0:
        return 0
    
    # Calculate Laplacian variance (measure of texture)
    # Cracked eggs have higher Laplacian variance
    laplacian = cv2.Laplacian(masked_roi, cv2.CV_64F)
    laplacian_variance = laplacian.var()
    
    # Calculate gradient magnitude to detect sharp changes in intensity
    sobel_x = cv2.Sobel(masked_roi, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(masked_roi, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = cv2.magnitude(sobel_x, sobel_y)
    
    # Calculate standard deviation of gradient magnitude
    # Higher values indicate more texture variation (cracks)
    gradient_std = np.std(gradient_magnitude)
    
    # Create a normalized texture score
    texture_score = (laplacian_variance / 2000.0) + (gradient_std / 50.0)
    
    return min(texture_score, 1.0)  # Cap at 1.0

def convert_to_robot_coordinates(egg_center, egg_radius, image_shape, camera_params):
    """
    Convert egg position from image coordinates to robot-relative coordinates.
    
    Args:
        egg_center (tuple): (x, y) center of egg in image
        egg_radius (int): Radius of egg in pixels
        image_shape (tuple): (height, width) of image
        camera_params (dict): Camera parameters including position, fov, etc.
    
    Returns:
        tuple: (x, y) coordinates of the egg relative to robot in meters
    """
    # Image center
    img_height, img_width = image_shape[:2]
    img_center_x = img_width / 2
    img_center_y = img_height / 2
    
    # Calculate displacement from image center in pixels
    dx_pixels = egg_center[0] - img_center_x
    dy_pixels = egg_center[1] - img_center_y
    
    # Known information
    # - Standard egg size: ~45mm length, ~35mm diameter for medium egg
    
    # Estimate depth using egg size
    # Assuming standard egg medium size (~35mm diameter)
    STANDARD_EGG_DIAMETER_MM = 35.0
    
    # Convert egg radius from pixels to mm using the standard egg size
    # This helps estimate the distance from camera to egg
    mm_per_pixel = STANDARD_EGG_DIAMETER_MM / (2 * egg_radius)
    
    # Convert pixel displacement to mm
    dx_mm = dx_pixels * mm_per_pixel
    dy_mm = dy_pixels * mm_per_pixel
    
    # Assuming the camera is mounted on the robot at a known position
    # Default values if not provided in camera_params
    camera_height_mm = camera_params.get('height_mm', 150)    # Estimated camera height from ground
    camera_angle_rad = camera_params.get('angle_rad', 0.5)    # Camera tilt down angle in radians
    camera_offset_x_mm = camera_params.get('offset_x_mm', 0)  # Camera offset from robot center
    camera_offset_y_mm = camera_params.get('offset_y_mm', 0)  # Camera offset from robot center
    
    # Field of view
    fov_h_rad = camera_params.get('fov_h_rad', 1.05)  # Horizontal FOV in radians (~60 degrees)
    
    # Estimate depth using the egg size
    depth_mm = (STANDARD_EGG_DIAMETER_MM * camera_params.get('focal_length_pixels', img_width)) / (2 * egg_radius)
    
    # Adjust for camera angle (simple projection)
    ground_distance_mm = depth_mm * cos(camera_angle_rad)
    height_correction_mm = depth_mm * sin(camera_angle_rad)
    
    # Calculate 3D coordinates relative to camera
    cam_rel_x_mm = dx_mm
    cam_rel_y_mm = ground_distance_mm
    
    # Transform to robot coordinates (robot at origin)
    robot_rel_x_mm = cam_rel_x_mm + camera_offset_x_mm
    robot_rel_y_mm = cam_rel_y_mm + camera_offset_y_mm
    
    # Convert to meters for the final output
    robot_rel_x_m = robot_rel_x_mm / 1000.0
    robot_rel_y_m = robot_rel_y_mm / 1000.0
    
    return (robot_rel_x_m, robot_rel_y_m)

def analyze_eggs(image, centers, radii, camera_params=None):
    """
    Analyze detected eggs to determine if they're broken and calculate their positions.
    
    Args:
        image (numpy.ndarray): Original image
        centers (list): List of (x, y) center coordinates of detected eggs
        radii (list): List of radii corresponding to detected eggs
        camera_params (dict, optional): Camera parameters
    
    Returns:
        list: List of dict with egg information {broken: bool, coordX: float, coordY: float}
    """
    if camera_params is None:
        # Default camera parameters (estimates)
        camera_params = {
            'height_mm': 150,           # Height of camera from ground
            'angle_rad': 0.5,           # Camera tilt angle in radians (~30 degrees down)
            'offset_x_mm': 0,           # X offset from robot center
            'offset_y_mm': 0,           # Y offset from robot center
            'focal_length_pixels': 800, # Estimated focal length in pixels
            'fov_h_rad': 1.05           # Horizontal FOV in radians (~60 degrees)
        }
    
    egg_info_list = []
    
    for center, radius in zip(centers, radii):
        # First check if it's a valid egg
        if is_valid_egg(image, center, radius):
            # If it's a valid egg, check if it's broken
            is_broken = detect_broken_egg(image, center, radius)
            
            # Calculate egg position relative to robot
            coord_x, coord_y = convert_to_robot_coordinates(
                center, radius, image.shape, camera_params
            )
            
            # Create egg info dictionary
            egg_info = {
                'broken': bool(is_broken),  # Convert to native Python bool
                'coordX': float(round(coord_x, 3)),  # Convert to native Python float
                'coordY': float(round(coord_y, 3))   # Convert to native Python float
            }
            
            egg_info_list.append(egg_info)
    
    return egg_info_list