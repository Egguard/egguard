import cv2
import numpy as np
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PoseStamped, TransformStamped
from math import atan2, cos, sin, sqrt, pi
from tf_transformations import euler_from_quaternion

class EggAnalyzer:
    """
    Class for analyzing eggs in images to determine if they're broken
    and calculate their positions in world coordinates.
    """
    
    def __init__(self):
        # Default camera parameters (can be updated later)
        self.default_camera_params = {
            'height_mm': 150,           # Height of camera from ground
            'angle_rad': 0.5,           # Camera tilt angle in radians (~30 degrees down)
            'offset_x_mm': 0,           # X offset from robot center
            'offset_y_mm': 0,           # Y offset from robot center
            'focal_length_pixels': 800, # Estimated focal length in pixels
            'fov_h_rad': 1.05           # Horizontal FOV in radians (~60 degrees)
        }
        
        # Standard egg size for calculations
        self.STANDARD_EGG_DIAMETER_MM = 35.0
        
        # Current robot pose from odometry (will be updated)
        self.robot_pose = {
            'x': 0.0,
            'y': 0.0,
            'z': 0.0,
            'roll': 0.0,
            'pitch': 0.0,
            'yaw': 0.0
        }
        
        # Logger - will be set by the node
        self.logger = None
    
    def set_logger(self, logger):
        """Set a logger for debugging purposes"""
        self.logger = logger
    
    def log_debug(self, message):
        """Helper method to log debug messages if logger is available"""
        if self.logger:
            self.logger.debug(message)
    
    def update_robot_pose(self, odom_msg):
        """
        Update robot pose from odometry message.
        
        Args:
            odom_msg: ROS2 nav_msgs/msg/Odometry message
        """
        # Extract position
        self.robot_pose['x'] = odom_msg.pose.pose.position.x
        self.robot_pose['y'] = odom_msg.pose.pose.position.y
        self.robot_pose['z'] = odom_msg.pose.pose.position.z
        
        # Extract orientation (convert quaternion to Euler angles)
        orientation_q = odom_msg.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        roll, pitch, yaw = euler_from_quaternion(orientation_list)
        
        self.robot_pose['roll'] = roll
        self.robot_pose['pitch'] = pitch
        self.robot_pose['yaw'] = yaw
    
    def is_valid_egg(self, image, center, radius):
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
            self.log_debug(f"ROI is empty for egg at ({x}, {y}) with radius {radius}")
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
        is_egg_color = (saturation < 150) and (value > 50)
        
        # Check shape compactness (eggs are roughly circular/oval)
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        binary = cv2.bitwise_and(binary, mask)
        
        # Find contours in the masked binary image
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            # No contours found, but we'll be more lenient here
            # If there's a detected circle with reasonable color, let's assume it's an egg
            return is_egg_color
        
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
        is_egg_shape = 0.5 < circularity < 1.0
        
        # Combine color and shape evidence
        return is_egg_color and is_egg_shape

    def detect_broken_egg(self, image, center, radius):
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
        if not self.is_valid_egg(image, center, radius):
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
        texture_score = self._calculate_texture_features(gray_roi, mask)
        
        # More precise threshold for edge density - eggs naturally have some edges
        # but broken eggs have significantly more edge pixels
        is_broken = edge_density > 0.08 and texture_score > 0.2
        
        return is_broken

    def _calculate_texture_features(self, gray_roi, mask):
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

    def convert_to_robot_coordinates(self, egg_center, egg_radius, image_shape, camera_params=None):
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
        # Use default camera parameters if none provided
        if camera_params is None:
            camera_params = self.default_camera_params
            
        # Image center
        img_height, img_width = image_shape[:2]
        img_center_x = img_width / 2
        img_center_y = img_height / 2
        
        # Calculate displacement from image center in pixels
        dx_pixels = egg_center[0] - img_center_x
        dy_pixels = egg_center[1] - img_center_y
        
        # Convert egg radius from pixels to mm using the standard egg size
        # This helps estimate the distance from camera to egg
        mm_per_pixel = self.STANDARD_EGG_DIAMETER_MM / (2 * egg_radius)
        
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
        depth_mm = (self.STANDARD_EGG_DIAMETER_MM * camera_params.get('focal_length_pixels', img_width)) / (2 * egg_radius)
        
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

    def convert_to_world_coordinates(self, robot_rel_coords):
        """
        Convert robot-relative coordinates to world coordinates using the robot's pose.
        
        Args:
            robot_rel_coords (tuple): (x, y) coordinates relative to robot
        
        Returns:
            tuple: (x, y) coordinates in world frame
        """
        # Extract robot position and orientation
        robot_x = self.robot_pose['x']
        robot_y = self.robot_pose['y']
        robot_yaw = self.robot_pose['yaw']
        
        # Get robot-relative coordinates
        rel_x, rel_y = robot_rel_coords
        
        # Apply rotation based on robot's orientation
        world_x = robot_x + rel_x * cos(robot_yaw) - rel_y * sin(robot_yaw)
        world_y = robot_y + rel_x * sin(robot_yaw) + rel_y * cos(robot_yaw)
        
        return (world_x, world_y)

    def analyze_eggs(self, image, centers, radii, camera_params=None):
        """
        Analyze detected eggs to determine if they're broken and calculate their positions.
        
        Args:
            image (numpy.ndarray): Original image
            centers (list): List of (x, y) center coordinates of detected eggs
            radii (list): List of radii corresponding to detected eggs
            camera_params (dict, optional): Camera parameters
        
        Returns:
            list: List of dict with egg information {broken: bool, coordX: float, coordY: float, worldX: float, worldY: float}
        """
        if camera_params is None:
            camera_params = self.default_camera_params
        
        egg_info_list = []
        
        # Log how many eggs we're analyzing
        self.log_debug(f"Starting analysis of {len(centers)} eggs")
        
        # Ensure centers and radii have the same length
        if len(centers) != len(radii):
            self.log_debug(f"Mismatch between centers ({len(centers)}) and radii ({len(radii)})")
            # Truncate to the shorter length
            min_len = min(len(centers), len(radii))
            centers = centers[:min_len]
            radii = radii[:min_len]
        
        for i, (center, radius) in enumerate(zip(centers, radii)):
            try:
                # Process each egg separately
                self.log_debug(f"Processing egg #{i+1} at {center} with radius {radius}")
                
                # First check if it's a valid egg
                valid_egg = self.is_valid_egg(image, center, radius)
                self.log_debug(f"Egg #{i+1} valid: {valid_egg}")
                
                if valid_egg:
                    # If it's a valid egg, check if it's broken
                    is_broken = self.detect_broken_egg(image, center, radius)
                    self.log_debug(f"Egg #{i+1} broken: {is_broken}")
                    
                    # Calculate egg position relative to robot
                    robot_rel_x, robot_rel_y = self.convert_to_robot_coordinates(
                        center, radius, image.shape, camera_params
                    )
                    
                    # Convert to world coordinates
                    world_x, world_y = self.convert_to_world_coordinates((robot_rel_x, robot_rel_y))
                    
                    # Create egg info dictionary
                    egg_info = {
                        'coordX': float(round(robot_rel_x, 3)),  # Robot-relative X
                        'coordY': float(round(robot_rel_y, 3)),  # Robot-relative Y
                        'worldX': float(round(world_x, 3)),      # World X
                        'worldY': float(round(world_y, 3)),      # World Y
                        'broken': bool(is_broken)               # Convert to native Python bool
                    }
                    
                    # Print egg information in EXACTLY the same format as simulation.py
                    status = "BROKEN" if is_broken else "OK"
                    print(f"Egg #{i+1}: {status}, Position: ({robot_rel_x:.3f}, {robot_rel_y:.3f})")
                    
                    egg_info_list.append(egg_info)
                else:
                    self.log_debug(f"Egg #{i+1} is not a valid egg, skipping")
            except Exception as e:
                self.log_debug(f"Error processing egg #{i+1}: {str(e)}")
                # Add default info for this egg to maintain the list length matching detected eggs
                egg_info = {
                    'coordX': 0.0,
                    'coordY': 0.0,
                    'worldX': 0.0,
                    'worldY': 0.0,
                    'broken': False
                }
                egg_info_list.append(egg_info)
        
        self.log_debug(f"Finished analyzing {len(egg_info_list)} eggs")
        return egg_info_list