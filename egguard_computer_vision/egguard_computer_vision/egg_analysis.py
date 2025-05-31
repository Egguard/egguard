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
    Updated to work with YOLOv8n detections.
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
        
        # YOLOv8n confidence - eggs detected by YOLO are already validated
        self.yolo_confidence_threshold = 0.5
    
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
        Verify if the detected object is actually an egg.
        Since YOLOv8n already provides high-confidence egg detections,
        this method is simplified but maintains compatibility.
        
        Args:
            image (numpy.ndarray): Original image in BGR format
            center (tuple): (x, y) center coordinates of the potential egg
            radius (int): Radius of the detected potential egg
        
        Returns:
            bool: True if the object appears to be an egg, False otherwise
        """
        # Since YOLOv8n is trained specifically for eggs and has high accuracy,
        # we can be more confident in its detections. However, we still perform
        # basic sanity checks.
        
        # Basic bounds checking
        height, width = image.shape[:2]
        x, y = int(center[0]), int(center[1])
        
        # Check if center is within image bounds
        if x < 0 or x >= width or y < 0 or y >= height:
            self.log_debug(f"Egg center ({x}, {y}) is outside image bounds")
            return False
        
        # Check if radius is reasonable
        if radius < 5 or radius > min(width, height) // 4:
            self.log_debug(f"Egg radius {radius} is unreasonable for image size {width}x{height}")
            return False
        
        # Extract the region of interest (ROI) containing the potential egg
        r = int(radius * 1.2)  # Slightly larger ROI to ensure full egg coverage
        
        # Define ROI boundaries ensuring they're within image bounds
        x_min = max(0, x - r)
        y_min = max(0, y - r)
        x_max = min(width, x + r)
        y_max = min(height, y + r)
        
        # Extract the ROI
        roi = image[y_min:y_max, x_min:x_max]
        
        if roi.size == 0:
            self.log_debug(f"ROI is empty for egg at ({x}, {y}) with radius {radius}")
            return False
        
        # Create a circular mask for the egg region
        mask = np.zeros((y_max - y_min, x_max - x_min), dtype=np.uint8)
        
        # Adjust center to ROI coordinates
        roi_center_x = x - x_min
        roi_center_y = y - y_min
        
        # Ensure the center is within the ROI
        if roi_center_x < 0 or roi_center_x >= mask.shape[1] or roi_center_y < 0 or roi_center_y >= mask.shape[0]:
            self.log_debug(f"Adjusted center ({roi_center_x}, {roi_center_y}) is outside ROI")
            return False
        
        cv2.circle(mask, (roi_center_x, roi_center_y), int(radius * 0.9), 255, -1)
        
        # Convert ROI to HSV for color analysis
        try:
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        except cv2.error as e:
            self.log_debug(f"Error converting ROI to HSV: {e}")
            return False
        
        # Mask the ROI to focus on the egg region
        masked_hsv = cv2.bitwise_and(hsv_roi, hsv_roi, mask=mask)
        
        # Skip if no pixels are masked
        non_zero = cv2.countNonZero(mask)
        if non_zero == 0:
            self.log_debug("No pixels in mask")
            return False
        
        # Calculate color statistics in the egg region
        mean_hsv = cv2.mean(masked_hsv, mask=mask)
        
        # Relaxed egg color heuristics since YOLO already identified it as an egg
        # We mainly check for obviously non-egg colors (like very bright/saturated colors)
        saturation = mean_hsv[1]
        value = mean_hsv[2]
        
        # Very relaxed color checks - mainly to filter out obvious false positives
        # Eggs can vary significantly in color (white, brown, etc.)
        is_reasonable_color = (saturation < 200) and (value > 20) and (value < 250)
        
        if not is_reasonable_color:
            self.log_debug(f"Unreasonable color profile: S={saturation}, V={value}")
            return False
        
        # Since YOLO detected it as an egg, we trust its classification
        # and only reject obvious outliers
        return True

    def detect_broken_egg(self, image, center, radius):
        """
        Detect if an egg is broken by analyzing its contour regularity and texture.
        Optimized for YOLOv8n detections.
        
        Args:
            image (numpy.ndarray): Original image in BGR format
            center (tuple): (x, y) center coordinates of the egg
            radius (int): Radius of the detected egg
        
        Returns:
            bool: True if the egg appears to be broken, False otherwise
        """
        # First, verify this is actually an egg (simplified for YOLO detections)
        if not self.is_valid_egg(image, center, radius):
            self.log_debug("Object failed egg validation")
            return False
        
        # Extract the region of interest (ROI) containing the egg
        x, y = int(center[0]), int(center[1])
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
            self.log_debug("ROI is empty, cannot analyze for breaks")
            return False
        
        # Convert ROI to grayscale
        try:
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        except cv2.error as e:
            self.log_debug(f"Error converting ROI to grayscale: {e}")
            return False
        
        # Create a circular mask for the egg
        mask = np.zeros_like(gray_roi)
        roi_center_x = x - x_min
        roi_center_y = y - y_min
        
        # Ensure the center is within the ROI
        if (roi_center_x < 0 or roi_center_x >= mask.shape[1] or 
            roi_center_y < 0 or roi_center_y >= mask.shape[0]):
            self.log_debug("Center is outside ROI for break detection")
            return False
        
        cv2.circle(mask, (roi_center_x, roi_center_y), int(radius * 0.8), 255, -1)
        
        # Apply edge detection to find cracks
        edges = cv2.Canny(gray_roi, 50, 150)
        
        # Count edge pixels within the egg mask
        edge_count = cv2.countNonZero(cv2.bitwise_and(edges, edges, mask=mask))
        
        # Normalize by the egg area
        egg_area = np.pi * (radius * 0.8) ** 2
        edge_density = edge_count / egg_area if egg_area > 0 else 0
        
        # Calculate texture features
        texture_score = self._calculate_texture_features(gray_roi, mask)
        
        # Adjusted thresholds for YOLO detections
        # Since YOLO gives us more accurate egg boundaries, we can be more precise
        is_broken = edge_density > 0.06 and texture_score > 0.15
        
        self.log_debug(f"Break analysis - Edge density: {edge_density:.4f}, Texture score: {texture_score:.4f}, Broken: {is_broken}")
        
        return is_broken

    def _calculate_texture_features(self, gray_roi, mask):
        """
        Calculate texture features to help identify broken eggs.
        Enhanced for YOLOv8n detections.
        
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
            return 0.0
        
        try:
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
            
            # Calculate local binary pattern-like feature for additional texture analysis
            # This helps detect the irregular patterns typical of cracked eggs
            mean_intensity = cv2.mean(masked_roi, mask=mask)[0]
            intensity_variation = np.std(masked_roi[mask > 0]) if np.any(mask > 0) else 0
            
            # Create a more sophisticated texture score
            # Normalize components to reasonable ranges
            laplacian_component = min(laplacian_variance / 2000.0, 0.5)
            gradient_component = min(gradient_std / 50.0, 0.3)
            variation_component = min(intensity_variation / 100.0, 0.2)
            
            texture_score = laplacian_component + gradient_component + variation_component
            
            return min(texture_score, 1.0)  # Cap at 1.0
            
        except Exception as e:
            self.log_debug(f"Error calculating texture features: {e}")
            return 0.0

    def convert_to_robot_coordinates(self, egg_center, egg_radius, image_shape, camera_params=None):
        """
        Convert egg position from image coordinates to robot-relative coordinates.
        Enhanced for YOLOv8n detections with more accurate size estimation.
        
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
        
        # Improved distance estimation using YOLO's more accurate bounding boxes
        # YOLO provides better size estimates than Hough circles
        
        # Calculate mm per pixel using the standard egg size
        # Since YOLO gives more accurate egg boundaries, we can be more precise
        mm_per_pixel = self.STANDARD_EGG_DIAMETER_MM / (2 * egg_radius)
        
        # Apply a correction factor for YOLO detections (empirically determined)
        # YOLO tends to be more consistent in size estimation
        yolo_correction_factor = 1.1  # Adjust based on testing
        mm_per_pixel *= yolo_correction_factor
        
        # Convert pixel displacement to mm
        dx_mm = dx_pixels * mm_per_pixel
        dy_mm = dy_pixels * mm_per_pixel
        
        # Camera parameters
        camera_height_mm = camera_params.get('height_mm', 150)
        camera_angle_rad = camera_params.get('angle_rad', 0.5)
        camera_offset_x_mm = camera_params.get('offset_x_mm', 0)
        camera_offset_y_mm = camera_params.get('offset_y_mm', 0)
        fov_h_rad = camera_params.get('fov_h_rad', 1.05)
        
        # Improved depth estimation for YOLO detections
        focal_length_pixels = camera_params.get('focal_length_pixels', img_width * 0.8)
        depth_mm = (self.STANDARD_EGG_DIAMETER_MM * focal_length_pixels) / (2 * egg_radius * yolo_correction_factor)
        
        # Adjust for camera angle (more accurate projection)
        ground_distance_mm = depth_mm * cos(camera_angle_rad)
        height_correction_mm = depth_mm * sin(camera_angle_rad)
        
        # Calculate 3D coordinates relative to camera
        # Improved coordinate transformation
        cam_rel_x_mm = dx_mm * (ground_distance_mm / focal_length_pixels)
        cam_rel_y_mm = ground_distance_mm
        
        # Transform to robot coordinates (robot at origin)
        robot_rel_x_mm = cam_rel_x_mm + camera_offset_x_mm
        robot_rel_y_mm = cam_rel_y_mm + camera_offset_y_mm
        
        # Convert to meters for the final output
        robot_rel_x_m = robot_rel_x_mm / 1000.0
        robot_rel_y_m = robot_rel_y_mm / 1000.0
        
        self.log_debug(f"Coordinate conversion - Pixel: {egg_center}, Robot: ({robot_rel_x_m:.3f}, {robot_rel_y_m:.3f})")
        
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
        
        self.log_debug(f"World coordinate conversion - Robot: ({rel_x:.3f}, {rel_y:.3f}) -> World: ({world_x:.3f}, {world_y:.3f})")
        
        return (world_x, world_y)

    def analyze_eggs(self, image, centers, radii, camera_params=None):
        """
        Analyze detected eggs to determine if they're broken and calculate their positions.
        Optimized for YOLOv8n detections with improved validation and error handling.
        
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
        self.log_debug(f"Starting YOLO-based analysis of {len(centers)} eggs")
        
        # Ensure centers and radii have the same length
        if len(centers) != len(radii):
            self.log_debug(f"Mismatch between centers ({len(centers)}) and radii ({len(radii)})")
            # Truncate to the shorter length
            min_len = min(len(centers), len(radii))
            centers = centers[:min_len]
            radii = radii[:min_len]
        
        # Validate input data
        if not centers or not radii:
            self.log_debug("No eggs to analyze")
            return egg_info_list
        
        for i, (center, radius) in enumerate(zip(centers, radii)):
            try:
                # Process each egg separately
                self.log_debug(f"Processing YOLO egg #{i+1} at {center} with radius {radius}")
                
                # Basic validation of input data
                if not isinstance(center, (tuple, list)) or len(center) != 2:
                    self.log_debug(f"Invalid center format for egg #{i+1}: {center}")
                    continue
                
                if not isinstance(radius, (int, float)) or radius <= 0:
                    self.log_debug(f"Invalid radius for egg #{i+1}: {radius}")
                    continue
                
                # Since YOLO has already detected these as eggs with high confidence,
                # we can be more lenient in validation while still doing basic checks
                valid_egg = self.is_valid_egg(image, center, radius)
                self.log_debug(f"YOLO egg #{i+1} passed validation: {valid_egg}")
                
                if valid_egg:
                    # Check if the egg is broken
                    is_broken = self.detect_broken_egg(image, center, radius)
                    self.log_debug(f"YOLO egg #{i+1} broken status: {is_broken}")
                    
                    # Calculate egg position relative to robot
                    try:
                        robot_rel_x, robot_rel_y = self.convert_to_robot_coordinates(
                            center, radius, image.shape, camera_params
                        )
                        
                        # Convert to world coordinates
                        world_x, world_y = self.convert_to_world_coordinates((robot_rel_x, robot_rel_y))
                        
                        # Create egg info dictionary with both relative and world coordinates
                        egg_info = {
                            'coordX': float(round(robot_rel_x, 3)),  # Robot-relative X
                            'coordY': float(round(robot_rel_y, 3)),  # Robot-relative Y
                            'worldX': float(round(world_x, 3)),      # World X
                            'worldY': float(round(world_y, 3)),      # World Y
                            'broken': bool(is_broken)               # Convert to native Python bool
                        }
                        
                        # Print egg information with world coordinates
                        status = "BROKEN" if is_broken else "OK"
                        print(f"Egg #{i+1}: {status}, World Position: ({world_x:.3f}, {world_y:.3f})")
                        
                        egg_info_list.append(egg_info)
                        
                    except Exception as e:
                        self.log_debug(f"Error calculating coordinates for egg #{i+1}: {str(e)}")
                        # Add default info to maintain list consistency
                        egg_info = {
                            'coordX': 0.0,
                            'coordY': 0.0,
                            'worldX': 0.0,
                            'worldY': 0.0,
                            'broken': False
                        }
                        egg_info_list.append(egg_info)
                else:
                    self.log_debug(f"YOLO egg #{i+1} failed validation, skipping")
                    # Even though YOLO detected it, basic validation failed
                    # This might happen with edge cases or poor image quality
                    egg_info = {
                        'coordX': 0.0,
                        'coordY': 0.0,
                        'worldX': 0.0,
                        'worldY': 0.0,
                        'broken': False
                    }
                    egg_info_list.append(egg_info)
                    
            except Exception as e:
                self.log_debug(f"Error processing YOLO egg #{i+1}: {str(e)}")
                # Add default info for this egg to maintain the list length
                egg_info = {
                    'coordX': 0.0,
                    'coordY': 0.0,
                    'worldX': 0.0,
                    'worldY': 0.0,
                    'broken': False
                }
                egg_info_list.append(egg_info)
        
        self.log_debug(f"Finished YOLO-based analysis: {len(egg_info_list)} eggs processed")
        return egg_info_list