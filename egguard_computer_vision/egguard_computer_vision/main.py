#!/usr/bin/env python3

import rclpy
import cv2
import numpy as np
import time
import json
import requests
import os
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import Int32, String
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import ReliabilityPolicy, QoSProfile

# Import our egg detection and analysis classes
from egguard_computer_vision.egg_detector import EggDetector
from egguard_computer_vision.egg_analysis import EggAnalyzer

class EggDetectionNode(Node):
    """
    ROS2 Node for egg detection using computer vision.
    Subscribes to camera images and processes them to detect chicken eggs.
    """
    def __init__(self, processing_interval=1.0, debug_mode=True, backend_url=None):
        """
        Initialize the egg detection node.
        
        Args:
            processing_interval (float): Time interval between image processing in seconds
            debug_mode (bool): Whether to show debug visualization
            backend_url (str): URL of the backend API to send egg data
        """
        super().__init__('egg_detection_node')
        
        # Initialize CV bridge for ROS-OpenCV image conversion
        self.bridge = CvBridge()
        
        # Initialize egg detector and analyzer
        self.egg_detector = EggDetector()
        self.egg_analyzer = EggAnalyzer()
        
        # Create subscription to camera topic
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw', #change to /image when real robot camera is used
            self.camera_callback,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        )
        
        # Create subscription to odometry topic to get robot pose
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        )
        
        # Create subscription to camera info topic (for intrinsic parameters)
        self.camera_info_sub = self.create_subscription(
            String,  # Using String for simplicity, could be CameraInfo type
            '/camera/camera_info',
            self.camera_info_callback,
            10
        )
        
        # Publisher for egg count
        self.egg_count_pub = self.create_publisher(
            Int32,
            '/egguard/egg_count',
            10
        )
        
        # Publisher for detailed egg data
        self.egg_data_pub = self.create_publisher(
            String,
            '/egguard/egg_data',
            10
        )
        
        # Time tracking for processing interval
        self.last_process_time = time.time()
        self.processing_interval = processing_interval
        self.debug_mode = debug_mode
        self.backend_url = backend_url or os.environ.get('EGGUARD_BACKEND_URL', 'http://localhost:8080/eggs')
        
        # Flag to check if we've received odometry data
        self.odom_received = False
        
        # Camera parameters (will be updated from camera_info if available)
        self.camera_params = {
            'height_mm': 150,           # Height of camera from ground
            'angle_rad': 0.5,           # Camera tilt angle in radians (~30 degrees down)
            'offset_x_mm': 0,           # X offset from robot center
            'offset_y_mm': 69,          # Y offset from robot center (half of robot length)
            'focal_length_pixels': 800, # Estimated focal length in pixels
            'fov_h_rad': 1.05           # Horizontal FOV in radians (~60 degrees)
        }
        
        self.get_logger().info('Egg detection node initialized')
        self.get_logger().info(f'Processing interval set to {processing_interval} seconds')
        self.get_logger().info(f'Backend URL: {self.backend_url}')
    
    def odom_callback(self, msg):
        """
        Callback function for odometry subscription.
        Updates robot pose in the egg analyzer.
        
        Args:
            msg (Odometry): Odometry message containing robot pose
        """
        try:
            # Update the robot pose in the egg analyzer
            self.egg_analyzer.update_robot_pose(msg)
            
            # Mark that we've received odometry data
            self.odom_received = True
            
            self.get_logger().debug(f'Updated robot pose: x={msg.pose.pose.position.x:.2f}, y={msg.pose.pose.position.y:.2f}')
        except Exception as e:
            self.get_logger().error(f'Error updating robot pose: {e}')
    
    def camera_info_callback(self, msg):
        """
        Callback function for camera info subscription.
        Updates camera parameters from camera info message.
        
        Args:
            msg (String): Camera info message (simplified for this example)
        """
        try:
            # In a real implementation, you would parse CameraInfo message
            # For simplicity, we're assuming String with JSON
            info_dict = json.loads(msg.data)
            
            if 'focal_length_pixels' in info_dict:
                self.camera_params['focal_length_pixels'] = info_dict['focal_length_pixels']
            if 'fov_h_rad' in info_dict:
                self.camera_params['fov_h_rad'] = info_dict['fov_h_rad']
                
            self.get_logger().info('Updated camera parameters from camera info')
        except Exception as e:
            self.get_logger().error(f'Error parsing camera info: {e}')
    
    def camera_callback(self, data):
        """
        Callback function for the image subscription.
        Processes images at the specified interval.
        
        Args:
            data (Image): The incoming image message
        """
        current_time = time.time()
        
        # Process image only if enough time has passed since last processing
        if current_time - self.last_process_time >= self.processing_interval:
            self.last_process_time = current_time
            self.process_image(data)
    
    def process_image(self, img_msg):
        """
        Process the incoming image to detect eggs.
        
        Args:
            img_msg (Image): The ROS image message
        """
        try:
            # Check if we have odometry data
            if not self.odom_received:
                self.get_logger().warning('No odometry data received yet. World coordinates will be inaccurate.')
            
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
            
            # Preprocess the image and detect eggs using the detector class
            preprocessed_img = self.egg_detector.preprocess_image(cv_image)
            egg_centers, egg_radii = self.egg_detector.detect_eggs(preprocessed_img)
            
            # Analyze eggs (detect if broken and calculate coordinates) using the analyzer class
            egg_info_list = self.egg_analyzer.analyze_eggs(cv_image, egg_centers, egg_radii, self.camera_params)
            
            # Ensure egg_info_list matches the number of detected eggs
            # This line is crucial - it ensures our egg info list is complete
            if len(egg_info_list) != len(egg_centers):
                self.get_logger().warning(f"Mismatch between detected eggs ({len(egg_centers)}) and analyzed eggs ({len(egg_info_list)})")
            
            # Publish egg count
            count_msg = Int32()
            count_msg.data = len(egg_centers)
            self.egg_count_pub.publish(count_msg)
            
            # Publish detailed egg data
            egg_data_msg = String()
            egg_data_msg.data = json.dumps(egg_info_list)
            self.egg_data_pub.publish(egg_data_msg)
            
            # Send data to backend API
            self.send_to_backend(egg_info_list)
            
            self.get_logger().info(f'Detected {len(egg_centers)} eggs')
            
            # If debug mode is enabled, show visualization
            if self.debug_mode:
                # Draw detection results on a copy of the original image
                result_img = self.draw_debug_image(cv_image.copy(), egg_centers, egg_radii, egg_info_list)
                
                # Show the result
                cv2.imshow("Egg Detection", result_img)
                cv2.waitKey(1)
                
        except CvBridgeError as e:
            self.get_logger().error(f'CV Bridge error: {e}')
        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')
    
    def draw_debug_image(self, image, centers, radii, egg_info_list):
        """
        Draw debug information on the image with format matching simulation.py.
        
        Args:
            image (numpy.ndarray): Image to draw on
            centers (list): List of egg centers
            radii (list): List of egg radii
            egg_info_list (list): List of egg information dictionaries
            
        Returns:
            numpy.ndarray: Image with debug information
        """
        # Use the detector's draw_detections method to visualize the eggs
        result = self.egg_detector.draw_detections(image, centers, radii, egg_info_list)
        
        # Add robot position and orientation
        robot_x = self.egg_analyzer.robot_pose['x']
        robot_y = self.egg_analyzer.robot_pose['y']
        robot_yaw = self.egg_analyzer.robot_pose['yaw']
        cv2.putText(
            result,
            f"Robot: ({robot_x:.2f}, {robot_y:.2f}, {robot_yaw:.2f} rad)",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )
        
        return result
    
    def send_to_backend(self, egg_info_list):
        """
        Send egg information to the backend API.
        
        Args:
            egg_info_list (list): List of egg information dictionaries
        """
        if not self.backend_url:
            return
            
        try:
            for egg_info in egg_info_list:
                # Make API request to backend
                response = requests.post(
                    self.backend_url,
                    json=egg_info,
                    headers={'Content-Type': 'application/json'},
                    timeout=1.0  # Short timeout to avoid blocking
                )
                
                if response.status_code == 200:
                    self.get_logger().debug(f'Successfully sent egg data: {egg_info}')
                else:
                    self.get_logger().warning(f'Failed to send egg data: {response.status_code}')
        except requests.exceptions.RequestException as e:
            self.get_logger().warning(f'Error sending data to backend: {e}')

def main(args=None):
    """Main function to initialize and run the egg detection node."""
    rclpy.init(args=args)
    
    # Get backend URL from environment variable or use default
    backend_url = os.environ.get('EGGUARD_BACKEND_URL', 'http://localhost:8080/api/v1/robots/1/eggs')
    
    # Create the egg detection node with specified parameters
    egg_detection_node = EggDetectionNode(
        debug_mode=True,
        backend_url=backend_url
    )
    
    try:
        # Spin the node
        rclpy.spin(egg_detection_node)
    except KeyboardInterrupt:
        egg_detection_node.get_logger().info('Node stopped by keyboard interrupt')
    except Exception as e:
        egg_detection_node.get_logger().error(f'Node stopped due to exception: {e}')
    finally:
        # Clean up
        egg_detection_node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()