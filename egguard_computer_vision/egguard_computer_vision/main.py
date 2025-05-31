#!/usr/bin/env python3

import rclpy
import cv2
import numpy as np
import time
import json
import requests
import os
import threading
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
    def __init__(self, processing_interval=1.0, debug_mode=True, backend_url=None, model_path=None):
        """
        Initialize the egg detection node.
        
        Args:
            processing_interval (float): Time interval between image processing in seconds
            debug_mode (bool): Whether to show debug visualization
            backend_url (str): URL of the backend API to send egg data
            model_path (str): Path to the YOLO model file
        """
        super().__init__('egg_detection_node')
        
        # Initialize CV bridge for ROS-OpenCV image conversion
        self.bridge = CvBridge()
        
        # Flag to indicate if detector is ready
        self.detector_ready = False
        self.egg_detector = None
        self.egg_analyzer = None
        
        # Store initialization parameters
        self.model_path = model_path
        
        # Log que el nodo se está inicializando
        self.get_logger().info('Egg detection node initializing...')
        
        # Initialize detector in background thread to avoid blocking
        self.detector_thread = threading.Thread(target=self._initialize_detector)
        self.detector_thread.daemon = True
        self.detector_thread.start()
        
        # Create subscription to camera topic with BEST_EFFORT reliability
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',  # Changed to the correct topic name
            self.camera_callback,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        )
        self.get_logger().info('Subscribed to camera topic: /camera/image_raw')
        
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
        self.last_image_time = time.time()  # Initialize last_image_time
        self.processing_interval = processing_interval
        self.debug_mode = debug_mode
        self.backend_url = backend_url or os.environ.get('EGGUARD_BACKEND_URL', 'http://localhost:8081/eggs')
        
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
        
        # Publisher for processed images (for RViz visualization)
        self.processed_image_pub = self.create_publisher(
            Image,
            '/egguard/processed_image',
            10
        )

        # Publisher for YOLO raw detections
        self.yolo_raw_pub = self.create_publisher(
            Image,
            '/egguard/yolo_raw',
           10
        )
        
        self.get_logger().info('Egg detection node subscribers and publishers initialized')
        self.get_logger().info(f'Processing interval set to {processing_interval} seconds')
        self.get_logger().info(f'Backend URL: {self.backend_url}')
        self.get_logger().info('YOLO detector initialization in progress...')
        
        # Create a timer to check if we're receiving images
        self.create_timer(5.0, self._check_image_reception)
    
    def _initialize_detector(self):
        """
        Initialize the egg detector in a separate thread to avoid blocking the main thread.
        """
        try:
            self.get_logger().info('Starting YOLO detector initialization...')
            
            # Initialize egg detector with model path
            self.egg_detector = EggDetector(model_path=self.model_path)
            self.get_logger().info('Successfully initialized EggDetector with YOLO model')
            
            # Log model info
            model_info = self.egg_detector.get_model_info()
            self.get_logger().info(f'Model info: {model_info}')
            
            # Initialize egg analyzer
            self.egg_analyzer = EggAnalyzer()
            self.egg_analyzer.set_logger(self.get_logger())  # Set logger for debugging
            self.get_logger().info('Successfully initialized EggAnalyzer')
            
            # Mark detector as ready
            self.detector_ready = True
            self.get_logger().info('✅ YOLO detector and analyzer are ready!')
            
        except Exception as e:
            self.get_logger().error(f'❌ Failed to initialize EggDetector: {e}')
            self.detector_ready = False
            # No hacer raise aquí para no terminar el nodo
    
    def odom_callback(self, msg):
        """
        Callback function for odometry subscription.
        Updates robot pose in the egg analyzer.
        
        Args:
            msg (Odometry): Odometry message containing robot pose
        """
        try:
            # Check if analyzer is ready before updating pose
            if self.detector_ready and self.egg_analyzer is not None:
                # Update the robot pose in the egg analyzer
                self.egg_analyzer.update_robot_pose(msg)
                
                # Mark that we've received odometry data
                self.odom_received = True
                
                self.get_logger().debug(f'Updated robot pose: x={msg.pose.pose.position.x:.2f}, y={msg.pose.pose.position.y:.2f}')
            else:
                self.get_logger().debug('Odometry received but analyzer not ready yet')
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
    
    def camera_callback(self, msg):
        """Callback for processing camera images."""
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Detect eggs using YOLO
            results = self.egg_detector.model(cv_image, conf=self.egg_detector.confidence_threshold, device=self.egg_detector.device, verbose=False)
            
            # Analyze eggs for visualization
            centers = []
            radii = []
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    for box in boxes:
                        x1, y1, x2, y2 = box
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        radius = max((x2 - x1), (y2 - y1)) / 2
                        centers.append((center_x, center_y))
                        radii.append(radius)
            
            # Get egg information
            egg_info_list = self.egg_analyzer.analyze_eggs(cv_image, centers, radii)
            
            # Log analysis results for debugging
            for i, egg_info in enumerate(egg_info_list):
                status = "BROKEN" if egg_info.get('broken', False) else "OK"
                world_x = egg_info.get('worldX', 0.0)
                world_y = egg_info.get('worldY', 0.0)
                self.get_logger().debug(f"Egg #{i+1}: {status}, World Position: ({world_x:.3f}, {world_y:.3f})")
            
            # Show visualization with egg information
            self.egg_detector.show_live_yolo_processing(cv_image, results, egg_info_list)
            
            # Process image at specified interval
        current_time = time.time()
        if current_time - self.last_process_time >= self.processing_interval:
                self.process_image(cv_image)
            self.last_process_time = current_time
            
        except Exception as e:
            self.get_logger().error(f"Error in camera callback: {str(e)}")
            import traceback
            self.get_logger().error(f"Traceback: {traceback.format_exc()}")
    
    def process_image(self, img_msg):
        """
        Process the incoming image to detect eggs.
        This method handles the actual data processing and backend communication.
        
        Args:
            img_msg: Either a ROS Image message or a numpy array (OpenCV image)
        """
        try:
            # Double check that detector is ready
            if not self.detector_ready or self.egg_detector is None or self.egg_analyzer is None:
                self.get_logger().warning('Detector or analyzer not ready, skipping image processing')
                return
            
            # Check if we have odometry data
            if not self.odom_received:
                self.get_logger().warning('No odometry data received yet. World coordinates will be inaccurate.')
            
            # Convert ROS image to OpenCV format if needed
            if isinstance(img_msg, np.ndarray):
                cv_image = img_msg
            else:
                try:
            cv_image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
                except CvBridgeError as e:
                    self.get_logger().error(f'CV Bridge error: {e}')
                    return
            
            # Detect eggs using the YOLO-based detector
            egg_centers, egg_radii = self.egg_detector.detect_eggs(cv_image)
            
            # Log detection details
            self.get_logger().info(f'YOLO detected {len(egg_centers)} potential eggs')
            
            # Analyze eggs (detect if broken and calculate coordinates)
            egg_info_list = self.egg_analyzer.analyze_eggs(cv_image, egg_centers, egg_radii, self.camera_params)
            
            # Log analysis results
            for i, egg_info in enumerate(egg_info_list):
                self.get_logger().info(
                    f"Egg #{i+1}: {'BROKEN' if egg_info['broken'] else 'OK'}, "
                    f"Position: ({egg_info['coordX']:.3f}, {egg_info['coordY']:.3f})"
                )
            
            # Filter out invalid eggs
            valid_eggs = []
            valid_centers = []
            valid_radii = []
            valid_info = []
            
            for i, (center, radius, info) in enumerate(zip(egg_centers, egg_radii, egg_info_list)):
                if info['coordX'] != 0.0 or info['coordY'] != 0.0 or info['worldX'] != 0.0 or info['worldY'] != 0.0:
                    valid_eggs.append(i)
                    valid_centers.append(center)
                    valid_radii.append(radius)
                    valid_info.append(info)
                else:
                    self.get_logger().debug(f'Egg #{i+1} failed validation - not a valid egg')
            
            # Update lists to only include valid eggs
            egg_centers = valid_centers
            egg_radii = valid_radii
            egg_info_list = valid_info
            
            self.get_logger().info(f'After validation: {len(egg_centers)} valid eggs')
            
            # Publish egg count
            count_msg = Int32()
            count_msg.data = len(egg_centers)
            self.egg_count_pub.publish(count_msg)
            
            # Publish detailed egg data
            egg_data_msg = String()
            egg_data_msg.data = json.dumps(egg_info_list)
            self.egg_data_pub.publish(egg_data_msg)
            
            # Send data to backend
            if egg_info_list:
                self.send_to_backend(egg_info_list)
                
            # Publish processed images for RViz visualization
            try:
                # Create YOLO raw visualization
                yolo_viz = self.egg_detector.show_live_yolo_processing(cv_image, egg_info_list)
                yolo_msg = self.bridge.cv2_to_imgmsg(yolo_viz, encoding="bgr8")
                self.yolo_raw_pub.publish(yolo_msg)
    
                # Create final processed image
                result_img = self.egg_detector.draw_detections(cv_image, egg_centers, egg_radii, egg_info_list)
                result_img = self.add_debug_info(result_img)
    
                # Publish processed image
                processed_msg = self.bridge.cv2_to_imgmsg(result_img, encoding="bgr8")
                self.processed_image_pub.publish(processed_msg)
                
                self.get_logger().debug(f'Published YOLO raw and processed images to topics')
    
            except CvBridgeError as e:
                self.get_logger().error(f'Error publishing processed images: {e}')
            except Exception as e:
                self.get_logger().error(f'Error during image visualization: {e}')
                
        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')
            import traceback
            self.get_logger().error(f'Traceback: {traceback.format_exc()}')
    
    def add_debug_info(self, image):
        """
        Add additional debug information to the image.
        
        Args:
            image (numpy.ndarray): Image to add debug info to
            
        Returns:
            numpy.ndarray: Image with additional debug information
        """
        # Only add debug info if analyzer is ready
        if self.detector_ready and self.egg_analyzer is not None:
            # Add robot position and orientation
            robot_x = self.egg_analyzer.robot_pose['x']
            robot_y = self.egg_analyzer.robot_pose['y']
            robot_yaw = self.egg_analyzer.robot_pose['yaw']
            cv2.putText(
                image,
                f"Robot: ({robot_x:.2f}, {robot_y:.2f}, {robot_yaw:.2f} rad)",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )
        else:
            # Show that detector is not ready yet
            cv2.putText(
                image,
                "Detector initializing...",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )
        
        # Add processing interval info
        cv2.putText(
            image,
            f"Interval: {self.processing_interval}s",
            (10, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
        
        return image
    
    def send_to_backend(self, egg_info_list):
        """
        Send egg information to the backend API.
        Use world coordinates instead of robot-relative coordinates.
        
        Args:
            egg_info_list (list): List of egg information dictionaries
        """
        if not self.backend_url:
            return
            
        try:
            for egg_info in egg_info_list:
                # Create new dict with world coordinates instead of robot-relative coordinates
                backend_data = {
                    'coordX': egg_info['worldX'],  # Use world coordinates instead of robot-relative
                    'coordY': egg_info['worldY'],  # Use world coordinates instead of robot-relative
                    'broken': egg_info['broken']
                }
                
                # Make API request to backend
                response = requests.post(
                    self.backend_url,
                    json=backend_data,
                    headers={'Content-Type': 'application/json'},
                    timeout=1.0  # Short timeout to avoid blocking
                )
                
                if response.status_code == 200:
                    self.get_logger().debug(f'Successfully sent egg data with world coordinates: {backend_data}')
                else:
                    self.get_logger().warning(f'Failed to send egg data: {response.status_code}')
        except requests.exceptions.RequestException as e:
            self.get_logger().warning(f'Error sending data to backend: {e}')

    def _check_image_reception(self):
        """Check if we're receiving images and log a warning if not."""
        if not hasattr(self, 'last_image_time'):
            self.get_logger().warning('No images received yet. Check if camera is publishing to /camera/image_raw topic')
            return
            
        time_since_last_image = time.time() - self.last_image_time
        if time_since_last_image > 5.0:
            self.get_logger().warning(f'No images received for {time_since_last_image:.1f} seconds. Check camera topic /camera/image_raw')

def main(args=None):
    """Main function to initialize and run the egg detection node."""
    rclpy.init(args=args)
    
    # Get backend URL from environment variable or use default
    backend_url = os.environ.get('EGGUARD_BACKEND_URL', 'http://localhost:8081/api/v1/robots/1/eggs')
    
    # Get model path from environment variable or use None (will use default search paths)
    model_path = os.environ.get('EGGUARD_MODEL_PATH', None)
    
    # Create the egg detection node with specified parameters
    try:
        egg_detection_node = EggDetectionNode(
            debug_mode=True,
            backend_url=backend_url,
            model_path=model_path
        )
        
        egg_detection_node.get_logger().info('🚀 Starting egg detection node...')
        egg_detection_node.get_logger().info('Node is running. YOLO detector will be ready shortly...')
        
        # Spin the node
        rclpy.spin(egg_detection_node)
        
    except KeyboardInterrupt:
        print('Node stopped by keyboard interrupt')
    except Exception as e:
        print(f'Node stopped due to exception: {e}')
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        try:
            egg_detection_node.destroy_node()
        except:
            pass
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()