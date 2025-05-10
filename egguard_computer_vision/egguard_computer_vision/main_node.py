#!/usr/bin/env python3
# main_node.py - ROS2 node for egg detection
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String, Float32
from cv_bridge import CvBridge
import json
import time
import os
import cv2
from threading import Lock

# Import our egg detector
from egg_detector import EggDetector

class EggDetectionNode(Node):
    """
    ROS2 node for egg detection that leverages the EggDetector module.
    Subscribes to camera images and publishes detection results.
    """
    def __init__(self):
        super().__init__('egg_detection_node')
        
        # Declare node parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                ('detection_interval', 3.0),  # Time interval between detections (seconds)
                ('debug_mode', False),        # Enable debug mode for detector
                ('save_images', False),       # Save processed images to disk
                ('output_dir', 'egg_detection_output'),  # Output directory for saved images
                ('min_radius', 10),           # Minimum egg radius
                ('max_radius', 30),           # Maximum egg radius
                ('min_area', 370),            # Minimum egg area
                ('max_area', 1100),           # Maximum egg area
                ('border_size', 40)           # Border size for distance transform
            ]
        )
        
        # Get parameters
        self.configure_parameters()
        
        # Create output directory if necessary
        self.setup_output_directory()
        
        # Initialize CV bridge for ROS-OpenCV conversion
        self.bridge = CvBridge()
        
        # Initialize egg detector with parameters
        self.setup_detector()
        
        # Create subscriptions
        self.create_subscriptions()
        
        # Create publishers
        self.create_publishers()
            
        # Control variables
        self.last_detection_time = time.time()
        self.processing_lock = Lock()
        self.processed_count = 0
        
        self.get_logger().info('Egg detection node initialized')
        self.get_logger().info(f'Detection interval: {self.detection_interval} seconds')
    
    def configure_parameters(self):
        """Configure node parameters from ROS param server"""
        self.detection_interval = self.get_parameter('detection_interval').value
        self.debug_mode = self.get_parameter('debug_mode').value
        self.save_images = self.get_parameter('save_images').value
        self.output_dir = self.get_parameter('output_dir').value
        self.min_radius = self.get_parameter('min_radius').value
        self.max_radius = self.get_parameter('max_radius').value
        self.min_area = self.get_parameter('min_area').value
        self.max_area = self.get_parameter('max_area').value
        self.border_size = self.get_parameter('border_size').value
    
    def setup_output_directory(self):
        """Set up output directory for saving images if enabled"""
        if self.save_images:
            os.makedirs(self.output_dir, exist_ok=True)
            self.get_logger().info(f'Processed images will be saved to: {self.output_dir}')
    
    def setup_detector(self):
        """Initialize the egg detector with parameters"""
        self.detector = EggDetector(debug=self.debug_mode)
        self.detector.set_size_parameters(
            min_radius=self.min_radius,
            max_radius=self.max_radius,
            min_area=self.min_area,
            max_area=self.max_area
        )
        self.detector.set_border_size(self.border_size)
        self.get_logger().info('Egg detector initialized with parameters')
    
    def create_subscriptions(self):
        """Create all node subscriptions"""
        # Subscription to raw camera image
        self.image_subscription = self.create_subscription(
            Image,
            'image_raw',
            self.image_callback,
            10)
        self.get_logger().info('Subscribed to topic: image_raw')
    
    def create_publishers(self):
        """Create all node publishers"""
        # Publisher for detection results as JSON
        self.results_publisher = self.create_publisher(
            String,
            'egg_detection/results',
            10)
        
        # Publisher for processed image
        self.image_publisher = self.create_publisher(
            Image,
            'egg_detection/image',
            10)
        
        # Publisher for count as separate topic for easy monitoring
        self.count_publisher = self.create_publisher(
            Float32,
            'egg_detection/count',
            10)
        
        # Publisher for processing time metrics
        self.timing_publisher = self.create_publisher(
            Float32,
            'egg_detection/processing_time',
            10)
        
        self.get_logger().info('Publishers created')
    
    def should_process_image(self):
        """Determine if an image should be processed based on time interval"""
        current_time = time.time()
        return (current_time - self.last_detection_time) >= self.detection_interval
    
    def image_callback(self, msg):
        """
        Callback for processing images from topic.
        
        Args:
            msg (sensor_msgs.msg.Image): ROS Image message
        """
        # Check if enough time has passed since last detection
        if not self.should_process_image():
            return
        
        # Try to acquire the lock (non-blocking)
        if not self.processing_lock.acquire(blocking=False):
            self.get_logger().debug('Processing already in progress, skipping frame')
            return
        
        try:
            self.last_detection_time = time.time()
            self.processed_count += 1
            
            # Process image and publish results
            self.process_and_publish_image(msg)
            
        except Exception as e:
            self.get_logger().error(f'Error in detection: {str(e)}')
        finally:
            # Always release the lock
            self.processing_lock.release()
    
    def process_and_publish_image(self, msg):
        """
        Process image and publish results.
        
        Args:
            msg (sensor_msgs.msg.Image): ROS Image message
        """
        # Convert ROS message to OpenCV image
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        # Log image info
        self.get_logger().debug(f'Processing image #{self.processed_count}: {cv_image.shape}')
        
        # Detect eggs and measure time
        start_time = time.time()
        results, annotated_image = self.detector.detect_eggs(cv_image)
        processing_time = time.time() - start_time
        
        # Publish all results
        self.publish_results(results, annotated_image, msg.header, processing_time)
        
        # Save image if configured
        self.save_image_if_enabled(annotated_image)
        
        # Log results
        self.get_logger().info(f'Detected {results["count"]} eggs in {processing_time:.2f} seconds')
    
    def publish_results(self, results, annotated_image, header, processing_time):
        """
        Publish all detection results to relevant topics.
        
        Args:
            results (dict): Detection results dictionary
            annotated_image (numpy.ndarray): Annotated image
            header (std_msgs.msg.Header): Original message header
            processing_time (float): Processing time in seconds
        """
        # Publish results as JSON
        json_msg = String()
        json_msg.data = json.dumps(results)
        self.results_publisher.publish(json_msg)
        
        # Publish processed image
        img_msg = self.bridge.cv2_to_imgmsg(annotated_image, encoding='bgr8')
        img_msg.header = header  # Maintain the same header
        self.image_publisher.publish(img_msg)
        
        # Publish count as separate topic
        count_msg = Float32()
        count_msg.data = float(results["count"])
        self.count_publisher.publish(count_msg)
        
        # Publish processing time
        time_msg = Float32()
        time_msg.data = processing_time
        self.timing_publisher.publish(time_msg)
    
    def save_image_if_enabled(self, image):
        """
        Save processed image to disk if enabled.
        
        Args:
            image (numpy.ndarray): Image to save
        """
        if self.save_images:
            # Generate unique filename with timestamp
            filename = f'egg_detection_{self.processed_count:04d}.jpg'
            filepath = os.path.join(self.output_dir, filename)
            cv2.imwrite(filepath, image)
            self.get_logger().debug(f'Saved image to: {filepath}')
    
    def process_image_file(self, image_path):
        """
        Process image from file (useful for CLI operation).
        
        Args:
            image_path (str): Path to image file
            
        Returns:
            tuple: (results dict, annotated image)
        """
        try:
            # Read image with OpenCV
            cv_image = cv2.imread(image_path)
            if cv_image is None:
                self.get_logger().error(f'Could not read image: {image_path}')
                return None, None
            
            # Detect eggs
            results, annotated_image = self.detector.detect_eggs(cv_image)
            
            # Save image if configured
            if self.save_images:
                base_name = os.path.basename(image_path)
                output_path = os.path.join(self.output_dir, f"detected_{base_name}")
                cv2.imwrite(output_path, annotated_image)
                self.get_logger().info(f'Image saved to: {output_path}')
            
            return results, annotated_image
            
        except Exception as e:
            self.get_logger().error(f'Error processing image {image_path}: {str(e)}')
            return None, None

def process_cli_args(args, node):
    """
    Process command line arguments for direct image processing.
    
    Args:
        args: Command line arguments
        node: EggDetectionNode instance
    """
    import argparse
    import glob
    
    parser = argparse.ArgumentParser(description='Egg Detector')
    parser.add_argument('images', nargs='+', help='Image paths or directories')
    parser.add_argument('--save', action='store_true', help='Save processed images')
    
    try:
        cli_args = parser.parse_args(args[1:])
        
        # Configure saving
        node.save_images = cli_args.save or node.save_images
        
        # Process each provided path
        for path in cli_args.images:
            if os.path.isfile(path):
                # Process individual file
                node.get_logger().info(f'Processing image: {path}')
                results, annotated_image = node.process_image_file(path)
                if results:
                    print(json.dumps(results, indent=2))
                    
                    # Try to display image
                    try:
                        cv2.imshow("Egg Detection", annotated_image)
                        cv2.waitKey(0)
                    except:
                        node.get_logger().info('Could not display image on screen')
                        
            elif os.path.isdir(path):
                # Process directory
                node.get_logger().info(f'Processing directory: {path}')
                
                # Find images in directory
                image_files = []
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                    image_files.extend(glob.glob(os.path.join(path, ext)))
                
                if not image_files:
                    node.get_logger().warning(f'No images found in {path}')
                    continue
                    
                # Process each image
                for img_path in image_files:
                    node.get_logger().info(f'Processing image: {img_path}')
                    results, _ = node.process_image_file(img_path)
                    if results:
                        print(f"{img_path}: {results['count']} eggs")
            else:
                node.get_logger().error(f'Invalid path: {path}')
        
        # Close windows
        try:
            cv2.destroyAllWindows()
        except:
            pass
    except Exception as e:
        node.get_logger().error(f'Error in CLI mode: {str(e)}')
        return False
        
    return True

def main(args=None):
    """Main function to run the node"""
    rclpy.init(args=args)
    node = EggDetectionNode()
    
    # If command line arguments were provided to process images
    if args and len(args) > 1:
        if process_cli_args(args, node):
            # CLI processing successful, clean up and exit
            node.destroy_node()
            rclpy.shutdown()
            return
    
    # Normal ROS mode
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    import sys
    main(sys.argv)