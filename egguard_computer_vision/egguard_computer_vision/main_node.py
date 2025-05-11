#!/usr/bin/env python3
# main_node.py - ROS2 node for egg detection
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import time
import json
from threading import Lock

# Import our egg detector
from .egg_detector import EggDetector

class EggDetectionNode(Node):
    """
    ROS2 node for egg detection that leverages the EggDetector module.
    Subscribes to camera images and publishes detection results.
    """
    def __init__(self):
        super().__init__('egg_detection_node')
        
        # Initialize CV bridge for ROS-OpenCV conversion
        self.bridge = CvBridge()
        
        # Initialize egg detector
        self.detector = EggDetector()
        
        # Create subscription to camera image
        self.image_subscription = self.create_subscription(
            Image,
            'camera/image_raw',
            self.image_callback,
            10)
        self.get_logger().info('Subscribed to topic: camera/image_raw')
        
        # Create publisher for detection results
        self.results_publisher = self.create_publisher(
            String,
            'egg_detection/results',
            10)
        
        # Control variables
        self.last_detection_time = time.time()
        self.processing_lock = Lock()
        self.detection_interval = 3.0  # seconds
        
        self.get_logger().info('Egg detection node initialized')
    
    def image_callback(self, msg):
        """
        Callback for processing images from camera topic.
        
        Args:
            msg (sensor_msgs.msg.Image): ROS Image message
        """
        # Check if enough time has passed since last detection
        current_time = time.time()
        if (current_time - self.last_detection_time) < self.detection_interval:
            return
        
        # Try to acquire the lock (non-blocking)
        if not self.processing_lock.acquire(blocking=False):
            self.get_logger().debug('Processing already in progress, skipping frame')
            return
        
        try:
            self.last_detection_time = current_time
            
            # Convert ROS message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Detect eggs
            results = self.detector.detect_eggs(cv_image)
            
            # Format results for backend
            formatted_results = self.format_results_for_backend(results)
            
            # Publish results
            self.publish_results(formatted_results)
            
            # Log results
            self.get_logger().info(f'Processed detection results: {formatted_results}')
            
        except Exception as e:
            self.get_logger().error(f'Error in detection: {str(e)}')
        finally:
            # Always release the lock
            self.processing_lock.release()
    
    def format_results_for_backend(self, detection_results):
        """
        Format detection results to match backend requirements.
        
        Args:
            detection_results: Results from egg detector
            
        Returns:
            dict: Formatted results with broken status and coordinates
        """
        # Default result if no eggs detected
        result = {
            "broken": False,
            "coordX": 0.0,
            "coordY": 0.0
        }
        
        # If eggs were detected, update with actual values
        if detection_results and "eggs" in detection_results:
            # For now, we'll use the first detected egg
            # This can be modified based on how egg_detector.py provides the results
            first_egg = detection_results["eggs"][0]
            result.update({
                "broken": first_egg.get("broken", False),
                "coordX": first_egg.get("x", 0.0),
                "coordY": first_egg.get("y", 0.0)
            })
        
        return result
    
    def publish_results(self, results):
        """
        Publish detection results to topic.
        
        Args:
            results (dict): Detection results dictionary
        """
        msg = String()
        msg.data = json.dumps(results)
        self.results_publisher.publish(msg)

def main(args=None):
    """Main function to run the node"""
    rclpy.init(args=args)
    node = EggDetectionNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()