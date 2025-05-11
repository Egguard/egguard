#!/usr/bin/env python3

import rclpy
import cv2
import numpy as np
import time
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import Int32
from rclpy.node import Node
from rclpy.qos import ReliabilityPolicy, QoSProfile

# Import our egg detection module
from egguard_computer_vision.egg_detector import (
    preprocess_image,
    detect_eggs,
    draw_detections
)

class EggDetectionNode(Node):
    """
    ROS2 Node for egg detection using computer vision.
    Subscribes to camera images and processes them to detect chicken eggs.
    """
    def __init__(self, processing_interval=3.0, debug_mode=True):
        """
        Initialize the egg detection node.
        
        Args:
            processing_interval (float): Time interval between image processing in seconds
            debug_mode (bool): Whether to show debug visualization
        """
        super().__init__('egg_detection_node')
        
        # Initialize CV bridge for ROS-OpenCV image conversion
        self.bridge = CvBridge()
        
        # Create subscription to camera topic
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.camera_callback,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        )
        
        # Publisher for egg count
        self.egg_count_pub = self.create_publisher(
            Int32,
            '/egguard/egg_count',
            10
        )
        
        # Time tracking for processing interval
        self.last_process_time = time.time()
        self.processing_interval = processing_interval
        self.debug_mode = debug_mode
        
        self.get_logger().info('Egg detection node initialized')
        self.get_logger().info(f'Processing interval set to {processing_interval} seconds')
    
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
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
            
            # Preprocess the image (resize, filter, etc.)
            processed_img = preprocess_image(cv_image)
            
            # Detect eggs in the image
            egg_centers, egg_radii = detect_eggs(processed_img)
            
            # Publish egg count
            count_msg = Int32()
            count_msg.data = len(egg_centers)
            self.egg_count_pub.publish(count_msg)
            
            self.get_logger().info(f'Detected {len(egg_centers)} eggs')
            
            # If debug mode is enabled, show visualization
            if self.debug_mode:
                # Draw detection results on a copy of the original image
                result_img = draw_detections(cv_image.copy(), egg_centers, egg_radii)
                
                # Show the result
                cv2.imshow("Egg Detection", result_img)
                cv2.waitKey(1)
                
        except CvBridgeError as e:
            self.get_logger().error(f'CV Bridge error: {e}')
        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

def main(args=None):
    """Main function to initialize and run the egg detection node."""
    rclpy.init(args=args)
    
    # Create the egg detection node with 3-second processing interval
    egg_detection_node = EggDetectionNode(processing_interval=3.0)
    
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