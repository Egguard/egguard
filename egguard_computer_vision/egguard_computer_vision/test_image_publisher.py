#!/usr/bin/env python3
# test_image_publisher.py - Publicador de imágenes de prueba para simulación
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
import glob
import time

class TestImagePublisher(Node):
    def __init__(self):
        super().__init__('test_image_publisher')
        
        # Basic parameters
        self.declare_parameter('publish_rate', 1.0)  # Hz
        self.declare_parameter('image_dir', '')      # Directory with test images
        self.declare_parameter('loop', True)         # Loop through images
        
        # Get parameters
        self.publish_rate = self.get_parameter('publish_rate').value
        self.image_dir = self.get_parameter('image_dir').value
        self.loop = self.get_parameter('loop').value
        
        # Bridge for image conversion
        self.bridge = CvBridge()
        
        # Image publisher
        self.publisher = self.create_publisher(
            Image,
            'camera/image_raw',  # Updated topic name to match main_node
            10)
            
        # Load image list
        self.images = []
        self.current_index = 0
        
        if self.image_dir and os.path.isdir(self.image_dir):
            # Find images in directory
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                self.images.extend(glob.glob(os.path.join(self.image_dir, ext)))
            
            if self.images:
                self.get_logger().info(f'Loaded {len(self.images)} images from {self.image_dir}')
            else:
                self.get_logger().warning(f'No images found in {self.image_dir}')
        else:
            self.get_logger().warning(f'Invalid directory: {self.image_dir}')
        
        # Create timer for periodic publishing
        period = 1.0 / self.publish_rate if self.publish_rate > 0 else 1.0
        self.timer = self.create_timer(period, self.publish_image)
        
        self.get_logger().info(f'Image publisher started (rate: {self.publish_rate} Hz)')
        
    def generate_synthetic_image(self, index):
        """Generate a synthetic image with eggs for testing"""
        # Image dimensions
        width, height = 640, 480
        
        # Create light gray background
        image = np.ones((height, width, 3), dtype=np.uint8) * 230
        
        # Add some shadow
        shadow_width = int(width / 2)
        shadow = np.ones((height, shadow_width, 3), dtype=np.uint8) * 210
        image[:, :shadow_width] = shadow
        
        # Generate 1-3 eggs based on index
        num_eggs = 1 + (index % 3)
        
        for i in range(num_eggs):
            # Position (reproducible based on index)
            x = (100 + (index * 50 + i * 123) % (width - 200))
            y = (100 + (index * 70 + i * 167) % (height - 200))
            
            # Size (within our detection parameters)
            major_axis = 30 + (index + i) % 20  # 30-50 pixels
            minor_axis = 20 + (index + i) % 15  # 20-35 pixels
            
            # Angle
            angle = (index * 30 + i * 45) % 180
            
            # Brown color (BGR)
            brown_b = 40 + (index + i) % 20
            brown_g = 50 + (index + i) % 30
            brown_r = 80 + (index + i) % 40
            color = (brown_b, brown_g, brown_r)
            
            # Draw egg (ellipse)
            cv2.ellipse(image, (x, y), (major_axis, minor_axis), angle, 0, 360, color, -1)
            
            # Add some texture
            cv2.ellipse(image, (x-5, y-5), (major_axis-10, minor_axis-10), angle, 0, 360, 
                        (brown_b+10, brown_g+10, brown_r+10), -1)
            
            # Randomly make some eggs appear broken
            if (index + i) % 3 == 0:
                # Add a crack or irregularity
                crack_points = []
                for j in range(5):
                    px = x + int(np.cos(angle * np.pi/180) * (j * 10))
                    py = y + int(np.sin(angle * np.pi/180) * (j * 10))
                    crack_points.append((px, py))
                cv2.polylines(image, [np.array(crack_points)], False, (0, 0, 0), 2)
        
        # Add slight noise
        noise = np.random.normal(0, 3, image.shape).astype(np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Slight blur
        image = cv2.GaussianBlur(image, (3, 3), 0)
        
        return image
        
    def publish_image(self):
        """Publish the next image from the list or generate a synthetic one"""
        try:
            # Use real images if available
            if self.images:
                if self.current_index >= len(self.images):
                    if self.loop:
                        self.current_index = 0
                    else:
                        self.get_logger().info('End of image sequence')
                        return
                
                image_path = self.images[self.current_index]
                self.get_logger().debug(f'Publishing image: {image_path}')
                
                image = cv2.imread(image_path)
                if image is None:
                    self.get_logger().warning(f'Could not read image: {image_path}')
                    self.current_index += 1
                    return
            else:
                # Generate synthetic image
                image = self.generate_synthetic_image(self.current_index)
                self.get_logger().debug(f'Publishing synthetic image #{self.current_index}')
            
            # Convert to ROS message
            msg = self.bridge.cv2_to_imgmsg(image, encoding='bgr8')
            
            # Add timestamp and frame_id
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'camera'
            
            # Publish
            self.publisher.publish(msg)
            
            # Increment index for next time
            self.current_index += 1
            
        except Exception as e:
            self.get_logger().error(f'Error publishing image: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    node = TestImagePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()