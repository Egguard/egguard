#!/usr/bin/env python3
# egg_detector.py - Enhanced egg detection logic
import cv2
import numpy as np
from typing import Dict, List, Tuple, Union, Any
import json
import os

class EggDetector:
    """
    Egg detector class that identifies eggs in images and returns their status and position.
    """
    def __init__(self):
        # Basic detection parameters
        self.min_area = 370         # Minimum egg area
        self.max_area = 1100        # Maximum egg area
        self.min_radius = 10        # Minimum egg radius
        self.max_radius = 30        # Maximum egg radius
        
    def detect_eggs(self, image: np.ndarray) -> dict:
        """
        Detect eggs in the image and return their status and position.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            dict: Detection results with broken status and coordinates
        """
        if image is None or image.size == 0:
            return {
                "broken": False,
                "coordX": 0.0,
                "coordY": 0.0
            }
            
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define color ranges for brown eggs
        lower_brown1 = np.array([0, 30, 40])
        upper_brown1 = np.array([20, 255, 220])
        lower_brown2 = np.array([20, 30, 40])
        upper_brown2 = np.array([40, 255, 220])
        
        # Create masks for brown colors
        mask1 = cv2.inRange(hsv, lower_brown1, upper_brown1)
        mask2 = cv2.inRange(hsv, lower_brown2, upper_brown2)
        mask_combined = cv2.bitwise_or(mask1, mask2)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        mask_cleaned = cv2.morphologyEx(mask_combined, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Process contours to find eggs
        eggs_found = []
        for contour in contours:
            # Calculate area
            area = cv2.contourArea(contour)
            if area < self.min_area or area > self.max_area:
                continue
                
            # Calculate minimal enclosing circle
            (x, y), radius = cv2.minEnclosingCircle(contour)
            if radius < self.min_radius or radius > self.max_radius:
                continue
                
            # Calculate circularity
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter + 1e-5)
            
            # Check if the egg appears broken (low circularity)
            is_broken = circularity < 0.5
            
            eggs_found.append({
                "broken": is_broken,
                "coordX": float(x),
                "coordY": float(y)
            })
        
        # If no eggs found, return default values
        if not eggs_found:
            return {
                "broken": False,
                "coordX": 0.0,
                "coordY": 0.0
            }
        
        # For now, return the first egg found
        # This can be modified later to handle multiple eggs
        return eggs_found[0]

# Example usage when run directly
if __name__ == "__main__":
    import sys
    import glob
    import os
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Egg Detector')
    parser.add_argument('path', nargs='?', help='Image path or directory')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--out', default='output_images', help='Output directory')
    parser.add_argument('--min-radius', type=int, default=10, help='Minimum egg radius')
    parser.add_argument('--max-radius', type=int, default=30, help='Maximum egg radius')
    parser.add_argument('--min-area', type=int, default=370, help='Minimum egg area')
    parser.add_argument('--max-area', type=int, default=1100, help='Maximum egg area')
    parser.add_argument('--border', type=int, default=40, help='Border size for distance transform')
    args = parser.parse_args()
    
    detector = EggDetector()
    
    # Output directory for results
    output_dir = args.out
    os.makedirs(output_dir, exist_ok=True)
    
    # Process a single image
    def process_single_image(image_path):
        print(f"Processing {image_path}")
        
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not read image: {image_path}")
            return
            
        # Display original image info
        print(f"Image dimensions: {image.shape}")
        cv2.imwrite(os.path.join(output_dir, os.path.basename(image_path)), image)
            
        # Detect eggs
        results = detector.detect_eggs(image)
        print(json.dumps(results, indent=2))
        
        # Save image with detections
        base_name = os.path.basename(image_path)
        output_path = os.path.join(output_dir, f"detected_{base_name}")
        cv2.imwrite(output_path, image)
        print(f"Annotated image saved to: {output_path}")
        
        # Try to display image if display is available
        try:
            cv2.imshow("Egg Detection", image)
            cv2.waitKey(0)
        except:
            print("Could not display image on screen, but it was saved to disk")
    
    # Process a directory of images
    def process_directory(directory_path):
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(glob.glob(os.path.join(directory_path, ext)))
        
        if not image_files:
            print(f"No images found in {directory_path}")
            return
            
        print(f"Found {len(image_files)} images to process")
        for image_path in image_files:
            process_single_image(image_path)
    
    # Run example
    if args.path:
        path = args.path
        if os.path.isdir(path):
            process_directory(path)
        elif os.path.isfile(path):
            process_single_image(path)
        else:
            print(f"Path does not exist: {path}")
    else:
        print("Usage: python egg_detector.py [image_path or directory] [--debug] [--out output_directory]")
        # If no arguments, try to use default directory
        default_dir = "images/eggs"
        if os.path.exists(default_dir):
            print(f"Using default directory: {default_dir}")
            process_directory(default_dir)
        else:
            print(f"Default directory {default_dir} does not exist")
            # Process at least one example image
            print("Creating a test image...")
            # Create a synthetic image with a brown oval for testing
            test_image = np.ones((400, 600, 3), dtype=np.uint8) * 230  # Light grayish background
            # Add some shadow (gray) in one part
            shadow = np.ones((400, 300, 3), dtype=np.uint8) * 210
            test_image[:, :300] = shadow
            # Add a brown egg
            cv2.ellipse(test_image, (300, 200), (100, 150), 30, 0, 360, (42, 42, 165), -1)  # Brown oval
            test_path = os.path.join(output_dir, "test_egg.jpg")
            cv2.imwrite(test_path, test_image)
            process_single_image(test_path)
            cv2.destroyAllWindows()