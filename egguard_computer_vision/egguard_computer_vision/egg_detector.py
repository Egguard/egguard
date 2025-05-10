#!/usr/bin/env python3
# egg_detector.py - Enhanced egg detection logic
import cv2
import numpy as np
from typing import Dict, List, Tuple, Union, Any
import json
import os

class EggDetector:
    """
    Enhanced egg detector class that utilizes multiple detection techniques
    to identify brown eggs on a white/gray background.
    """
    def __init__(self, debug=False):
        # Configurable detection parameters
        self.min_area = 370         # Minimum egg area
        self.max_area = 1100        # Maximum egg area
        self.min_radius = 10        # Minimum egg radius
        self.max_radius = 30        # Maximum egg radius
        self.min_circularity = 0.5  # Minimum circularity (0-1)
        self.min_ratio = 0.5        # Minimum axis ratio
        self.max_ratio = 2.0        # Maximum axis ratio
        self.border_size = 40       # Border size for distance transform
        self.debug = debug          # Debug mode to visualize steps
        self.debug_dir = "debug_images"  # Directory for debug images
        
        if self.debug:
            os.makedirs(self.debug_dir, exist_ok=True)
    
    def set_size_parameters(self, min_radius=None, max_radius=None, min_area=None, max_area=None):
        """Set size parameters for egg detection"""
        if min_radius is not None and min_radius != '':
            self.min_radius = int(min_radius)
        if max_radius is not None and max_radius != '':
            self.max_radius = int(max_radius)
        if min_area is not None and min_area != '':
            self.min_area = int(min_area)
        if max_area is not None and max_area != '':
            self.max_area = int(max_area)
            
    def set_border_size(self, border_size):
        """Set border size parameter for distance transform"""
        self.border_size = int(border_size)
        
    def save_debug_image(self, name, image):
        """Save a debug image if debug mode is enabled"""
        if self.debug:
            path = os.path.join(self.debug_dir, f"{name}.jpg")
            cv2.imwrite(path, image)
            print(f"Debug: Saved image {name} to {path}")
            
    def show_debug_image(self, name, image):
        """Show a debug image if debug mode is enabled"""
        if self.debug:
            cv2.imshow(name, image)
            cv2.waitKey(100)  # Short pause to update window
    
    def detect_eggs_distance_transform(self, image: np.ndarray) -> Tuple[List[Dict], np.ndarray]:
        """
        Detect eggs using distance transform method adapted from the example code.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Tuple of (list of egg data, annotated image)
        """
        if image is None or image.size == 0:
            return [], image
            
        # Create copy for visualization
        output = image.copy()
        height, width = image.shape[:2]
        
        if self.debug:
            self.save_debug_image("01_original", image)
            self.show_debug_image("Original", image)
            
        # Convert to HSV and extract value channel
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        if self.debug:
            self.save_debug_image("02_hsv", hsv)
            self.show_debug_image("HSV", hsv)
            
        # Apply threshold on value channel
        _, bw = cv2.threshold(hsv[:, :, 2], 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        if self.debug:
            self.save_debug_image("03_threshold", bw)
            self.show_debug_image("Threshold", bw)
            
        # Apply morphological closing
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        morph = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
        if self.debug:
            self.save_debug_image("04_morphology", morph)
            self.show_debug_image("Morphology", morph)
            
        # Apply distance transform
        dist = cv2.distanceTransform(morph, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        if self.debug:
            # Normalize for visualization
            dist_viz = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            self.save_debug_image("05_distance_transform", dist_viz)
            self.show_debug_image("Distance Transform", dist_viz)
            
        # Add border and create template
        border_size = self.border_size
        distborder = cv2.copyMakeBorder(dist, border_size, border_size, border_size, border_size,
                                        cv2.BORDER_CONSTANT | cv2.BORDER_ISOLATED, 0)
        
        # Create elliptical template
        gap = 10
        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                           (2 * (border_size - gap) + 1, 2 * (border_size - gap) + 1))
        kernel2 = cv2.copyMakeBorder(kernel2, gap, gap, gap, gap,
                                    cv2.BORDER_CONSTANT | cv2.BORDER_ISOLATED, 0)
        
        # Apply distance transform to template
        distTempl = cv2.distanceTransform(kernel2, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        
        # Template matching
        nxcor = cv2.matchTemplate(distborder, distTempl, cv2.TM_CCOEFF_NORMED)
        
        # Find peaks
        mn, mx, _, _ = cv2.minMaxLoc(nxcor)
        _, peaks = cv2.threshold(nxcor, mx * 0.5, 255, cv2.THRESH_BINARY)
        peaks8u = cv2.convertScaleAbs(peaks)
        
        if self.debug:
            self.save_debug_image("06_peaks", peaks8u)
            self.show_debug_image("Peaks", peaks8u)
            
        # Find contours in peaks
        contours, _ = cv2.findContours(peaks8u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if self.debug:
            contours_img = output.copy()
            cv2.drawContours(contours_img, contours, -1, (0, 255, 0), 2)
            self.save_debug_image("07_all_contours", contours_img)
            self.show_debug_image("All Contours", contours_img)
            print(f"Found {len(contours)} initial contours")
        
        # Process each contour
        eggs_data = []
        valid_contours = []
        
        for i, contour in enumerate(contours):
            # Calculate minimal enclosing circle
            (x, y), radius = cv2.minEnclosingCircle(contour)
            radius = int(radius)
            
            # Skip if radius is outside range
            if radius < self.min_radius or radius > self.max_radius:
                continue
                
            # Calculate bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Process contours with enough points for ellipse fitting
            if len(contour) >= 5:
                # Fit ellipse
                ellipse = cv2.fitEllipse(contour)
                (center, axes, angle) = ellipse
                
                # Calculate coordinates
                coord_x, coord_y = int(center[0]), int(center[1])
                ax1, ax2 = int(axes[0]) - 2, int(axes[1]) - 2
                
                # Calculate area
                area = cv2.contourArea(contour)
                
                # Skip if area is outside range
                if area < self.min_area or area > self.max_area:
                    continue
                
                # Calculate axis ratio
                ratio = max(ax1, ax2) / (min(ax1, ax2) + 1e-5)
                if ratio < self.min_ratio or ratio > self.max_ratio:
                    continue
                    
                # Calculate circularity
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter * perimeter + 1e-5)
                if circularity < self.min_circularity:
                    continue
                
                # Verify color
                mask_contour = np.zeros((height, width), dtype=np.uint8)
                cv2.drawContours(mask_contour, [contour], 0, 255, -1)
                mean_color = cv2.mean(hsv, mask=mask_contour)
                mean_h, mean_s, mean_v = mean_color[0], mean_color[1], mean_color[2]
                
                # Check if color is in brown range (hue between 0-40 typically for brown)
                is_brown = ((0 <= mean_h <= 40) and (mean_s >= 20))
                
                if not is_brown and not self.debug:  # In debug mode, accept all shapes to see what's detected
                    continue
                
                # Store valid contour
                valid_contours.append(contour)
                
                # Draw on output image
                cv2.ellipse(output, ellipse, (0, 255, 0), 2)
                cv2.putText(output, f"#{len(eggs_data)+1}", (coord_x, coord_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Store egg data
                egg_info = {
                    "id": len(eggs_data) + 1,
                    "position": {
                        "x": coord_x,
                        "y": coord_y
                    },
                    "size": {
                        "width": w,
                        "height": h,
                        "radius": radius
                    },
                    "shape": {
                        "major_axis": ax1,
                        "minor_axis": ax2,
                        "angle": angle
                    },
                    "area": float(area),
                    "color": {
                        "h": float(mean_h),
                        "s": float(mean_s),
                        "v": float(mean_v)
                    },
                    "confidence": float(circularity * 0.7 + is_brown * 0.3)  # Weight shape and color
                }
                
                eggs_data.append(egg_info)
        
        if self.debug:
            valid_contours_img = image.copy()
            cv2.drawContours(valid_contours_img, valid_contours, -1, (0, 255, 0), 2)
            self.save_debug_image("08_valid_contours", valid_contours_img)
            self.show_debug_image("Valid Contours", valid_contours_img)
            self.save_debug_image("09_final_result", output)
            self.show_debug_image("Final Result", output)
            print(f"Detected {len(eggs_data)} eggs")
            
        return eggs_data, output
    
    def detect_eggs_color_based(self, image: np.ndarray) -> Tuple[List[Dict], np.ndarray]:
        """
        Detect eggs using color-based method from the original implementation.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Tuple of (list of egg data, annotated image)
        """
        if image is None or image.size == 0:
            return [], image
            
        # Create copy for visualization
        output = image.copy()
        height, width = image.shape[:2]
        
        if self.debug:
            self.save_debug_image("10_original_color", image)
            self.show_debug_image("Original Color", image)
            
        # Apply denoising
        denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        if self.debug:
            self.save_debug_image("11_denoised", denoised)
            self.show_debug_image("Denoised", denoised)
            
        # Convert to HSV
        hsv = cv2.cvtColor(denoised, cv2.COLOR_BGR2HSV)
        if self.debug:
            self.save_debug_image("12_hsv", hsv)
            self.show_debug_image("HSV", hsv)
            
        # Define color ranges for brown eggs
        # Range 1: Reddish browns
        lower_brown1 = np.array([0, 30, 40])
        upper_brown1 = np.array([20, 255, 220])
        mask1 = cv2.inRange(hsv, lower_brown1, upper_brown1)
        
        # Range 2: Yellowish browns
        lower_brown2 = np.array([20, 30, 40])
        upper_brown2 = np.array([40, 255, 220])
        mask2 = cv2.inRange(hsv, lower_brown2, upper_brown2)
        
        # Combine masks
        mask_combined = cv2.bitwise_or(mask1, mask2)
        if self.debug:
            self.save_debug_image("13_mask_combined", mask_combined)
            self.show_debug_image("Combined Mask", mask_combined)
            
        # Apply morphological operations
        kernel = np.ones((5, 5), np.uint8)
        mask_closed = cv2.morphologyEx(mask_combined, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask_opened = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel, iterations=1)
        if self.debug:
            self.save_debug_image("14_morph_mask", mask_opened)
            self.show_debug_image("Morphology Mask", mask_opened)
            
        # Find contours
        contours, _ = cv2.findContours(mask_opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if self.debug:
            contours_img = output.copy()
            cv2.drawContours(contours_img, contours, -1, (0, 255, 0), 2)
            self.save_debug_image("15_color_contours", contours_img)
            self.show_debug_image("Color Contours", contours_img)
            print(f"Found {len(contours)} color contours")
            
        # Process each contour
        eggs_data = []
        valid_contours = []
        
        for i, contour in enumerate(contours):
            # Filter by area
            area = cv2.contourArea(contour)
            if area < self.min_area or area > self.max_area:
                continue
                
            # Need at least 5 points for ellipse fitting
            if len(contour) < 5:
                continue
                
            # Fit ellipse
            ellipse = cv2.fitEllipse(contour)
            (center, axes, angle) = ellipse
            
            # Extract parameters
            coord_x, coord_y = int(center[0]), int(center[1])
            ax1, ax2 = int(axes[0]), int(axes[1])
            
            # Calculate axis ratio
            ratio = max(ax1, ax2) / (min(ax1, ax2) + 1e-5)
            if ratio < self.min_ratio or ratio > self.max_ratio:
                continue
                
            # Calculate circularity
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter + 1e-5)
            if circularity < self.min_circularity:
                continue
                
            # Store valid contour
            valid_contours.append(contour)
            
            # Calculate bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Draw on output
            cv2.ellipse(output, ellipse, (255, 0, 0), 2)
            cv2.putText(output, f"#{len(eggs_data)+1}", (coord_x, coord_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Store egg data
            egg_info = {
                "id": len(eggs_data) + 1,
                "position": {
                    "x": coord_x,
                    "y": coord_y
                },
                "size": {
                    "width": w,
                    "height": h
                },
                "shape": {
                    "major_axis": ax1,
                    "minor_axis": ax2,
                    "angle": angle
                },
                "area": float(area),
                "circularity": float(circularity),
                "confidence": float(circularity)
            }
            
            eggs_data.append(egg_info)
            
        if self.debug:
            valid_contours_img = image.copy()
            cv2.drawContours(valid_contours_img, valid_contours, -1, (0, 255, 0), 2)
            self.save_debug_image("16_valid_color_contours", valid_contours_img)
            self.show_debug_image("Valid Color Contours", valid_contours_img)
            print(f"Detected {len(eggs_data)} eggs by color")
            
        return eggs_data, output
        
    def detect_eggs_combined(self, image: np.ndarray) -> Tuple[Dict[str, Any], np.ndarray]:
        """
        Combined egg detection using both methods.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Tuple of (results dictionary, annotated image)
        """
        if image is None or image.size == 0:
            return {"count": 0, "eggs": [], "success": False, "error": "Empty or invalid image"}, None
            
        # Create output image
        output = image.copy()
        
        # Method 1: Distance transform
        eggs_dt, output_dt = self.detect_eggs_distance_transform(image)
        
        # Method 2: Color based
        eggs_color, output_color = self.detect_eggs_color_based(image)
        
        # Combine results (merge overlapping detections)
        all_eggs = eggs_dt.copy()
        
        # Check each egg from color-based detection
        for egg_color in eggs_color:
            # Check if this egg is already detected by distance transform
            x_color, y_color = egg_color["position"]["x"], egg_color["position"]["y"]
            is_duplicate = False
            
            for egg_dt in eggs_dt:
                x_dt, y_dt = egg_dt["position"]["x"], egg_dt["position"]["y"]
                # Calculate distance between centers
                distance = np.sqrt((x_color - x_dt)**2 + (y_color - y_dt)**2)
                
                # If centers are close, consider them the same egg
                if distance < 30:  # Threshold for considering duplicates
                    is_duplicate = True
                    break
                    
            # If not a duplicate, add to results
            if not is_duplicate:
                egg_color["id"] = len(all_eggs) + 1  # Update ID
                all_eggs.append(egg_color)
        
        # Draw all eggs on final output
        for i, egg in enumerate(all_eggs):
            x, y = egg["position"]["x"], egg["position"]["y"]
            
            # Draw either an ellipse if shape data is available
            if "shape" in egg:
                center = (x, y)
                axes = (egg["shape"]["major_axis"], egg["shape"]["minor_axis"])
                angle = egg["shape"]["angle"]
                cv2.ellipse(output, (center, axes, angle), (0, 255, 0), 2)
            else:
                # Or a circle if only position is available
                radius = int(np.sqrt(egg["area"] / np.pi))
                cv2.circle(output, (x, y), radius, (0, 255, 0), 2)
                
            # Add ID text
            cv2.putText(output, f"#{i+1}", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Create results dictionary
        results = {
            "count": len(all_eggs),
            "eggs": all_eggs,
            "success": True
        }
        
        if self.debug:
            self.save_debug_image("17_final_combined", output)
            self.show_debug_image("Final Combined", output)
            print(f"Total detected eggs: {len(all_eggs)}")
            
        return results, output
        
    def detect_eggs(self, image: np.ndarray) -> Tuple[Dict[str, Any], np.ndarray]:
        """
        Main egg detection function that uses the combined approach.
        
        Args:
            image: Input image (numpy array in BGR format)
            
        Returns:
            Tuple of (results dictionary, annotated image)
        """
        return self.detect_eggs_combined(image)
        
    def process_image(self, image_data, encoding="bgr8") -> Dict[str, Any]:
        """
        Process an image and return detection results
        
        Args:
            image_data: Image data (numpy array or path)
            encoding: Image encoding (bgr8, rgb8, etc.)
            
        Returns:
            Dict with detection results
        """
        # If it's a file path
        if isinstance(image_data, str):
            image = cv2.imread(image_data)
            if image is None:
                return {"count": 0, "eggs": [], "success": False, "error": f"Could not read image: {image_data}"}
        # If it's a numpy array
        elif isinstance(image_data, np.ndarray):
            image = image_data.copy()
            # Convert RGB to BGR if needed
            if encoding.lower() == "rgb8":
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            return {"count": 0, "eggs": [], "success": False, "error": "Unsupported image format"}
        
        # Perform detection
        results, annotated_image = self.detect_eggs(image)
        
        return results

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
    
    detector = EggDetector(debug=args.debug)
    detector.set_size_parameters(args.min_radius, args.max_radius, args.min_area, args.max_area)
    detector.set_border_size(args.border)
    
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
        results, annotated_image = detector.detect_eggs(image)
        print(json.dumps(results, indent=2))
        
        # Save image with detections
        base_name = os.path.basename(image_path)
        output_path = os.path.join(output_dir, f"detected_{base_name}")
        cv2.imwrite(output_path, annotated_image)
        print(f"Annotated image saved to: {output_path}")
        
        # Try to display image if display is available
        try:
            cv2.imshow("Egg Detection", annotated_image)
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