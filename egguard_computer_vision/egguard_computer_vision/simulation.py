#!/usr/bin/env python3

import cv2
import numpy as np
import json
import argparse
import os
from egg_detector import EggDetector
from egg_analysis import EggAnalyzer

def simulate_broken_egg(image, center, radius, severity=0.7):
    """
    Add simulated cracks to an egg for testing
    """
    # Create crack mask
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    x, y = center
    r = radius
    
    # Draw random cracks
    num_cracks = int(3 + severity * 5)
    for _ in range(num_cracks):
        # Random start point near edge of egg
        angle = np.random.uniform(0, 2 * np.pi)
        start_r = r * 0.7
        start_x = int(x + start_r * np.cos(angle))
        start_y = int(y + start_r * np.sin(angle))
        
        # Random end point
        end_r = r * np.random.uniform(0.8, 1.0)
        end_angle = angle + np.random.uniform(-0.5, 0.5)
        end_x = int(x + end_r * np.cos(end_angle))
        end_y = int(y + end_r * np.sin(end_angle))
        
        # Draw crack line
        cv2.line(mask, (start_x, start_y), (end_x, end_y), 255, thickness=1)
        
        # Add some small branches
        branches = np.random.randint(1, 4)
        for _ in range(branches):
            branch_len = int(r * np.random.uniform(0.1, 0.3))
            branch_angle = end_angle + np.random.uniform(-0.8, 0.8)
            branch_x = int(end_x + branch_len * np.cos(branch_angle))
            branch_y = int(end_y + branch_len * np.sin(branch_angle))
            cv2.line(mask, (end_x, end_y), (branch_x, branch_y), 255, thickness=1)
    
    # Dilate cracks slightly
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    
    # Apply cracks to image with slight darkening and color change
    crack_color = np.array([60, 60, 60], dtype=np.uint8)  # Dark gray
    crack_overlay = np.zeros_like(image)
    crack_overlay[mask > 0] = crack_color
    
    # Blend cracks with original image
    alpha = 0.7
    result = cv2.addWeighted(image, 1.0, crack_overlay, alpha, 0)
    
    return result

def add_text_overlay(image, egg_info_list):
    """Add text overlay with egg data to the image"""
    # Create a copy of the image
    result = image.copy()
    
    # Add egg data as text
    y = 30
    for i, egg_info in enumerate(egg_info_list):
        line = f"Egg #{i+1}: {json.dumps(egg_info)}"
        cv2.putText(
            result, 
            line, 
            (10, y), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (255, 255, 255), 
            1, 
            cv2.LINE_AA
        )
        y += 20
    
    return result

def main():
    parser = argparse.ArgumentParser(description='Egg detection simulation')
    parser.add_argument('--image', type=str, help='Path to input image file')
    parser.add_argument('--output', type=str, default='simulation_data/egg_detection_result.jpg', help='Path to output image file')
    parser.add_argument('--json_output', type=str, default='simulation_data/egg_data.json', help='Path to output JSON file')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    os.makedirs(os.path.dirname(args.json_output), exist_ok=True)
    
    # Initialize our detector and analyzer classes
    egg_detector = EggDetector()
    egg_analyzer = EggAnalyzer()
    
    # Use default egg image if not provided
    image_path = args.image
    if not image_path or not os.path.isfile(image_path):
        print("No valid image specified, using default test pattern")
        # Create a test pattern with eggs
        image = np.ones((480, 640, 3), dtype=np.uint8) * 120  # Dark gray background

        
        # Add some eggs
        egg_positions = [
            ((150, 150), 30, False),   # (x, y), radius, broken
            ((300, 200), 40, True),
            ((450, 250), 35, False),
            ((200, 350), 45, True)
        ]
        
        for (x, y), r, is_broken in egg_positions:
            # Determine egg type: white or brown
            egg_type = np.random.choice(['white', 'brown'])

            if egg_type == 'white':
                egg_color = (230, 230, 230)  # Light gray-white
            else:
                egg_color = (80, 130, 180)   # Light brown

            # Random angle for oval rotation
            angle = np.random.randint(0, 180)

            # Draw a slightly larger ellipse as border (gray)
            border_color = (100, 100, 100)
            cv2.ellipse(
                image,
                (x, y),
                (r + 2, int((r + 2) * 1.3)),  # Slightly larger for border
                angle,
                0, 360,
                border_color,
                -1
            )

            # Draw the actual egg on top
            cv2.ellipse(
                image,
                (x, y),
                (r, int(r * 1.3)),
                angle,
                0, 360,
                egg_color,
                -1
            )

            # Add some texture
            texture = np.random.randint(0, 10, size=image.shape[:2], dtype=np.uint8)
            texture_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.ellipse(
                texture_mask, 
                (x, y), 
                (r, int(r*1.3)), 
                np.random.randint(0, 180), 
                0, 360, 
                255, 
                -1
            )
            image_region = image[texture_mask > 0]
            texture_region = texture[texture_mask > 0]
            image_region[:] = np.clip(image_region - texture_region[:, np.newaxis], 0, 255)
            
            # Add cracks if broken
            if is_broken:
                image = simulate_broken_egg(image, (x, y), r)
    else:
        # Load the specified image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            return
    
    # Preprocess the image using egg detector
    processed_img = egg_detector.preprocess_image(image)
    
    # Detect eggs using detector class
    centers, radii = egg_detector.detect_eggs(processed_img)
    
    # Default camera parameters for simulation
    camera_params = {
        'height_mm': 150,           # Height of camera from ground
        'angle_rad': 0.5,           # Camera tilt angle in radians (~30 degrees down)
        'offset_x_mm': 0,           # X offset from robot center
        'offset_y_mm': 69,          # Y offset from robot center
        'focal_length_pixels': 800, # Estimated focal length in pixels
        'fov_h_rad': 1.05           # Horizontal FOV in radians (~60 degrees)
    }
    
    # Analyze eggs (detect if broken and get coordinates)
    egg_info_list = []
    
    if not image_path or not os.path.isfile(image_path):
        # Simulation mode - build data without analyzing real position
        for i, ((center_x, center_y), radius) in enumerate(zip(centers, radii)):
            egg_info = {
                "coordX": 0.0,
                "coordY": 0.0,
                "broken": False
            }

            # Check if the egg is broken based on simulated positions
            for (bx, by), br, is_broken in egg_positions:
                dist = np.linalg.norm(np.array([center_x, center_y]) - np.array([bx, by]))
                if dist < (radius + br) * 0.5:
                    egg_info["broken"] = is_broken
                    break

            egg_info_list.append(egg_info)
    else:
        # Real analysis with distance using analyzer class
        egg_results = egg_analyzer.analyze_eggs(image, centers, radii, camera_params)
        for egg_data in egg_results:
            egg_info = {
                "coordX": egg_data['coordX'],
                "coordY": egg_data['coordY'],
                "broken": egg_data['broken']
            }
            egg_info_list.append(egg_info)
    
    # Print egg information
    print(f"Detected {len(centers)} eggs:")
    for i, egg_info in enumerate(egg_info_list):
        status = "BROKEN" if egg_info['broken'] else "OK"
        print(f"Egg #{i+1}: {status}, Position: ({egg_info['coordX']:.3f}, {egg_info['coordY']:.3f})")
    
    # Save individual JSON files for each egg (for backend compatibility)
    for i, egg_info in enumerate(egg_info_list):
        # Create a filename for each egg
        egg_filename = os.path.splitext(args.json_output)[0] + f"_egg{i+1}.json"
        with open(egg_filename, 'w') as f:
            json.dump(egg_info, f)
        print(f"Egg #{i+1} data saved to {egg_filename}")
    
    # Also save a consolidated file for reference
    with open(args.json_output, 'w') as f:
        json.dump(egg_info_list, f, indent=2)
    print(f"All egg data saved to {args.json_output}")
    
    # Draw detections on the image using detector class
    result_img = egg_detector.draw_detections(image.copy(), centers, radii, egg_info_list)
    
    # Add status indicators
    for i, ((x, y), r, egg_info) in enumerate(zip(centers, radii, egg_info_list)):
        status = "BROKEN" if egg_info['broken'] else "OK"
        color = (0, 0, 255) if egg_info['broken'] else (0, 255, 0)
        
        cv2.putText(
            result_img,
            f"{status} ({egg_info['coordX']:.2f}, {egg_info['coordY']:.2f})",
            (x - 20, y + r + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )
    
    # Add egg data overlay
    result_with_data = add_text_overlay(result_img, egg_info_list)
    
    # Save the output image
    cv2.imwrite(args.output, result_with_data)
    print(f"Result image saved to {args.output}")
    
    # Display the image if not in headless mode
    cv2.imshow("Egg Detection Result", result_with_data)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()