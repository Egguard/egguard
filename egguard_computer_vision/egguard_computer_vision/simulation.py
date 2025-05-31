#!/usr/bin/env python3

import cv2
import numpy as np
import json
import argparse
import os
from egg_detector import EggDetector
from egg_analysis import EggAnalyzer

def create_test_pattern(width=640, height=480):
    """
    Create a test pattern with simulated eggs (both good and broken)
    
    Args:
        width (int): Image width
        height (int): Image height
        
    Returns:
        tuple: (image, egg_positions) where egg_positions is a list of (center, radius, is_broken)
    """
    # Create dark gray background
    image = np.ones((height, width, 3), dtype=np.uint8) * 120
    
    # Define egg positions
    egg_positions = [
        ((150, 150), 30, False),   # (x, y), radius, broken
        ((300, 200), 40, True),
        ((450, 250), 35, False),
        ((200, 350), 45, True)
    ]
    
    print("Creating test pattern with simulated eggs...")
    
    for (x, y), r, is_broken in egg_positions:
        # Determine egg type: white or brown
        egg_type = np.random.choice(['white', 'brown'])
        egg_color = (230, 230, 230) if egg_type == 'white' else (80, 130, 180)
        
        # Random angle for oval rotation
        angle = np.random.randint(0, 180)
        
        # Draw egg border
        cv2.ellipse(
            image,
            (x, y),
            (r + 2, int((r + 2) * 1.3)),
            angle,
            0, 360,
            (100, 100, 100),
            -1
        )
        
        # Draw egg
        cv2.ellipse(
            image,
            (x, y),
            (r, int(r * 1.3)),
            angle,
            0, 360,
            egg_color,
            -1
        )
        
        # Add texture
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
            print(f"Added broken egg at position ({x}, {y})")
        else:
            print(f"Added normal egg at position ({x}, {y})")
    
    return image, egg_positions

def simulate_broken_egg(image, center, radius, severity=0.7):
    """
    Add simulated cracks to an egg for testing. Creates realistic cracks that
    propagate from an impact point, similar to how a real egg breaks.
    
    Args:
        image (numpy.ndarray): Input image
        center (tuple): (x, y) center coordinates
        radius (int): Egg radius
        severity (float): Crack severity (0.0-1.0)
        
    Returns:
        numpy.ndarray: Image with simulated cracks
    """
    # Create egg mask first
    egg_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    x, y = center
    r = radius
    
    # Draw egg contour
    cv2.ellipse(egg_mask, (x, y), (r, int(r * 1.3)), 0, 0, 360, 255, -1)
    
    # Create crack mask
    crack_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    # Determine impact point (slightly off-center)
    impact_angle = np.random.uniform(0, 2 * np.pi)
    impact_distance = r * np.random.uniform(0.3, 0.5)
    impact_x = int(x + impact_distance * np.cos(impact_angle))
    impact_y = int(y + impact_distance * np.sin(impact_angle))
    
    # Draw main cracks radiating from impact point
    num_main_cracks = int(2 + severity * 3)  # 2-5 main cracks
    
    # First, create the main cracks that radiate from the impact point
    for i in range(num_main_cracks):
        # Calculate angle for this crack
        crack_angle = impact_angle + (i * 2 * np.pi / num_main_cracks) + np.random.uniform(-0.2, 0.2)
        
        # Calculate end point (near edge of egg)
        end_distance = r * np.random.uniform(0.8, 1.0)
        end_x = int(x + end_distance * np.cos(crack_angle))
        end_y = int(y + end_distance * np.sin(crack_angle))
        
        # Draw main crack
        cv2.line(crack_mask, (impact_x, impact_y), (end_x, end_y), 255, thickness=1)
        
        # Add some branching cracks along the main crack
        num_branches = np.random.randint(1, 3)
        for _ in range(num_branches):
            # Calculate branch position (random point along main crack)
            t = np.random.uniform(0.3, 0.7)
            branch_x = int(impact_x + (end_x - impact_x) * t)
            branch_y = int(impact_y + (end_y - impact_y) * t)
            
            # Calculate branch angle (perpendicular to main crack with some variation)
            branch_angle = crack_angle + np.pi/2 + np.random.uniform(-0.3, 0.3)
            branch_length = r * np.random.uniform(0.2, 0.4)
            
            # Calculate branch end point
            end_branch_x = int(branch_x + branch_length * np.cos(branch_angle))
            end_branch_y = int(branch_y + branch_length * np.sin(branch_angle))
            
            # Draw branch
            cv2.line(crack_mask, (branch_x, branch_y), (end_branch_x, end_branch_y), 255, thickness=1)
    
    # Add some smaller cracks near the impact point
    if severity > 0.5:
        num_small_cracks = int(3 + severity * 4)
        for _ in range(num_small_cracks):
            # Random angle for small crack
            small_angle = np.random.uniform(0, 2 * np.pi)
            small_length = r * np.random.uniform(0.1, 0.3)
            
            # Calculate end point
            end_small_x = int(impact_x + small_length * np.cos(small_angle))
            end_small_y = int(impact_y + small_length * np.sin(small_angle))
            
            # Draw small crack
            cv2.line(crack_mask, (impact_x, impact_y), (end_small_x, end_small_y), 255, thickness=1)
    
    # Dilate cracks slightly to make them more visible
    kernel = np.ones((2, 2), np.uint8)
    crack_mask = cv2.dilate(crack_mask, kernel, iterations=1)
    
    # Ensure cracks are only within the egg
    crack_mask = cv2.bitwise_and(crack_mask, egg_mask)
    
    # Create a copy of the image for the cracks
    result = image.copy()
    
    # Create a dark overlay for the cracks
    dark_overlay = np.zeros_like(image)
    dark_overlay[crack_mask > 0] = [20, 20, 20]  # Very dark color for cracks
    
    # Apply the dark overlay only to the crack areas
    result[crack_mask > 0] = cv2.addWeighted(result[crack_mask > 0], 0.3, dark_overlay[crack_mask > 0], 0.7, 0)
    
    return result

def process_frame(frame, egg_detector, egg_analyzer, camera_params, is_test_pattern=False):
    """
    Process a single frame to detect and analyze eggs
    
    Args:
        frame (numpy.ndarray): Input frame
        egg_detector (EggDetector): Initialized egg detector
        egg_analyzer (EggAnalyzer): Initialized egg analyzer
        camera_params (dict): Camera parameters
        is_test_pattern (bool): Whether this is a test pattern (to show world coordinates)
        
    Returns:
        tuple: (processed_frame, egg_info_list)
    """
    # Detect eggs using YOLO
    results = egg_detector.model(frame, conf=egg_detector.confidence_threshold, device=egg_detector.device, verbose=False)
    
    # Extract centers and radii from detections
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
    
    # Analyze eggs
    egg_info_list = egg_analyzer.analyze_eggs(frame, centers, radii, camera_params)
    
    # Create visualization
    vis_image = frame.copy()
    
    # Add total count
    cv2.putText(
        vis_image,
        f"Total Eggs: {len(centers)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )
    
    # Add model info
    cv2.putText(
        vis_image,
        f"Model: YOLOv8n",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2
    )
    
    # Process each detection
    for i, (result, egg_info) in enumerate(zip(results, egg_info_list)):
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            
            for box, conf in zip(boxes, confidences):
                if conf >= egg_detector.confidence_threshold:
                    # Get bounding box
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Determine egg status and color
                    status = "BROKEN" if egg_info['broken'] else "OK"
                    color = (0, 0, 255) if egg_info['broken'] else (0, 255, 0)
    
                    # Draw bounding box
                    cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
                    
                    # Prepare text information
                    text_lines = [
                        f"Egg #{i+1}",
                        f"Status: {status}",
                        f"Conf: {conf:.2f}"
                    ]
                    
                    # Calculate text dimensions
                    text_size = cv2.getTextSize(text_lines[0], cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                    text_height = text_size[1] + 4
                    text_width = max(cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0] for line in text_lines)
                    
                    # Draw semi-transparent background
                    overlay = vis_image.copy()
                    cv2.rectangle(overlay, (x1, y1 - text_height * len(text_lines) - 4),
                                (x1 + text_width + 8, y1),
                                (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.7, vis_image, 0.3, 0, vis_image)
                    
                    # Draw text
                    for j, line in enumerate(text_lines):
                        y = y1 - (text_height * (len(text_lines) - j - 1)) - 4
                        cv2.putText(vis_image, line, (x1 + 4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return vis_image, egg_info_list

def main():
    parser = argparse.ArgumentParser(description='Egg detection simulation')
    parser.add_argument('--input', type=str, help='Path to input image or video file')
    parser.add_argument('--output', type=str, default='simulation_data/egg_detection_result.jpg', help='Path to output file')
    parser.add_argument('--json_output', type=str, default='simulation_data/egg_data.json', help='Path to output JSON file')
    parser.add_argument('--model_path', type=str, required=True, help='Path to YOLO model file')
    args = parser.parse_args()
    
    print('🚀 Starting egg detection simulation...')
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    os.makedirs(os.path.dirname(args.json_output), exist_ok=True)
    
    print('Initializing YOLO detector and analyzer...')
    
    # Initialize detector and analyzer
    try:
        egg_detector = EggDetector(model_path=args.model_path)
        print('✅ YOLO detector initialized successfully')
        
        # Log model info
        model_info = egg_detector.get_model_info()
        print(f'Model info: {model_info}')
        
        egg_analyzer = EggAnalyzer()
        print('✅ Egg analyzer initialized successfully')
    except Exception as e:
        print(f'❌ Failed to initialize detector or analyzer: {e}')
        return
    
    # Default camera parameters
    camera_params = {
        'height_mm': 150,           # Height of camera from ground
        'angle_rad': 0.5,           # Camera tilt angle in radians (~30 degrees down)
        'offset_x_mm': 0,           # X offset from robot center
        'offset_y_mm': 69,          # Y offset from robot center
        'focal_length_pixels': 800, # Estimated focal length in pixels
        'fov_h_rad': 1.05           # Horizontal FOV in radians (~60 degrees)
    }
    
    # Process input
    if args.input:
        if args.input.lower().endswith(('.mp4', '.avi', '.mov')):
            # Process video
            print(f"Processing video: {args.input}")
            cap = cv2.VideoCapture(args.input)
            if not cap.isOpened():
                print(f"❌ Failed to open video: {args.input}")
                return
            
            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
            
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Process frame (not test pattern)
                processed_frame, egg_info_list = process_frame(frame, egg_detector, egg_analyzer, camera_params, is_test_pattern=False)
                
                # Write frame
                out.write(processed_frame)
                
                # Show frame
                cv2.imshow('Egg Detection', processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                frame_count += 1
                if frame_count % 30 == 0:  # Log every 30 frames
                    print(f"Processed {frame_count} frames...")
            
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            print(f"✅ Video processing completed. Saved to: {args.output}")
        else:
            # Process single image
            print(f"Processing image: {args.input}")
            image = cv2.imread(args.input)
            if image is None:
                print(f"❌ Failed to load image: {args.input}")
                return
            
            # Process image (not test pattern)
            processed_image, egg_info_list = process_frame(image, egg_detector, egg_analyzer, camera_params, is_test_pattern=False)
            
            # Save results
            cv2.imwrite(args.output, processed_image)
            print(f"✅ Processed image saved to: {args.output}")
            
            # Save egg data
            with open(args.json_output, 'w') as f:
                json.dump(egg_info_list, f, indent=2)
            print(f"✅ Egg data saved to: {args.json_output}")
            
            # Show results
            cv2.imshow('Egg Detection', processed_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()    
    else:
        # Generate and process test pattern
        print("No input specified, generating test pattern...")
        image, egg_positions = create_test_pattern()
        
        # Process test pattern (with world coordinates)
        processed_image, egg_info_list = process_frame(image, egg_detector, egg_analyzer, camera_params, is_test_pattern=True)
        
        # Save results
        cv2.imwrite(args.output, processed_image)
        print(f"✅ Test pattern saved to: {args.output}")
        
        # Save egg data
        with open(args.json_output, 'w') as f:
            json.dump(egg_info_list, f, indent=2)
        print(f"✅ Egg data saved to: {args.json_output}")
        
        # Show results
        cv2.imshow('Egg Detection', processed_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    print("\n✨ Simulation completed successfully!")

if __name__ == "__main__":
    main()