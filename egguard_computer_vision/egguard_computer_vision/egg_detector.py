import cv2
import numpy as np
from ultralytics import YOLO
import os
import torch

class EggDetector:
    """
    Class for detecting eggs in images using YOLOv8n model.
    Replaces the previous OpenCV-based detection with trained YOLO model.
    """
    
    def __init__(self, model_path=None, confidence_threshold=0.5, target_width=640):
        """
        Initialize the egg detector with YOLOv8n model.
        
        Args:
            model_path (str): Path to the trained YOLOv8n model file (.pt)
            confidence_threshold (float): Minimum confidence for detections (0.0-1.0)
            target_width (int): Target width for resized image during preprocessing
        """
        self.target_width = target_width
        self.confidence_threshold = confidence_threshold
        
        # Primero imprimir información de debug antes de cargar el modelo
        print(f"[DEBUG] Iniciando EggDetector con confidence_threshold={confidence_threshold}")
        
        # Default model path - adjust based on your model location
        if model_path is None:
            # Try common locations for the model, including the relative path you mentioned
            possible_paths = [
                "./egguard_models/best.pt",        # Your actual model location (relative)
                "egguard_models/best.pt",          # Alternative relative path
                os.path.expanduser("~/egguard_models/best.pt"),
                os.path.expanduser("~/models/egg_yolo.pt"),
                "/opt/egguard/models/best.pt",
                "./models/best.pt",
                "./best.pt",
                # Also try from the package directory
                os.path.join(os.path.dirname(__file__), "egguard_models", "best.pt"),
                os.path.join(os.path.dirname(__file__), "..", "egguard_models", "best.pt"),
            ]
            
            model_path = None
            for path in possible_paths:
                print(f"[DEBUG] Checking path: {path} - {'EXISTS' if os.path.exists(path) else 'NOT FOUND'}")
                if os.path.exists(path):
                    model_path = path
                    print(f"[DEBUG] Found YOLO model at: {path}")
                    break
        
        if model_path is None or not os.path.exists(model_path):
            # Print current working directory for debugging
            current_dir = os.getcwd()
            print(f"[DEBUG] Current working directory: {current_dir}")
            
            # List contents of current directory
            print("[DEBUG] Contents of current directory:")
            try:
                for item in os.listdir(current_dir):
                    print(f"  {item}")
            except:
                print("  Could not list directory contents")
            
            # Check if egguard_models directory exists
            if os.path.exists("./egguard_models"):
                print("[DEBUG] Contents of ./egguard_models directory:")
                try:
                    for item in os.listdir("./egguard_models"):
                        print(f"  {item}")
                except:
                    print("  Could not list egguard_models directory contents")
            
            raise FileNotFoundError(
                f"YOLOv8n model not found. Please provide a valid model_path or place your model at one of these locations: {possible_paths}"
            )
        
        self.model_path = model_path
        
        try:
            print(f"[DEBUG] Intentando cargar modelo YOLO desde: {model_path}")
            
            # Check if CUDA is available BEFORE loading the model
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"[DEBUG] Using device: {self.device}")
            
            # Load YOLOv8n model - Especificar device desde el inicio
            print(f"[DEBUG] Cargando modelo YOLO...")
            self.model = YOLO(model_path)
            print(f"[DEBUG] Modelo YOLO cargado exitosamente")
            
            # Mover el modelo al dispositivo apropiado
            print(f"[DEBUG] Moviendo modelo al dispositivo: {self.device}")
            self.model.to(self.device)
            print(f"[DEBUG] Modelo movido al dispositivo exitosamente")
            
            print(f"[DEBUG] Successfully loaded YOLOv8n model from: {model_path}")
            
        except Exception as e:
            print(f"[ERROR] Failed to load YOLOv8n model: {str(e)}")
            raise RuntimeError(f"Failed to load YOLOv8n model: {str(e)}")
        
        # Store original image dimensions for coordinate scaling
        self.original_height = None
        self.original_width = None
        self.scale_x = 1.0
        self.scale_y = 1.0
        
        print(f"[DEBUG] EggDetector inicializado correctamente")
        
    def preprocess_image(self, image):
        """
        Preprocess the input image for YOLOv8n inference.
        Store scaling factors for coordinate conversion.
        
        Args:
            image (numpy.ndarray): Input image in BGR format
        
        Returns:
            numpy.ndarray: Preprocessed image
        """
        # Store original dimensions
        self.original_height, self.original_width = image.shape[:2]
        
        # Calculate aspect ratio to maintain proportion when resizing
        aspect_ratio = self.original_height / self.original_width
        target_height = int(self.target_width * aspect_ratio)
        
        # Resize image
        resized = cv2.resize(image, (self.target_width, target_height))
        
        # Calculate scaling factors for coordinate conversion
        self.scale_x = self.original_width / self.target_width
        self.scale_y = self.original_height / target_height
        
        return resized

    def detect_eggs_yolo(self, image):
        """
        Detect eggs using YOLOv8n model.
        
        Args:
            image (numpy.ndarray): Input image in BGR format
        
        Returns:
            tuple: (centers, radii) where centers is a list of (x, y) coordinates
                and radii is a list of corresponding radii
        """
        try:
            # Run inference - Usar verbose=False para reducir output
            results = self.model(image, conf=self.confidence_threshold, device=self.device, verbose=False)
            
            centers = []
            radii = []
            
            # Process results
            for result in results:
                if result.boxes is not None:
                    # Get detection boxes and confidences
                    boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2 format
                    confidences = result.boxes.conf.cpu().numpy()
                    
                    for box, conf in zip(boxes, confidences):
                        if conf >= self.confidence_threshold:
                            x1, y1, x2, y2 = box
                            
                            # Calculate center coordinates
                            center_x = int((x1 + x2) / 2)
                            center_y = int((y1 + y2) / 2)
                            
                            # Calculate radius as half of the average of width and height
                            width = x2 - x1
                            height = y2 - y1
                            radius = int((width + height) / 4)  # Average of half-width and half-height
                            
                            # Scale coordinates back to original image size if needed
                            if hasattr(self, 'scale_x') and hasattr(self, 'scale_y'):
                                center_x = int(center_x * self.scale_x)
                                center_y = int(center_y * self.scale_y)
                                radius = int(radius * ((self.scale_x + self.scale_y) / 2))
                            
                            centers.append((center_x, center_y))
                            radii.append(radius)
            
            return centers, radii
            
        except Exception as e:
            print(f"Error during YOLO inference: {str(e)}")
            return [], []

    def detect_eggs_hough(self, preprocessed_img):
        """
        Detect eggs using Hough Circle Transform (backup method).
        Kept for fallback purposes if YOLO fails.
        
        Args:
            preprocessed_img (numpy.ndarray): Preprocessed grayscale image
        
        Returns:
            tuple: (centers, radii) where centers is a list of (x, y) coordinates
                and radii is a list of corresponding radii
        """
        # Convert to grayscale if needed
        if len(preprocessed_img.shape) == 3:
            gray = cv2.cvtColor(preprocessed_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = preprocessed_img.copy()
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Define parameter ranges for Hough Circle detection
        min_dist = 20  # Minimum distance between detected centers
        param1 = 50    # Higher threshold for Canny edge detector
        param2 = 30    # Accumulator threshold (lower = more false circles)
        min_radius = 10
        max_radius = 100
        
        # Apply Hough Circle Transform
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=min_dist,
            param1=param1,
            param2=param2,
            minRadius=min_radius,
            maxRadius=max_radius
        )
        
        centers = []
        radii = []
        
        # Process detected circles
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles[0, :]:
                x, y, r = circle
                
                # Scale coordinates back to original image size if needed
                if hasattr(self, 'scale_x') and hasattr(self, 'scale_y'):
                    x = int(x * self.scale_x)
                    y = int(y * self.scale_y)
                    r = int(r * ((self.scale_x + self.scale_y) / 2))
                
                centers.append((x, y))
                radii.append(r)
        
        return centers, radii

    def detect_eggs(self, image):
        """
        Main function to detect eggs using YOLOv8n model with Hough fallback.
        
        Args:
            image (numpy.ndarray): Input image in BGR format
        
        Returns:
            tuple: (centers, radii) where centers is a list of (x, y) coordinates
                and radii is a list of corresponding radii
        """
        try:
            # Preprocess image (this will set scaling factors)
            preprocessed = self.preprocess_image(image)
            
            # Primary detection method: YOLOv8n
            centers, radii = self.detect_eggs_yolo(preprocessed)
            
            # If YOLO didn't detect anything, try Hough circles as fallback
            if not centers:
                print("[DEBUG] YOLO didn't detect any eggs, trying Hough circles as fallback...")
                centers, radii = self.detect_eggs_hough(preprocessed)
            
            # Apply non-maximum suppression to remove overlapping detections
            final_centers, final_radii = self._non_max_suppression(centers, radii)
            
            print(f"[DEBUG] Detected {len(final_centers)} eggs using {'YOLO' if centers else 'Hough fallback'}")
            
            return final_centers, final_radii
            
        except Exception as e:
            print(f"Error in egg detection: {str(e)}")
            # Return empty results on error
            return [], []

    def _non_max_suppression(self, centers, radii, overlap_threshold=0.5):
        """
        Apply non-maximum suppression to remove overlapping detections.
        
        Args:
            centers (list): List of (x, y) center coordinates
            radii (list): List of radii corresponding to centers
            overlap_threshold (float): IoU threshold for considering detections as duplicates
        
        Returns:
            tuple: (final_centers, final_radii) after removing overlapping detections
        """
        if not centers:
            return [], []
        
        # Convert lists to numpy arrays
        centers = np.array(centers, dtype=np.float64)
        radii = np.array(radii, dtype=np.float64)
        
        # Sort by radius (larger radius first)
        idx = np.argsort(radii)[::-1]
        centers = centers[idx]
        radii = radii[idx]
        
        # Initialize list to keep track of which detections to keep
        keep = [0]  # Start by keeping the largest circle
        
        # For each detection
        for i in range(1, len(centers)):
            # Check if this detection overlaps significantly with any kept detection
            should_keep = True
            
            for j in keep:
                # Calculate distance between centers
                x1, y1 = centers[i].astype(np.float64)
                x2, y2 = centers[j].astype(np.float64)
                distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                
                # Calculate overlap
                r1 = float(radii[i])
                r2 = float(radii[j])
                
                # Prevent division by zero
                min_radius = min(r1, r2)
                if min_radius <= 0:
                    continue
                    
                # Calculate overlap ratio more safely
                overlap = max(0.0, (r1 + r2 - distance)) / min_radius
                
                # If overlap is too high, don't keep this detection
                if overlap > overlap_threshold:
                    should_keep = False
                    break
            
            # If no significant overlap with any kept detection, keep this one too
            if should_keep:
                keep.append(i)
        
        # Return the kept detections
        final_centers = [tuple(map(int, centers[i])) for i in keep]
        final_radii = [int(radii[i]) for i in keep]
        
        return final_centers, final_radii

    def draw_detections(self, image, centers, radii, egg_info_list=None):
        """
        Draw circles around detected eggs and annotate with information.
        Format exactly matches simulation.py visualization:
        - Draw circle around each egg
        - Add text below with status and coordinates
        - Green color for OK eggs, red for BROKEN eggs
        
        Args:
            image (numpy.ndarray): Original image to draw on
            centers (list): List of (x, y) center coordinates
            radii (list): List of radii corresponding to centers
            egg_info_list (list, optional): List of dictionaries with egg information
                                           including 'broken', 'coordX', 'coordY'
        
        Returns:
            numpy.ndarray: Image with drawn detections
        """
        # Create a copy of the image
        result = image.copy()
        
        # Draw each detection
        for i, ((x, y), r) in enumerate(zip(centers, radii)):
            # Default values in case egg_info_list doesn't have this egg
            broken = False
            coord_x = 0.0
            coord_y = 0.0
            
            # Get egg info if available
            if egg_info_list and i < len(egg_info_list):
                egg_info = egg_info_list[i]
                broken = egg_info.get('broken', False)
                coord_x = egg_info.get('coordX', 0.0)
                coord_y = egg_info.get('coordY', 0.0)
            
            status = "BROKEN" if broken else "OK"
            color = (0, 0, 255) if broken else (0, 255, 0)  # Red for broken, green for OK
            
            # Draw the circle around the egg with appropriate color
            cv2.circle(result, (x, y), r, color, 2)
            
            # Draw the center of the circle
            cv2.circle(result, (x, y), 2, (0, 0, 255), 3)
            
            # Add label with egg number above the egg
            cv2.putText(
                result,
                f"Egg #{i+1}",
                (x - 20, y - r - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )
            
            # Add status and coordinates below the egg - EXACTLY as in simulation.py
            cv2.putText(
                result,
                f"{status} ({coord_x:.2f}, {coord_y:.2f})",
                (x - 20, y + r + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )
        
        # Add total count
        cv2.putText(
            result,
            f"Total eggs: {len(centers)} (YOLO)",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )
        
        # Add model info
        cv2.putText(
            result,
            f"Model: YOLOv8n",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )
        
        return result

    def get_model_info(self):
        """
        Get information about the loaded model.
        
        Returns:
            dict: Model information
        """
        return {
            'model_path': self.model_path,
            'device': self.device,
            'confidence_threshold': self.confidence_threshold,
            'target_width': self.target_width
        }
        
    def detect_eggs_yolo_with_visualization(self, image, show_raw_detections=True):
        """
        Detect eggs using YOLOv8n model with enhanced visualization.
        
        Args:
            image (numpy.ndarray): Input image in BGR format
            show_raw_detections (bool): Whether to show raw YOLO detections
        
        Returns:
            tuple: (centers, radii, annotated_image) where annotated_image shows YOLO results
        """
        try:
            # Preprocess image
            preprocessed = self.preprocess_image(image)

            # Run YOLO inference - verbose=False para reducir output
            results = self.model(preprocessed, conf=self.confidence_threshold, device=self.device, verbose=False)
            result_image = preprocessed.copy()

            centers = []
            radii = []

            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()

                    for box, conf in zip(boxes, confidences):
                        if conf >= self.confidence_threshold:
                            x1, y1, x2, y2 = box
                            center_x = int((x1 + x2) / 2)
                            center_y = int((y1 + y2) / 2)
                            width = x2 - x1
                            height = y2 - y1
                            radius = int((width + height) / 4)

                            # Scale back to original size
                            if hasattr(self, 'scale_x') and hasattr(self, 'scale_y'):
                                center_x = int(center_x * self.scale_x)
                                center_y = int(center_y * self.scale_y)
                                radius = int(radius * ((self.scale_x + self.scale_y) / 2))
                                x1 = int(x1 * self.scale_x)
                                y1 = int(y1 * self.scale_y)
                                x2 = int(x2 * self.scale_x)
                                y2 = int(y2 * self.scale_y)

                            centers.append((center_x, center_y))
                            radii.append(radius)

                            if show_raw_detections:
                                cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 255), 2)
                                cv2.putText(result_image, f"{conf:.2f}", (x1, y1 - 5),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            # Resize back the annotated image to match original image size
            annotated_image = cv2.resize(result_image, (self.original_width, self.original_height))

            return centers, radii, annotated_image

        except Exception as e:
            print(f"Error during enhanced YOLO detection: {str(e)}")
            return [], [], image

    def show_live_yolo_processing(self, image, results, egg_info_list=None):
        """
        Show live YOLO processing results with egg information.
        Implements smart label positioning to avoid overlaps and ensure visibility.
        
        Args:
            image (numpy.ndarray): Original image
            results (list): YOLO detection results
            egg_info_list (list, optional): List of egg information dictionaries
        """
        try:
            # Create a copy of the image for visualization
            vis_image = image.copy()
            height, width = vis_image.shape[:2]
            
            # Calculate new dimensions for larger display
            # Maintain aspect ratio while scaling up
            scale_factor = 2  # Increase this value for larger window
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            
            # Resize the image for display
            vis_image = cv2.resize(vis_image, (new_width, new_height))
            
            # Store label positions to avoid overlaps
            label_positions = []
            
            # Process YOLO detection results
            egg_count = 0
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    
                    for i, (box, conf) in enumerate(zip(boxes, confidences)):
                        if conf >= self.confidence_threshold:
                            # Scale bounding box coordinates to new size
                            x1, y1, x2, y2 = map(int, box * scale_factor)
                            
                            # Get egg information if available
                            egg_info = egg_info_list[egg_count] if egg_info_list and egg_count < len(egg_info_list) else None
                            
                            # Determine egg status and color
                            if egg_info and 'broken' in egg_info:
                                status = "BROKEN" if egg_info['broken'] else "OK"
                                color = (0, 0, 255) if egg_info['broken'] else (0, 255, 0)  # Red for broken, green for OK
                            else:
                                status = "UNKNOWN"
                                color = (0, 255, 255)  # Yellow for unknown
                            
                            # Draw bounding box
                            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
                            
                            # Prepare text information
                            text_lines = [
                                f"Egg #{egg_count+1}",
                                f"Status: {status}",
                                f"Conf: {conf:.2f}"
                            ]
                            
                            # Add position information if available
                            if egg_info and 'worldX' in egg_info and 'worldY' in egg_info:
                                text_lines.append(f"World Pos: ({egg_info['worldX']:.2f}, {egg_info['worldY']:.2f})")
                            
                            # Calculate text dimensions (scaled for larger display)
                            font_scale = 0.5  # Increased font scale
                            text_size = cv2.getTextSize(text_lines[0], cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]
                            text_height = text_size[1] + 4
                            text_width = max(cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0] for line in text_lines)
                            
                            # Try different positions for the label
                            positions = [
                                # Above the box
                                (x1, y1 - text_height * len(text_lines) - 4),
                                # Below the box
                                (x1, y2 + 4),
                                # Right side
                                (x2 + 4, y1),
                                # Left side
                                (x1 - text_width - 8, y1),
                                # Inside the box (top)
                                (x1 + 4, y1 + text_height + 4)
                            ]
                            
                            # Find the best position that doesn't overlap and stays within bounds
                            best_position = None
                            for pos_x, pos_y in positions:
                                # Check if the label would be within image bounds
                                if (pos_x >= 0 and pos_x + text_width + 8 <= new_width and
                                    pos_y >= 0 and pos_y + text_height * len(text_lines) + 4 <= new_height):
                                    
                                    # Check for overlaps with existing labels
                                    label_rect = (pos_x, pos_y, text_width + 8, text_height * len(text_lines) + 4)
                                    overlap = False
                                    
                                    for existing_rect in label_positions:
                                        if (label_rect[0] < existing_rect[0] + existing_rect[2] and
                                            label_rect[0] + label_rect[2] > existing_rect[0] and
                                            label_rect[1] < existing_rect[1] + existing_rect[3] and
                                            label_rect[1] + label_rect[3] > existing_rect[1]):
                                            overlap = True
                                            break
                                    
                                    if not overlap:
                                        best_position = (pos_x, pos_y)
                                        break
                            
                            # If no good position found, use the first position and clip to image bounds
                            if best_position is None:
                                pos_x = max(0, min(x1, new_width - text_width - 8))
                                pos_y = max(0, min(y1, new_height - text_height * len(text_lines) - 4))
                                best_position = (pos_x, pos_y)
                            
                            label_x, label_y = best_position
                            
                            # Add this label position to the list
                            label_positions.append((label_x, label_y, text_width + 8, text_height * len(text_lines) + 4))
                            
                            # Draw semi-transparent background
                            overlay = vis_image.copy()
                            cv2.rectangle(overlay, (label_x, label_y),
                                        (label_x + text_width + 8, label_y + text_height * len(text_lines) + 4),
                                        (0, 0, 0), -1)
                            cv2.addWeighted(overlay, 0.7, vis_image, 0.3, 0, vis_image)
                            
                            # Draw text with larger font
                            for j, line in enumerate(text_lines):
                                y = label_y + (text_height * j) + text_height
                                cv2.putText(vis_image, line, (label_x + 4, y), 
                                          cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)
                            
                            egg_count += 1
            
            # Add total count with larger font
            cv2.putText(vis_image, f"Total Eggs: {egg_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
            
            # Show the image in a named window
            cv2.namedWindow("Egg Detection", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Egg Detection", new_width, new_height)
            cv2.imshow("Egg Detection", vis_image)
            cv2.waitKey(1)
            
        except Exception as e:
            self.log_debug(f"Error in visualization: {str(e)}")
            # Show original image if visualization fails
            cv2.namedWindow("Egg Detection", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Egg Detection", new_width, new_height)
            cv2.imshow("Egg Detection", image)
            cv2.waitKey(1)

    def visualize_raw_yolo_detections(self, image):
        """
        Visualize raw YOLO detections (bounding boxes and confidences) on the image.
        
        Args:
            image (numpy.ndarray): Input image in BGR format
        
        Returns:
            numpy.ndarray: Annotated image with raw YOLO detections
        """
        try:
            preprocessed = self.preprocess_image(image)
            # verbose=False para reducir output
            results = self.model(preprocessed, conf=self.confidence_threshold, device=self.device, verbose=False)
            annotated = preprocessed.copy()

            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()

                    for box, conf in zip(boxes, confidences):
                        if conf >= self.confidence_threshold:
                            x1, y1, x2, y2 = map(int, box)
                            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)
                            cv2.putText(annotated, f"{conf:.2f}", (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            annotated_resized = cv2.resize(annotated, (self.original_width, self.original_height))
            return annotated_resized

        except Exception as e:
            print(f"Error visualizing raw YOLO detections: {str(e)}")
            return image