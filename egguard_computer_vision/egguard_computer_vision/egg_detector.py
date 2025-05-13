import cv2
import numpy as np

class EggDetector:
    """
    Class for detecting eggs in images using computer vision techniques.
    Combines multiple detection methods for robust egg detection.
    """
    
    def __init__(self, target_width=640):
        """
        Initialize the egg detector.
        
        Args:
            target_width (int): Target width for resized image during preprocessing
        """
        self.target_width = target_width
        
    def preprocess_image(self, image):
        """
        Preprocess the input image by resizing and applying filters.
        
        Args:
            image (numpy.ndarray): Input image in BGR format
        
        Returns:
            numpy.ndarray: Preprocessed image
        """
        # Calculate aspect ratio to maintain proportion when resizing
        height, width = image.shape[:2]
        aspect_ratio = height / width
        target_height = int(self.target_width * aspect_ratio)
        
        # Resize image to reduce computational load
        resized = cv2.resize(image, (self.target_width, target_height))
        
        # Convert to grayscale
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        return blurred

    def detect_eggs_hough(self, preprocessed_img):
        """
        Detect eggs using Hough Circle Transform.
        
        Args:
            preprocessed_img (numpy.ndarray): Preprocessed grayscale image
        
        Returns:
            tuple: (centers, radii) where centers is a list of (x, y) coordinates
                and radii is a list of corresponding radii
        """
        # Define parameter ranges for Hough Circle detection
        # For brown eggs on white/gray background
        min_dist = 20  # Minimum distance between detected centers
        param1 = 50    # Higher threshold for Canny edge detector
        param2 = 30    # Accumulator threshold (lower = more false circles)
        min_radius = 10
        max_radius = 100
        
        # Apply Hough Circle Transform
        circles = cv2.HoughCircles(
            preprocessed_img,
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
                centers.append((x, y))
                radii.append(r)
        
        return centers, radii

    def detect_eggs_contour(self, preprocessed_img):
        """
        Detect eggs using contour detection.
        
        Args:
            preprocessed_img (numpy.ndarray): Preprocessed grayscale image
        
        Returns:
            tuple: (centers, radii) where centers is a list of (x, y) coordinates
                and radii is a list of corresponding radii
        """
        # Apply adaptive thresholding to separate eggs from background
        thresh = cv2.adaptiveThreshold(
            preprocessed_img,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,
            2
        )
        
        # Apply morphological operations to clean up the binary image
        kernel = np.ones((5, 5), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        centers = []
        radii = []
        
        # Filter contours by area and shape to find egg-like objects
        for contour in contours:
            # Calculate area
            area = cv2.contourArea(contour)
            
            # Filter small contours
            if area < 100:
                continue
            
            # Get enclosing circle
            (x, y), radius = cv2.minEnclosingCircle(contour)
            
            # Calculate circularity
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            
            # Eggs are approximately circular but not perfectly so
            if 0.5 < circularity < 0.9:
                centers.append((int(x), int(y)))
                radii.append(int(radius))
        
        return centers, radii

    def detect_eggs_color(self, image, preprocessed_img):
        """
        Detect eggs using color filtering techniques.
        
        Args:
            image (numpy.ndarray): Original image in BGR format
            preprocessed_img (numpy.ndarray): Preprocessed grayscale image
        
        Returns:
            tuple: (centers, radii) where centers is a list of (x, y) coordinates
                and radii is a list of corresponding radii
        """
        # Convert to HSV color space
        hsv = cv2.cvtColor(cv2.resize(image, (preprocessed_img.shape[1], preprocessed_img.shape[0])), cv2.COLOR_BGR2HSV)
        
        # Define range for brown egg color in HSV
        # These values may need adjustment based on lighting conditions
        lower_brown = np.array([5, 50, 50])
        upper_brown = np.array([30, 255, 255])
        
        # Create mask for brown color
        mask = cv2.inRange(hsv, lower_brown, upper_brown)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        centers = []
        radii = []
        
        # Process contours
        for contour in contours:
            # Calculate area
            area = cv2.contourArea(contour)
            
            # Filter small contours
            if area < 100:
                continue
            
            # Get enclosing circle
            (x, y), radius = cv2.minEnclosingCircle(contour)
            
            centers.append((int(x), int(y)))
            radii.append(int(radius))
        
        return centers, radii

    def detect_eggs(self, image):
        """
        Main function to detect eggs, combining multiple detection methods.
        
        Args:
            image (numpy.ndarray): Input image (either original or preprocessed)
        
        Returns:
            tuple: (centers, radii) where centers is a list of (x, y) coordinates
                and radii is a list of corresponding radii
        """
        # If input is already grayscale, use it as preprocessed
        if len(image.shape) == 2:
            preprocessed = image
            # Create a blank BGR image for color detection
            original = np.zeros((preprocessed.shape[0], preprocessed.shape[1], 3), dtype=np.uint8)
        else:
            # Otherwise, preprocess the color image
            original = image.copy()
            preprocessed = self.preprocess_image(image)
        
        # Apply multiple detection methods
        centers_hough, radii_hough = self.detect_eggs_hough(preprocessed)
        centers_contour, radii_contour = self.detect_eggs_contour(preprocessed)
        centers_color, radii_color = self.detect_eggs_color(original, preprocessed)
        
        # Combine results from different methods
        all_centers = centers_hough + centers_contour + centers_color
        all_radii = radii_hough + radii_contour + radii_color
        
        # Non-maximum suppression to remove overlapping detections
        final_centers, final_radii = self._non_max_suppression(all_centers, all_radii)
        
        return final_centers, final_radii

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
        centers = np.array(centers, dtype=np.float64)  # Use float64 to prevent overflow
        radii = np.array(radii, dtype=np.float64)      # Use float64 to prevent overflow
        
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
                x1, y1 = centers[i].astype(np.float64)  # Cast to float64 to prevent overflow
                x2, y2 = centers[j].astype(np.float64)  # Cast to float64 to prevent overflow
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
        final_centers = [tuple(map(int, centers[i])) for i in keep]  # Convert back to int tuples
        final_radii = [int(radii[i]) for i in keep]  # Convert back to int
        
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
            f"Total eggs: {len(centers)}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )
        
        return result