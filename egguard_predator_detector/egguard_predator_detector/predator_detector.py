import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import time
import threading
import requests
import json  # Added import
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input # Changed to MobileNetV3
from tensorflow.keras.preprocessing import image # Added for img_to_array
from datetime import datetime, timezone # Added for ISO timestamp
from ament_index_python.packages import get_package_share_directory # Added import
import os # Added import

class PredatorDetector(Node):
    """
    ROS 2 node for detecting predators in camera images using a Keras model.
    Periodically processes incoming images, checks for predators, and sends alerts.

    Attributes:
        model (keras.Model): Loaded predator detection model.
        bridge (CvBridge): ROS–OpenCV bridge.
        latest_frame (np.ndarray): Last received camera frame.
        last_detection_time (float): Timestamp of the last detection.
        frame_lock (threading.Lock): Lock for thread-safe access to latest_frame.
    """
    def __init__(self):
        """
        Initialize the PredatorDetector node, declare parameters, load model, and set up subscriptions and timers.

        ROS parameters:
            model_path (str): Path to the .keras model file.
            confidence_threshold (float): Detection confidence threshold.
            check_interval (float): Seconds between model inferences.
            cooldown (float): Minimum seconds between alerts.
            backend_url (str): Base URL for the backend API.
            backend_active (bool): Whether to attempt to send alerts to the backend.
        """
        super().__init__('predator_detector')

        # Default model path relative to package share
        default_model_path = os.path.join(
            get_package_share_directory('egguard_predator_detector'),
            'models',
            'predator_model.keras'
        )
        self.declare_parameter('model_path', default_model_path)

        # Declare and read parameters
        self.declare_parameter('confidence_threshold', 0.80)
        self.declare_parameter('check_interval', 3.0)
        self.declare_parameter('cooldown', 10.0)
        self.declare_parameter('backend_url', 'http://localhost:8080')
        self.declare_parameter('backend_active', False)  # New parameter

        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.confidence_threshold = self.get_parameter('confidence_threshold').get_parameter_value().double_value
        self.check_interval = self.get_parameter('check_interval').get_parameter_value().double_value
        self.cooldown = self.get_parameter('cooldown').get_parameter_value().double_value
        self.backend_url_base = self.get_parameter('backend_url').get_parameter_value().string_value
        self.is_backend_running = self.get_parameter('backend_active').get_parameter_value().bool_value # Read new parameter

        # Initialize utilities
        self.bridge = CvBridge()
        self.latest_frame = None
        self.last_detection_time = 0.0
        self.frame_lock = threading.Lock() # Initialize the lock

        # Load Keras model
        self.get_logger().info(f"[DEBUG] Loading model from: {model_path}")
        self.model = load_model(model_path)
        self.get_logger().info("[DEBUG] Model loaded successfully.")

        # Subscribe to camera topic
        self.sub = self.create_subscription(
            Image,
            '/image',
            self.image_callback,
            10)
        self.get_logger().info("[DEBUG] Subscribed to /camera/image_raw topic.")

        # Timer for periodic predator checks
        self.timer = self.create_timer(self.check_interval, self.check_for_predators)
        self.get_logger().info(f"[DEBUG] Timer started: checking every {self.check_interval} seconds.")

    def image_callback(self, msg: Image):
        """
        Callback for the camera image topic. Converts ROS image to OpenCV format.

        Args:
            msg (sensor_msgs.msg.Image): Incoming image message.
        """
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            with self.frame_lock: # Acquire lock before modifying latest_frame
                self.latest_frame = cv_image.copy()
            self.get_logger().debug("[DEBUG] Received and stored latest frame.")
        except CvBridgeError as e:
            self.get_logger().error(f"[ERROR] CvBridge failure: {e}")

    def check_for_predators(self):
        """
        Periodically called by timer. Runs the model on the latest frame if cooldown has elapsed.
        If a predator is detected above the confidence threshold, triggers an alert.
        """
        frame_to_process = None
        with self.frame_lock: # Acquire lock before reading latest_frame
            if self.latest_frame is not None:
                frame_to_process = self.latest_frame.copy()

        if frame_to_process is None:
            return

        now = time.time()
        if now - self.last_detection_time < self.cooldown:
            return

        # Preprocess image for model
        img_resized = cv2.resize(frame_to_process, (224, 224)) # frame_to_process is OpenCV BGR
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB (still an OpenCV image/Numpy array)

        # Align with Colab preprocessing style using Keras image utilities
        img_array = image.img_to_array(img_rgb) # Converts to np.array, ensures float32
        img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
        img_processed = preprocess_input(img_array) # Using preprocess_input from MobileNetV3


        # Predict probabilities: [aguila, mapache, no_depredador, zorro]
        preds = self.model.predict(img_processed)[0]
        idx = np.argmax(preds)
        prob = preds[idx]

        predator_labels = {0: 'aguila', 1: 'mapache', 3: 'zorro'}
        if idx in predator_labels and prob >= self.confidence_threshold:
            label = predator_labels[idx]
            self.get_logger().info(f"[INFO] Detected predator: {label} (prob={prob:.2f})")
            self.last_detection_time = now
            threading.Thread(target=self.send_alert, args=(label, prob)).start()

    def send_alert(self, label: str, prob: float):
        """
        Sends an HTTP POST to the backend with the detection alert and image
        if self.is_backend_running is True. Otherwise, logs the information.

        Args:
            label (str): Detected predator label.
            prob (float): Confidence score of the detection.
        """
        
        predator_messages = {
            "aguila": "Se ha encontrado un aguila! Ten cuidado para que no se lleve tus huevos.",
            "mapache": "Se ha encontrado un mapache! Asegura tus contenedores de basura y protege a las gallinas.",
            "zorro": "Se ha encontrado un zorro! Mantén a salvo a tus aves."
        }
        message_text = predator_messages.get(label, f"¡Alerta de depredador! Se ha detectado: {label}.")

        # Construct the RegisterNotificationRequest payload
        notification_payload_dict = {
            "message": message_text,
            "severity": "WARNING" 
        }
        
        img_encoded_bytes = None
        with self.frame_lock: # Acquire lock before accessing latest_frame
            if self.latest_frame is not None:
                _, img_encoded = cv2.imencode('.jpg', self.latest_frame)
                if img_encoded is not None:
                    img_encoded_bytes = img_encoded.tobytes()
            else:
                self.get_logger().warn("[WARN] Latest frame is None, cannot encode image for alert.")
                # Decide if you want to send an alert without an image or not send at all
                # For now, we'll allow sending without an image if encoding fails or frame is None
        
        payload_files = None
        if img_encoded_bytes:
            payload_files = {'image': ('frame.jpg', img_encoded_bytes, 'image/jpeg')}
        
        # The backend expects the JSON data as a form field named 'notification'
        payload_data = {'notification': json.dumps(notification_payload_dict)}
        
        robot_id = 1 # This might be part of farmId or another ID in the new payload. Kept for URL structure.
        alert_url = f"{self.backend_url_base}/robots/{robot_id}/notifications"

        self.get_logger().info(f"[DEBUG] Attempting to send alert. Backend active: {self.is_backend_running}")
        self.get_logger().info(f"[DEBUG] Alert URL: {alert_url}")
        self.get_logger().info(f"[DEBUG] Notification data (JSON string): {payload_data['notification']}")

        if self.is_backend_running:
            try:
                self.get_logger().info(f"[INFO] Sending alert to backend: {alert_url}")
                # Send the JSON as a form field 'notification' and the image as a file
                resp = requests.post(alert_url, data=payload_data, files=payload_files)
                if resp.status_code == 200:
                    self.get_logger().info(f"[INFO] Alert sent successfully. Response: {resp.text}")
                else:
                    self.get_logger().error(f"[ERROR] Failed to send alert: {resp.status_code} - {resp.text}")
            except requests.exceptions.ConnectionError as e:
                self.get_logger().warn(f"[WARN] Could not connect to backend to send alert: {e}")
            except Exception as e:
                self.get_logger().error(f"[ERROR] Exception sending alert: {e}")
        else:
            self.get_logger().info("[INFO] Backend is not active. Alert not sent. Predator data logged above.")
            # Log the image data size as an example of what could be sent
            if img_encoded_bytes:
                self.get_logger().info(f"[DEBUG] Image data size (if sent): {len(img_encoded_bytes)} bytes")

def main(args=None):
    """
    Entry point for the predator_detector node.
    Initializes ROS, spins the node, and handles shutdown.
    """
    rclpy.init(args=args)
    node = PredatorDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("[INFO] Shutting down predator_detector node.")
    finally:
        node.destroy_node()
        rclpy.shutdown()
