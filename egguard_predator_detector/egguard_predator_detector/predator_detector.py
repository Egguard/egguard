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

class PredatorDetector(Node):
    """
    ROS 2 node for detecting predators in camera images using a Keras model.
    Periodically processes incoming images, checks for predators, and sends alerts.

    Attributes:
        model (keras.Model): Loaded predator detection model.
        bridge (CvBridge): ROS–OpenCV bridge.
        latest_frame (np.ndarray): Last received camera frame.
        last_detection_time (float): Timestamp of the last detection.
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

        # Declare and read parameters
        self.declare_parameter('model_path', 'models/predator_model.keras')
        self.declare_parameter('confidence_threshold', 0.80)
        self.declare_parameter('check_interval', 3.0)
        self.declare_parameter('cooldown', 10.0)
        self.declare_parameter('backend_url', 'http://localhost:8080')
        self.declare_parameter('backend_active', True)  # New parameter

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

        # Load Keras model
        self.get_logger().info(f"[DEBUG] Loading model from: {model_path}")
        self.model = load_model(model_path)
        self.get_logger().info("[DEBUG] Model loaded successfully.")

        # Subscribe to camera topic
        self.sub = self.create_subscription(
            Image,
            '/camera/image_raw',
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
            self.latest_frame = cv_image.copy()
            self.get_logger().debug("[DEBUG] Received and stored latest frame.")
        except CvBridgeError as e:
            self.get_logger().error(f"[ERROR] CvBridge failure: {e}")

    def check_for_predators(self):
        """
        Periodically called by timer. Runs the model on the latest frame if cooldown has elapsed.
        If a predator is detected above the confidence threshold, triggers an alert.
        """
        if self.latest_frame is None:
            return

        now = time.time()
        if now - self.last_detection_time < self.cooldown:
            return

        # Preprocess image for model
        img = cv2.resize(self.latest_frame, (224, 224))
        img_array = img.astype('float32') / 255.0
        img_batch = np.expand_dims(img_array, axis=0)

        # Predict probabilities: [aguila, mapache, no_depredador, zorro]
        preds = self.model.predict(img_batch)[0]
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
        alert_text = f"Detected predator: {label} with confidence {prob:.2f}"
        notification_data = {"text": alert_text}
        _, img_encoded = cv2.imencode('.jpg', self.latest_frame)
        payload_files = {'image': ('frame.jpg', img_encoded.tobytes(), 'image/jpeg')}
        payload_data = {'notification': json.dumps(notification_data)}
        robot_id = 1
        alert_url = f"{self.backend_url_base}/robots/{robot_id}/notifications"

        self.get_logger().info(f"[DEBUG] Attempting to send alert. Backend active: {self.is_backend_running}")
        self.get_logger().info(f"[DEBUG] Alert URL: {alert_url}")
        self.get_logger().info(f"[DEBUG] Notification data (JSON string): {payload_data['notification']}")

        if self.is_backend_running:
            try:
                self.get_logger().info(f"[INFO] Sending alert to backend: {alert_url}")
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
            if img_encoded is not None:
                self.get_logger().info(f"[DEBUG] Image data size (if sent): {len(img_encoded.tobytes())} bytes")


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
