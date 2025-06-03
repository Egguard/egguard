import rclpy
import os
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import time
import threading
import requests
import json
from ament_index_python.packages import get_package_share_directory
from rclpy.qos import ReliabilityPolicy, QoSProfile

class PredatorDetector(Node):
    """
    ROS 2 node for detecting predators in camera images using a remote ML inference API.
    Periodically processes incoming images, checks for predators via API, and sends alerts.

    Attributes:
        bridge (CvBridge): ROS–OpenCV bridge.
        latest_frame (np.ndarray): Last received camera frame.
        last_detection_time (float): Timestamp of the last detection.
        frame_lock (threading.Lock): Lock for thread-safe access to latest_frame.
    """
    def __init__(self):
        """
        Initialize the PredatorDetector node, declare parameters, load model, and set up subscriptions and timers.

        ROS parameters:
            confidence_threshold (float): Detection confidence threshold.
            check_interval (float): Seconds between model inferences.
            cooldown (float): Minimum seconds between alerts.
            backend_url (str): Base URL for the backend API.
            backend_active (bool): Whether to attempt to send alerts to the backend.
            ml_api_url (str): URL for the ML inference API.
        """
        super().__init__('predator_detector')

        # Declare and read parameters
        self.declare_parameter('confidence_threshold', 0.70)
        self.declare_parameter('check_interval', 5.0)
        self.declare_parameter('cooldown', 10.0)
        self.declare_parameter('backend_url', 'http://localhost:8081/api/v1')
        self.declare_parameter('backend_active', True)  # New parameter
        self.declare_parameter('ml_api_url', 'http://localhost:8501')  # ML inference API

        self.confidence_threshold = self.get_parameter('confidence_threshold').get_parameter_value().double_value
        self.check_interval = self.get_parameter('check_interval').get_parameter_value().double_value
        self.cooldown = self.get_parameter('cooldown').get_parameter_value().double_value
        self.backend_url_base = self.get_parameter('backend_url').get_parameter_value().string_value
        self.is_backend_running = self.get_parameter('backend_active').get_parameter_value().bool_value # Read new parameter
        self.ml_api_url = self.get_parameter('ml_api_url').get_parameter_value().string_value

        # Initialize utilities
        self.bridge = CvBridge()
        self.latest_frame = None
        self.last_detection_time = 0.0
        self.frame_lock = threading.Lock() # Initialize the lock
        
        # Predator detection configuration
        self.predator_labels = ['aguila', 'mapache', 'zorro']
        self.predator_messages = {
            "aguila": "¡Se ha detectado un aguila en tu granja!",
            "mapache": "¡Se ha detectado un mapache en tu granja!",
            "zorro": "¡Se ha detectado un zorro en tu granja!"
        }
        self.notification_severity = "WARNING"

        self.sub = self.create_subscription(
            Image,
            '/image',
            self.image_callback,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.get_logger().info("[DEBUG] Subscribed to /camera/image_raw topic.")

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
        Periodically called by timer. Sends the latest frame to ML API for inference.
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

        """
        # For testing: load a saved image instead of using the live frame
        test_image_path = os.path.join(get_package_share_directory('egguard_predator_detector'), 'testing', 'zorro.jpg')

        frame_to_process = cv2.imread(test_image_path)
        if frame_to_process is None:
            self.get_logger().error(f"[ERROR] Could not load test image from {test_image_path}")
            return
        """

        _, img_encoded = cv2.imencode('.jpg', frame_to_process)
        if img_encoded is None:
            self.get_logger().error("[ERROR] Failed to encode frame for ML API")
            return

        img_bytes = img_encoded.tobytes()
        
        try:
            files = {'file': ('frame.jpg', img_bytes, 'image/jpeg')}
            response = requests.post(f"{self.ml_api_url}/predict", files=files, timeout=5.0)
            
            if response.status_code != 200:
                self.get_logger().error(f"[ERROR] ML API request failed: {response.status_code} - {response.text}")
                return
                
            result = response.json()
            label = result.get('prediction', 'unknown')
            prob = result.get('probability', 0.0)
            
            self.get_logger().debug(f"[DEBUG] ML API result: {label} (prob={prob:.2f})")
            
            if label in self.predator_labels and prob >= self.confidence_threshold:
                self.get_logger().info(f"[INFO] Detected predator: {label} (prob={prob:.2f})")
                self.last_detection_time = now
                threading.Thread(target=self.send_alert, args=(label, img_bytes)).start()
                
        except requests.exceptions.Timeout:
            self.get_logger().warn("[WARN] ML API request timed out")
        except requests.exceptions.ConnectionError:
            self.get_logger().warn("[WARN] Could not connect to ML API")
        except Exception as e:
            self.get_logger().error(f"[ERROR] Exception calling ML API: {e}")

    def send_alert(self, label: str, frame: bytes):
        """
        Sends an HTTP POST to the backend with the detection alert and image
        if self.is_backend_running is True. Otherwise, logs the information.

        Args:
            label (str): Detected predator label.
            prob (float): Detection probability.
            frame (bytes): The frame where the predator was detected.
        """
        
        message_text = self.predator_messages.get(label, f"¡Alerta de depredador! Se ha detectado un {label}.")

        notification_payload_dict = {
            "message": message_text,
            "severity": self.notification_severity 
        }
        
        # Create multipart form data properly
        files = {
            'notification': (None, json.dumps(notification_payload_dict), 'application/json')
        }
        
        if frame:
            files['image'] = ('frame.jpg', frame, 'image/jpeg')
        
        robot_id = 1
        alert_url = f"{self.backend_url_base}/robots/{robot_id}/notifications"

        self.get_logger().info(f"[DEBUG] Attempting to send alert. Backend active: {self.is_backend_running}")
        self.get_logger().info(f"[DEBUG] Alert URL: {alert_url}")
        self.get_logger().info(f"[DEBUG] Notification data (JSON string): {files}")

        if self.is_backend_running:
            try:
                self.get_logger().info(f"[INFO] Sending alert to backend: {alert_url}")
                resp = requests.post(alert_url, files=files)

                if resp.status_code == 201:
                    self.get_logger().info(f"[INFO] Alert sent successfully. Response: {resp.text}")
                else:
                    self.get_logger().error(f"[ERROR] Failed to send alert: {resp.status_code} - {resp.text}")
            except requests.exceptions.ConnectionError as e:
                self.get_logger().warn(f"[WARN] Could not connect to backend to send alert: {e}")
            except Exception as e:
                self.get_logger().error(f"[ERROR] Exception sending alert: {e}")
        else:
            self.get_logger().info("[INFO] Backend is not active. Alert not sent. Predator data logged above.")
            if frame:
                self.get_logger().info(f"[DEBUG] Image data size (if sent): {len(frame)} bytes")

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
