"""
Manual navigation controller
"""
import rclpy
from rclpy.node import Node
from egguard_custom_interfaces.msg import ManualNav
from egguard_custom_interfaces.msg import Mode
import time
from egguard_mode_manager import qos_config
from geometry_msgs.msg import Twist
from .manual_qos_config import get_manual_nav_qos_profile

class ManualController(Node):
    """
    A ROS 2 node that controls Manual navigation listening to the topic /manual_nav.
    """

    def __init__(self):
        """
        Initializes the ManualController node, subscribes to the /mode topic, ...
        """
        super().__init__('manual_controller')

        self.mode = "autonomous"

        self.manual_nav_subscription = None

        self.qos_profile = qos_config.get_common_qos_profile()

        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        self.mode_subscription = self.create_subscription(
            Mode,
            '/mode',
            self.mode_callback,
            self.qos_profile
        )

        # Timer to check mode periodically
        self.timer = self.create_timer(1.0, self.check_mode_and_navigate)

        self.last_feedback_print_time = time.time() 
        self.feedback_print_interval = 2  # Print feedback every 2 seconds

    def mode_callback(self, msg):
        """
        Callback method that updates the robot's mode when a message is received on /mode.

        Parameters:
        -----------
        msg : Mode
            Message containing the robot's current mode ("manual", "autonomous", "emergency").
        """
        self.mode = msg.mode
        self.get_logger().info(f"Current mode: {self.mode}")

    def check_mode_and_navigate(self):
        """
        Periodically checks if the robot is in "manual" mode. 
        If so, it ensures that a subscription to /manual_nav exists.
        If not, it removes the subscription to stop manual activity.
        """
        if self.mode == "manual":
            if self.manual_nav_subscription is None:
                manual_nav_qos = get_manual_nav_qos_profile()
                # Create the subscription only once when switching to manual mode.
                self.manual_nav_subscription = self.create_subscription(
                    ManualNav,
                    '/manual_nav',
                    self.manual_nav_callback,
                    manual_nav_qos
                )
                self.get_logger().info("Started listening to /manual_nav for manual instructions...")
        else:
            # If not in manual mode and a subscription exists, unregister it.
            if self.manual_nav_subscription is not None:
                self.destroy_subscription(self.manual_nav_subscription)
                self.manual_nav_subscription = None
                self.get_logger().info("Stopped listening to /manual_nav (switched mode).")
            # TODO: Implement logic to stop the robot or switch to another mode if needed.       
            pass

    def manual_nav_callback(self, msg):
        """
        Callback for processing incoming ManualNav messages on /manual_nav.
        Converts the received message into velocity commands and publishes to /cmd_vel.

        Parameters:
        -----------
        msg : ManualNav
            Message containing the manual navigation instructions.
            Attributes:
            - velocity: int (0 to 100) representing the desired speed percentage.
            - direction: str, one of "left", "right", "forward".
            - stop_now: bool indicating whether to stop the robot immediately.
        """
        # Maximum velocities for TurtleBot3 Burger
        max_linear_velocity = 0.22  # meters per second
        max_angular_velocity = 2.84  # radians per second

        linear_x = 0.0
        angular_z = 0.0

        if msg.stop_now:
            self.get_logger().info("Received stop command. Stopping the robot.")
        else:
            linear_x = (msg.velocity / 100.0) * max_linear_velocity

            if msg.direction == "forward":
                angular_z = 0.0
            elif msg.direction == "left":
                angular_z = max_angular_velocity
            elif msg.direction == "right":
                angular_z = -max_angular_velocity
            else:
                self.get_logger().warn(f"Unknown direction '{msg.direction}'. Stopping the robot.")
                linear_x = 0.0
                angular_z = 0.0

        twist_msg = Twist()
        twist_msg.linear.x = linear_x
        twist_msg.angular.z = angular_z
        self.cmd_vel_publisher.publish(twist_msg)

        self.get_logger().info(f"Published cmd_vel: linear_x={linear_x:.2f}, angular_z={angular_z:.2f}")

def main(args=None) -> None:
    """
    Main entry point of the ROS 2 node. Initializes rclpy, 
    creates an instance of AutonomousController, and starts the event loop.
    """
    try:
        rclpy.init()
        controller = ManualController()
        rclpy.spin(controller)
    finally: 
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

