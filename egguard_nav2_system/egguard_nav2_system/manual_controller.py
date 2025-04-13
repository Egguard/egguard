"""
Manual navigation controller with automatic timeout
"""
import rclpy
from rclpy.node import Node
from egguard_custom_interfaces.msg import ManualNav
from egguard_custom_interfaces.msg import Mode
from egguard_mode_manager.utils import mode_qos_config, modes
from geometry_msgs.msg import Twist
from .utils import manual_nav_qos_config, directions
from typing import Optional
import time

class ManualController(Node):
    """
    A ROS 2 node that controls Manual navigation listening to the topic /manual_nav.
    Includes timeout functionality to return to autonomous mode after inactivity.
    """

    def __init__(self) -> None:
        """
        Initializes the ManualController node, subscribes to the /mode topic, ...
        """
        super().__init__('manual_controller')

        self.mode: str = modes.Mode.AUTONOMOUS
        self.last_manual_command_time: float = 0.0
        self.manual_timeout: float = 30.0  # Timeout in seconds

        self.manual_nav_subscription: Optional[rclpy.subscription.Subscription] = None

        self.cmd_vel_publisher: rclpy.publisher.Publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        
        self.qos_profile = mode_qos_config.get_mode_qos_profile()
        self.mode_subscription: rclpy.subscription.Subscription = self.create_subscription(
            Mode,
            '/mode',
            self.mode_suscription_callback,
            self.qos_profile
        )

        # Create a publisher for the mode topic to be able to switch back to autonomous
        self.mode_publisher: rclpy.publisher.Publisher = self.create_publisher(
            Mode,
            '/mode',
            self.qos_profile
        )

        # Maximum velocities for TurtleBot3 Burger
        self.max_linear_velocity: float = 0.22  # meters per second
        self.constant_angular_velocity: float = 0.40  # radians per second (the max is 2.84)
        
        # Create a timer to check for timeout
        self.timer = self.create_timer(1.0, self.check_manual_navigation_timeout)

    def mode_suscription_callback(self, msg: Mode) -> None:
        """
        Callback method that updates the robot's mode when a message is received on /mode.

        Parameters:
        -----------
        msg : Mode
            Message containing the robot's current mode ("manual", "autonomous", "emergency").
        """
        self.mode = msg.mode
        self.get_logger().info(f"Current mode: {self.mode}")

        if self.mode == modes.Mode.MANUAL and self.manual_nav_subscription is None:
            self.start_manual_navigation()
        elif self.mode != modes.Mode.MANUAL and self.manual_nav_subscription is not None:
            self.stop_manual_navigation()

    def start_manual_navigation(self) -> None:
        """
        Starts manual navigation by creating a subscription to the /manual_nav topic.
        This method is called when the robot enters MANUAL mode.
        """
        manual_nav_qos = manual_nav_qos_config.get_manual_nav_qos_profile()
        # Create the subscription only once when switching to manual mode.
        self.manual_nav_subscription = self.create_subscription(
            ManualNav,
            '/manual_nav',
            self.manual_nav_suscription_callback,
            manual_nav_qos
        )
        
        # Record the time when manual mode started
        self.last_manual_command_time = time.time()
        
        self.get_logger().info("Started listening to /manual_nav for manual instructions...")

    def stop_manual_navigation(self) -> None:
        """
        Stops manual navigation by destroying the subscription to the /manual_nav topic
        and sending a stop command to the robot. This method is called when the robot
        exits MANUAL mode.

        It is important to destroy subscription first so that manual_nav_suscription_callback 
        doesnt get executed and mess up
        """
        if self.manual_nav_subscription:
            self.destroy_subscription(self.manual_nav_subscription)
            self.manual_nav_subscription = None
            self.get_logger().info("Stopped listening to /manual_nav (switched mode).")

            # Stop robot with cmd_vel
            self.publish_twist(0.0, 0.0)

    def publish_twist(self, linear_x: float, angular_z: float) -> None:
        """
        Publishes a Twist message to the /cmd_vel topic with the specified linear and angular velocities.

        Parameters:
        -----------
        linear_x : float
            Linear velocity along the x-axis in meters per second.
        angular_z : float
            Angular velocity around the z-axis in radians per second.
        """
        twist_msg = Twist()
        twist_msg.linear.x = linear_x
        twist_msg.angular.z = angular_z
        self.cmd_vel_publisher.publish(twist_msg)
        self.get_logger().info(f"Published cmd_vel: linear_x={linear_x:.2f}, angular_z={angular_z:.2f}")

    def manual_nav_suscription_callback(self, msg: ManualNav) -> None:
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
        self.last_manual_command_time = time.time()
        
        linear_x: float = 0.0
        angular_z: float = 0.0

        if msg.stop_now is True:
            self.get_logger().info("Received stop command. Stopping the robot.")
        else:
            linear_x = (msg.velocity / 100.0) * self.max_linear_velocity

            if msg.direction == directions.Direction.FORWARD:
                angular_z = 0.0
            elif msg.direction == directions.Direction.LEFT:
                if msg.velocity == 0:
                    angular_z = self.constant_angular_velocity  # Use constant value for in-place rotation
                else:
                    angular_z = self.constant_angular_velocity * (linear_x/self.max_linear_velocity)
            elif msg.direction == directions.Direction.RIGHT:
                if msg.velocity == 0:
                    angular_z = -self.constant_angular_velocity  # Use constant value for in-place rotation
                else:
                    angular_z = -self.constant_angular_velocity * (linear_x/self.max_linear_velocity)
            else:
                self.get_logger().warn(f"Unknown direction '{msg.direction}'. Stopping the robot.")

        self.publish_twist(linear_x, angular_z)

    def check_manual_navigation_timeout(self) -> None:
        """
        Checks if the robot has been in manual mode with no activity for longer than
        the timeout period. If so, switches to autonomous mode.
        """
        if self.mode == modes.Mode.MANUAL:
            current_time = time.time()
            time_since_last_command = current_time - self.last_manual_command_time
            
            if time_since_last_command >= self.manual_timeout:
                self.get_logger().info(f"Manual mode timeout after {self.manual_timeout} seconds of inactivity. Switching to autonomous mode.")
                self.stop_manual_navigation()
                self.switch_to_autonomous_mode()
    
    def switch_to_autonomous_mode(self) -> None:
        """
        Switches the robot back to autonomous mode by publishing to the /mode topic.
        """
        try:
            mode_msg = Mode()
            mode_msg.mode = modes.Mode.AUTONOMOUS
            self.mode_publisher.publish(mode_msg)
            
            self.get_logger().info(f"Published autonomous mode to /mode topic due to inactivity timeout")
            
        except Exception as e:
            self.get_logger().error(f"Error switching to autonomous mode: {e}")

def main(args: Optional[list] = None) -> None:
    """
    Main entry point of the ROS 2 node. Initializes rclpy, 
    creates an instance of ManualController, and starts the event loop.
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