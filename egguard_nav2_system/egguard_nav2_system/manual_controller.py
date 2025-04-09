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
from typing import Optional

class ManualController(Node):
    """
    A ROS 2 node that controls Manual navigation listening to the topic /manual_nav.
    """

    def __init__(self) -> None:
        """
        Initializes the ManualController node, subscribes to the /mode topic, ...
        """
        super().__init__('manual_controller')

        self.mode: str = "autonomous"

        self.manual_nav_subscription: Optional[rclpy.subscription.Subscription] = None
        self.qos_profile = qos_config.get_common_qos_profile()
        self.cmd_vel_publisher: rclpy.publisher.Publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        
        self.mode_subscription: rclpy.subscription.Subscription = self.create_subscription(
            Mode,
            '/mode',
            self.mode_callback,
            self.qos_profile
        )

    def mode_callback(self, msg: Mode) -> None:
        """
        Callback method that updates the robot's mode when a message is received on /mode.

        Parameters:
        -----------
        msg : Mode
            Message containing the robot's current mode ("manual", "autonomous", "emergency").
        """
        self.mode = msg.mode
        self.get_logger().info(f"Current mode: {self.mode}")

        if self.mode == "manual" and self.manual_nav_subscription is None:
            manual_nav_qos = get_manual_nav_qos_profile()
            # Create the subscription only once when switching to manual mode.
            self.manual_nav_subscription = self.create_subscription(
                ManualNav,
                '/manual_nav',
                self.manual_nav_callback,
                manual_nav_qos
            )
            self.get_logger().info("Started listening to /manual_nav for manual instructions...")
        elif self.mode != "manual" and self.manual_nav_subscription is not None:
            #It is important to destroy suscription first so that manual_nav_callback doesnt get executed and mess up
            self.destroy_subscription(self.manual_nav_subscription)
            self.manual_nav_subscription = None
            self.get_logger().info("Stopped listening to /manual_nav (switched mode).")

            #Stop robot with cmd_vel
            linear_x = 0.0
            angular_z = 0.0

            twist_msg = Twist()
            twist_msg.linear.x = linear_x
            twist_msg.angular.z = angular_z
            self.cmd_vel_publisher.publish(twist_msg)

    def manual_nav_callback(self, msg: ManualNav) -> None:
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
        max_linear_velocity: float = 0.22  # meters per second
        constant_angular_velocity: float = 0.40  # radians per second (the max is 2.84)

        linear_x: float = 0.0
        angular_z: float  = 0.0

        if msg.stop_now:
            self.get_logger().info("Received stop command. Stopping the robot.")
        else:
            linear_x = (msg.velocity / 100.0) * max_linear_velocity

            # TODO: stop hardcoding directions and add enum
            if msg.direction == "forward":
                angular_z = 0.0
            elif msg.direction == "left":
                angular_z = constant_angular_velocity*(linear_x/max_linear_velocity) 
            elif msg.direction == "right":
                angular_z = -constant_angular_velocity*(linear_x/max_linear_velocity)
            else:
                self.get_logger().warn(f"Unknown direction '{msg.direction}'. Stopping the robot.")
                linear_x = 0.0
                angular_z = 0.0

        twist_msg = Twist()
        twist_msg.linear.x = linear_x
        twist_msg.angular.z = angular_z
        self.cmd_vel_publisher.publish(twist_msg)

        self.get_logger().info(f"Published cmd_vel: linear_x={linear_x:.2f}, angular_z={angular_z:.2f}")

def main(args: Optional[list] = None) -> None:
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

