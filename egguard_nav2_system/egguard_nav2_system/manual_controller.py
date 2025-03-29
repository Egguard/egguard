"""
Manual navigation controller
"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from egguard_custom_interfaces.msg import Mode
import time
from egguard_mode_manager import qos_config

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
                # Create the subscription only once when switching to manual mode.
                self.manual_nav_subscription = self.create_subscription(
                    PoseStamped,
                    '/manual_nav',
                    self.manual_nav_callback,
                    self.qos_profile
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
        This is where the message is converted and published to /cmd_vel

        Parameters:
        -----------
        msg : ManualNav
            Message containing the manual navigation isntructions.
            it contains velocity, direction and stopNow.
            
            velocity is an integer between 0 and 100
            direction can be "left", "right", "forward"
            stopNow is a boolean indicating whether to stop the robot immediately.
        """
        # For now, we'll just log the received message.
        self.get_logger().info(f"Received manual nav instruction: {msg}")

        # TODO: Map the PoseStamped to the appropriate velocity commands or actions.
        # e.g., convert msg.pose into /cmd_vel commands.
        # I have to somehow map what I'm gonna receive to the topic /cmd_vel
        # This is a placeholder for the actual implementation

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

