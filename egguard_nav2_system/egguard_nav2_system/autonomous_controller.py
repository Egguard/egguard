import rclpy
from rclpy.node import Node
from nav2_msgs.action import FollowWaypoints
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped
from egguard_custom_interfaces.msg import Mode
import math
import time
from pprint import pprint
from egguard_mode_manager import qos_config

class AutonomousController(Node):
    """
    A ROS 2 node that controls autonomous navigation using FollowWaypoints.
    """

    def __init__(self):
        """
        Initializes the AutonomousController node, subscribes to the /mode topic, 
        and sets up the action client for FollowWaypoints.
        It also initializes a list of waypoints and manages their execution.
        """
        super().__init__('autonomous_controller')

        # Action client to send a sequence of waypoints
        self.action_client = ActionClient(self, FollowWaypoints, 'follow_waypoints')

        # List of waypoints (x, y, yaw)
        self.waypoints = [
            (-2.0, 1.0, 1.0),
            (0.0, 2.0, 1.5),
            (3.0, -1.0, 0.0)
        ]

        self.mode = "manual"  # Default mode
        self.current_goal_future = None  # No active goal

        qos_profile = qos_config.get_common_qos_profile()

        # Subscribe to the /mode topic
        self.mode_subscription = self.create_subscription(
            Mode,
            '/mode',
            self.mode_callback,
            qos_profile
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
        Periodically checks if the robot is in "autonomous" mode. 
        If so, it sends the waypoints to the FollowWaypoints action server.
        """
        if self.mode == "autonomous" and self.current_goal_future is None:
            self.send_waypoints()
        else:
            #Stop all the autonomous activity and waypoints following
            pass

    def send_waypoints(self):
        """
        Converts the waypoints list into a sequence of PoseStamped messages 
        and sends them to the FollowWaypoints action server.
        """
        goal_msg = FollowWaypoints.Goal()
        goal_msg.poses = []

        for x, y, yaw in self.waypoints:
            pose = PoseStamped()
            pose.header.frame_id = "map"
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.orientation.w = math.cos(yaw / 2)  # Quaternion conversion
            
            goal_msg.poses.append(pose)

        # Wait for the action server
        self.action_client.wait_for_server()
        self.get_logger().info(f"Sending {len(self.waypoints)} waypoints for autonomous navigation.")

        # Send goal and add callbacks
        future = self.action_client.send_goal_async(goal_msg, feedback_callback=self.feedback_callback)
        future.add_done_callback(self.goal_response_callback)

        # Track the current goal
        self.current_goal_future = future

    def goal_response_callback(self, future):
        """
        Callback executed when the goal is accepted or rejected.

        Parameters:
        -----------
        future : Future
            The future object containing the goal response.
        """
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Waypoints goal rejected :(')
            self.current_goal_future = None
            return

        self.get_logger().info('Waypoints goal accepted :)')

        # Wait for the action to complete
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        """
        Callback executed when the waypoints navigation is completed.

        Parameters:
        -----------
        future : Future
            The future object containing the result.
        """
        self.get_logger().info("Waypoints navigation completed. Waiting for next round.")
        self.current_goal_future = None  # Reset to allow new navigation

    def feedback_callback(self, feedback_msg):
        """
        Callback executed when feedback is received from FollowWaypoints.
        It prints distance remaining and estimated time remaining every 2 seconds.

        Parameters:
        -----------
        feedback_msg : FollowWaypoints.Feedback
            Feedback message containing navigation progress.
        """
        current_time = time.time()

        # Check if 2 seconds have passed before printing feedback
        if current_time - self.last_feedback_print_time >= self.feedback_print_interval:
            pprint(feedback_msg)
            self.last_feedback_print_time = current_time


def main():
    """
    Main entry point of the ROS 2 node. Initializes rclpy, 
    creates an instance of AutonomousController, and starts the event loop.
    """
    rclpy.init()
    controller = AutonomousController()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

