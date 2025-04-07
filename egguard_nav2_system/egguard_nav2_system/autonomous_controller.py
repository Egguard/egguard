"""
Autonomous navigation controller
"""
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
            (2.0, 1.44, 1.0),
            (0.8, 7.0, 1.5),
            (0.2, -6.6, 0.0)
        ]

        self.mode = "manual"
        self.current_goal_future = None  # No active goal
        self.goal_handle = None  # Store the goal handle for cancellation

        qos_profile = qos_config.get_common_qos_profile()

        self.mode_subscription = self.create_subscription(
            Mode,
            '/mode',
            self.mode_callback,
            qos_profile
        )

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
        self.get_logger().info(f"Mode changed to: {self.mode}")

        if self.mode == "autonomous" and self.current_goal_future is None:
            self.get_logger().info("Sending waypoints in check_mode_and_navigate() method")
            self.send_waypoints()
        if self.mode != "autonomous" and self.current_goal_future is not None:
            #We may have a problem if the stop_autonomous_mode() fails or cancel not accepted
            self.stop_autonomous_mode()

    def stop_autonomous_mode(self):
        """
        Stops any ongoing autonomous navigation by canceling the current goal.
        This allows manual control to take over.
        """
        self.get_logger().info("Stopping autonomous navigation")
        
        # Cancel the goal if we have a valid goal handle
        if self.goal_handle is not None and self.goal_handle.is_active:
            self.get_logger().info("Canceling active navigation goal")
            cancel_future = self.goal_handle.cancel_goal_async()
            cancel_future.add_done_callback(self.cancel_done_callback)
        
    def cancel_done_callback(self, future):
        """
        Callback executed when a goal cancellation is completed.
        
        Parameters:
        -----------
        future : Future
            The future object from the cancel request.
        """
        cancel_response = future.result()
        if cancel_response.return_code == 1:  # SUCCESS
            self.current_goal_future = None
            self.get_logger().info('Successfully canceled navigation goal')
        else:
            self.get_logger().warning(f'Failed to cancel goal, return code: {cancel_response.return_code}')
        
        # Reset goal handle after cancellation attempt
        self.goal_handle = None

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
        
        # Store the goal handle for potential cancellation later
        self.goal_handle = goal_handle

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
        result = future.result().result
        self.get_logger().info(f"Waypoints navigation completed. Result: {result}")
        
        # Reset tracking variables
        self.current_goal_future = None
        self.goal_handle = None

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


def main(args=None) -> None:
    """
    Main entry point of the ROS 2 node. Initializes rclpy, 
    creates an instance of AutonomousController, and starts the event loop.
    """
    try:
        rclpy.init()
        controller = AutonomousController()
        rclpy.spin(controller)
    finally: 
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
