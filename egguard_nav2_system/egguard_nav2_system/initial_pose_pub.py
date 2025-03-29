"""
Initial pose publisher for setting the robot's starting position in Nav2.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped

class InitialPosePublisher(Node):
    """Publishes the initial pose of the robot to the 'initialpose' topic."""

    def __init__(self) -> None:
        super().__init__('initial_pose_publisher')
        self.publisher_ = self.create_publisher(PoseWithCovarianceStamped, 'initialpose', 1)
        self.initial_pose_msg = self._create_initial_pose_msg()

        self.timer_ = self.create_timer(0.5, self.publish_initial_pose)

    def _create_initial_pose_msg(self) -> PoseWithCovarianceStamped:
        """
        Creates and returns the initial pose message.
        
        Returns:
            PoseWithCovarianceStamped: The initial pose message.
        """
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = 'map'
        msg.pose.pose.position.x = 3.2
        msg.pose.pose.position.y = 0.0
        msg.pose.pose.orientation.w = 1.0  # Quaternion must be normalized
        return msg

    def publish_initial_pose(self) -> None:
        """Publishes the initial pose message."""
        self.get_logger().info(f'Publishing Initial Position: X={self.initial_pose_msg.pose.pose.position.x}, '
                               f'Y={self.initial_pose_msg.pose.pose.position.y}, W={self.initial_pose_msg.pose.pose.orientation.w}')
        self.publisher_.publish(self.initial_pose_msg)

def main(args=None) -> None:
    """Initializes the node and starts publishing the initial pose."""
    rclpy.init(args=args)
    node = InitialPosePublisher()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
