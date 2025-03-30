"""
Initial mode publisher for the robot.

Publishes the 'autonomous' mode to the '/mode' topic.
"""
import rclpy
from rclpy.node import Node
from egguard_custom_interfaces.msg import Mode
from egguard_mode_manager import qos_config
from rclpy.exceptions import ROSInterruptException


class InitialModePublisher(Node):
    """
    Node that publishes the robot's initial mode ('autonomous') to the '/mode' topic.
    """

    def __init__(self) -> None:
        """
        Initializes the node and publishes the 'autonomous' mode.
        """
        try:
            super().__init__('initial_mode_publisher')
            qos_profile = qos_config.get_common_qos_profile()
            self.publisher = self.create_publisher(Mode, '/mode', qos_profile)

            msg = Mode()
            msg.mode = 'autonomous'
            self.publisher.publish(msg)
            self.get_logger().info('Initial mode published: autonomous')
        
        except Exception as e:
            self.get_logger().error(f"Error: {e}")


def main(args=None) -> None:
    """
    Initializes and runs the InitialModePublisher node.
    """
    try:
        rclpy.init(args=args)
        node = InitialModePublisher()
        rclpy.spin_once(node)
        node.destroy_node()
        rclpy.shutdown()
    
    except ROSInterruptException:
        pass
    except Exception as e:
        print(f"Error: {e}")
        rclpy.shutdown()


if __name__ == '__main__':
    main()
