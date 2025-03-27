import rclpy
from rclpy.node import Node
from egguard_custom_interfaces.msg import Mode
class InitialModePublisher(Node):
    def __init__(self):
        super().__init__('initial_mode_publisher')
        self.publisher = self.create_publisher(Mode, '/nav_mode', 10)
        
        # Publicar el modo inicial
        msg = Mode()
        msg.mode = 'autonomous'  # Modo inicial
        self.publisher.publish(msg)
        self.get_logger().info('Modo inicial publicado: autonomous')

def main(args=None):
    rclpy.init(args=args)
    node = InitialModePublisher()
    rclpy.spin_once(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()