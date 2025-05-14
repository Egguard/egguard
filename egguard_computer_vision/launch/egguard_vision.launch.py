from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    """
    Generate launch description for the egg detection node.
    """
    
    # Create egg detection node
    egg_detection_node = Node(
        package='egguard_computer_vision',
        executable='egguard_node',
        name='egg_detection_node',
        parameters=[{
        }],
        output='screen'
    )
    
    # Return launch description
    return LaunchDescription([
        egg_detection_node
    ])