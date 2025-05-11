from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    """
    Generate launch description for the egg detection node.
    """
    # Launch arguments
    processing_interval_arg = DeclareLaunchArgument(
        'processing_interval',
        default_value='3.0',
        description='Interval between image processing in seconds'
    )
    
    debug_mode_arg = DeclareLaunchArgument(
        'debug_mode',
        default_value='true',
        description='Enable debug visualization'
    )
    
    # Create egg detection node
    egg_detection_node = Node(
        package='egguard_computer_vision',
        executable='egguard_node',
        name='egg_detection_node',
        parameters=[{
            'processing_interval': LaunchConfiguration('processing_interval'),
            'debug_mode': LaunchConfiguration('debug_mode'),
        }],
        output='screen'
    )
    
    # Return launch description
    return LaunchDescription([
        processing_interval_arg,
        debug_mode_arg,
        egg_detection_node
    ])