"""
Launch file for the ManualController node.
"""
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description() -> LaunchDescription:
    """
    Generates the launch description for manual navigation
    
    Returns:
        LaunchDescription: The configured launch description.
    """
    return LaunchDescription([
        Node(
            package='egguard_nav2_system',  # Replace with the actual package name containing your node
            executable='manual_controller',  # Replace with the executable name as defined in your setup/configuration
            name='manual_controller',
            output='screen'
        )
    ])