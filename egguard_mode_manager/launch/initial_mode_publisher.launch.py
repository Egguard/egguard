"""
Launch configuration for the initial_mode_publisher node.
"""
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description() -> LaunchDescription:
    """
    Generate the launch description for the initial_mode_publisher node.

    Returns:
        LaunchDescription: The launch description containing the node configuration.
    """
    return LaunchDescription([
        Node(
            package='egguard_mode_manager',
            executable='initial_mode_publisher',
            name='initial_mode_publisher',
            output='screen',
        ),
    ])
