from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='egguard_mode_manager',
            executable='initial_mode_publisher',
            name='initial_mode_publisher',
            output='screen',
        ),
    ])