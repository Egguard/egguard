"""
Launch file for the egg detection system.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch_ros.actions import Node

def generate_launch_description() -> LaunchDescription:
    """
    Generates the launch description for egg detection system
    
    Returns:
        LaunchDescription: The configured launch description.
    """
    # Declare launch arguments
    detection_interval = LaunchConfiguration('detection_interval')
    use_test_publisher = LaunchConfiguration('use_test_publisher')
    publish_rate = LaunchConfiguration('publish_rate')
    
    # Create launch description
    return LaunchDescription([
        # Declare launch arguments
        DeclareLaunchArgument(
            'detection_interval',
            default_value='1.0',
            description='Interval between detections in seconds'
        ),
        DeclareLaunchArgument(
            'use_test_publisher',
            default_value='true',
            description='Whether to use the test image publisher'
        ),
        DeclareLaunchArgument(
            'publish_rate',
            default_value='1.0',
            description='Rate at which to publish test images'
        ),
        
        # Main detection node
        Node(
            package='egguard_computer_vision',
            executable='egg_detection_node',
            name='egg_detection_node',
            parameters=[{'detection_interval': detection_interval}],
            output='screen'
        ),
        
        # Test image publisher node
        Node(
            package='egguard_computer_vision',
            executable='test_image_publisher',
            name='test_image_publisher',
            parameters=[{'publish_rate': publish_rate}],
            condition=IfCondition(use_test_publisher),
            output='screen'
        )
    ])