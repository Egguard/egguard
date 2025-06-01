from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    """
    Generate launch description for the predator detection node.
    """
    
    # Create predator detection node
    predator_detector_node = Node(
        package='egguard_predator_detector',
        executable='predator_detector',
        name='predator_detector_node',
        parameters=[{
            # To explicitly set backend_active to False for testing, you can add:
            # 'backend_active': False
            # However, since it defaults to False in your Python script, this is optional here.
        }],
        output='screen'
    )
    
    # Return launch description
    return LaunchDescription([
        predator_detector_node
    ])