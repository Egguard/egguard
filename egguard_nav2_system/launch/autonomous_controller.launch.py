"""
Launch file for initializing autonomous navigation
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description() -> LaunchDescription:
    """
    Generates the launch description for autonomous navigation
    
    Returns:
        LaunchDescription: The configured launch description.
    """
    nav2_yaml = os.path.join(get_package_share_directory('egguard_nav2_system'), 'config', 'my_nav2_params.yaml')
 
    return LaunchDescription([
        Node(
            package = 'nav2_planner',
            executable = 'planner_server',
            name = 'planner_server',
            output = 'screen',
            parameters=[nav2_yaml]
        ),
        Node(
            package='nav2_waypoint_follower',
            executable='waypoint_follower',
            name='waypoint_follower',
            output='screen',
            parameters=[nav2_yaml, {'use_sim_time': True}]
        ),
        Node(
            package='nav2_controller',
            executable='controller_server',
            name='controller_server',
            output='screen',
            parameters=[nav2_yaml, {'use_sim_time': True}]
        ),
        Node(
            package='nav2_bt_navigator',
            executable='bt_navigator',
            name='bt_navigator',
            output='screen',
            parameters=[nav2_yaml, {'use_sim_time': True}]
        ),
        Node(
            package='nav2_recoveries',
            executable='recoveries_server',
            name='recoveries_server',
            output='screen',
            parameters=[nav2_yaml, {'use_sim_time': True}]
        ),
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_pathplanner',
            output='screen',
            parameters=[{'use_sim_time': True},
                        {'autostart': True},
                        {'node_names':['planner_server', 'controller_server', 'recoveries_server', 'bt_navigator', 'waypoint_follower']}]
        )
    ])