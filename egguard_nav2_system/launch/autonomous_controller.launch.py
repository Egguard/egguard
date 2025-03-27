import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import TimerAction
"""
Node(
    package='egguard_nav2_system',
    executable='autonomous_controller',
    name='autonomous_controller',
    output='screen',
),
"""

def generate_launch_description():

    nav2_yaml = os.path.join(get_package_share_directory('my_nav2_system'), 'config', 'my_nav2_params.yaml')
    map_file = os.path.join(get_package_share_directory('my_nav2_system'), 'config', 'my_map.yaml')
    rviz_config_dir = os.path.join(get_package_share_directory('my_nav2_system'), 'config', 'rviz_config.rviz')
   # urdf = os.path.join(get_package_share_directory('turtlebot3_description'), 'urdf', 'turtlebot3_burger.urdf')
   # world = os.path.join(get_package_share_directory('turtlebot3_gazebo'), 'worlds', 'turtlebot3_worlds/burger.model')

    waypoint_follower_node = TimerAction(
        period=5.0,  # Espera 5 segundos antes de lanzar el nodo
        actions=[
            Node(
                package='nav2_waypoint_follower',
                executable='waypoint_follower',
                name='waypoint_follower',
                output='screen',
                parameters=[nav2_yaml, {'use_sim_time': True}]
            )
        ]
    )

    return LaunchDescription([
        Node(
            package = 'nav2_map_server',
            executable = 'map_server',
            name = 'map_server',
            output = 'screen',
            parameters=[{'use_sim_time': True}, {'yaml_filename':map_file}]
        ),
        waypoint_follower_node,
        Node(
            package='nav2_amcl',
            executable='amcl',
            name='amcl',
            output='screen',
            parameters=[nav2_yaml]
        ),

        Node(
            package = 'nav2_planner',
            executable = 'planner_server',
            name = 'planner_server',
            output = 'screen',
            parameters=[nav2_yaml]
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
                        {'node_names':['map_server', 'amcl', 'planner_server', 'controller_server', 'recoveries_server', 'bt_navigator', 'waypoint_follower']}]
        ),

         Node(
               package='rviz2',
               executable='rviz2',
               name='rviz2',
               arguments=['-d', rviz_config_dir],
               parameters=[{'use_sim_time': True}],
               output='screen'
        )
    ])