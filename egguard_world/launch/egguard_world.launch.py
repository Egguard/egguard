import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, LogInfo
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node

TURTLEBOT3_MODEL = os.environ.get('TURTLEBOT3_MODEL', 'burger')  # Default: 'burger'

def generate_launch_description():
    # Definir rutas de archivos
    world_file_name = 'world/' + TURTLEBOT3_MODEL + '.model'
    urdf_file_name = 'urdf/turtlebot3_' + TURTLEBOT3_MODEL + '.urdf'
    pkg_share = FindPackageShare(package='egguard_world').find('egguard_world')
    gazebo_models_path = os.path.join(pkg_share, 'models', 'model_editor_models')

    # Configurar la variable de entorno GAZEBO_MODEL_PATH
    current_gazebo_model_path = os.environ.get("GAZEBO_MODEL_PATH", "")
    os.environ["GAZEBO_MODEL_PATH"] = f"{gazebo_models_path}:{current_gazebo_model_path}" if current_gazebo_model_path else gazebo_models_path

    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    world = os.path.join(pkg_share, world_file_name)
    urdf = os.path.join(pkg_share, urdf_file_name)

    launch_file_dir = os.path.join(get_package_share_directory('turtlebot3_gazebo'), 'launch')
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')

    # Imprimir la variable de entorno para depuración
    print("GAZEBO_MODEL_PATH:", os.environ["GAZEBO_MODEL_PATH"])


    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg_gazebo_ros, 'launch', 'gzserver.launch.py')
            ),
            launch_arguments={'world': world}.items(),
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg_gazebo_ros, 'launch', 'gzclient.launch.py')
            ),
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([launch_file_dir, '/robot_state_publisher.launch.py']),
            launch_arguments={'use_sim_time': use_sim_time}.items(),
        ),
    ])
