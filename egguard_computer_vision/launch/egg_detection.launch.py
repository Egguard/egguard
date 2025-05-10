#!/usr/bin/env python3
# egg_detection_launch.py - Archivo de lanzamiento para el detector de huevos
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Declarar argumentos de lanzamiento
    detection_interval = LaunchConfiguration('detection_interval')
    debug_mode = LaunchConfiguration('debug_mode')
    save_images = LaunchConfiguration('save_images')
    output_dir = LaunchConfiguration('output_dir')
    camera_topic = LaunchConfiguration('camera_topic')
    use_camera = LaunchConfiguration('use_camera')
    use_test_publisher = LaunchConfiguration('use_test_publisher')
    
    # Directorio de recursos del paquete
    pkg_dir = get_package_share_directory('egguard_computer_vision')
    default_output_dir = os.path.join(pkg_dir, 'output')
    
    # Definir argumentos con valores por defecto
    declare_detection_interval = DeclareLaunchArgument(
        'detection_interval',
        default_value='3.0',
        description='Intervalo entre detecciones en segundos'
    )
    
    declare_debug_mode = DeclareLaunchArgument(
        'debug_mode',
        default_value='False',
        description='Activar modo debug'
    )
    
    declare_save_images = DeclareLaunchArgument(
        'save_images',
        default_value='True',
        description='Guardar imágenes procesadas'
    )
    
    declare_output_dir = DeclareLaunchArgument(
        'output_dir',
        default_value=default_output_dir,
        description='Directorio para guardar imágenes procesadas'
    )
    
    declare_camera_topic = DeclareLaunchArgument(
        'camera_topic',
        default_value='image_raw',
        description='Topic de la cámara'
    )
    
    declare_use_camera = DeclareLaunchArgument(
        'use_camera',
        default_value='True',
        description='Usar cámara real (si está disponible)'
    )
    
    declare_use_test_publisher = DeclareLaunchArgument(
        'use_test_publisher',
        default_value='False',
        description='Usar publicador de prueba para simular cámara'
    )
    
    # Nodo de detección de huevos
    egg_detection_node = Node(
        package='egguard_computer_vision',
        executable='egg_detection_node',
        name='egg_detection_node',
        parameters=[{
            'detection_interval': detection_interval,
            'debug_mode': debug_mode,
            'save_images': save_images,
            'output_dir': output_dir
        }],
        remappings=[
            ('image_raw', camera_topic),
        ],
        output='screen'
    )
    
    # Nodo publicador de imágenes de prueba (opcional)
    test_publisher_node = Node(
        package='egguard_computer_vision',
        executable='test_image_publisher',
        name='test_image_publisher',
        parameters=[{
            'publish_rate': 1.0,  # Hz
            'image_dir': os.path.join(pkg_dir, 'test_images'),
        }],
        output='screen',
        condition=IfCondition(use_test_publisher)
    )
    
    # Crear y devolver la descripción de lanzamiento
    return LaunchDescription([
        # Declaración de argumentos
        declare_detection_interval,
        declare_debug_mode,
        declare_save_images,
        declare_output_dir,
        declare_camera_topic,
        declare_use_camera,
        declare_use_test_publisher,
        
        # Nodos
        egg_detection_node,
        test_publisher_node
    ])