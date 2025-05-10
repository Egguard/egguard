from setuptools import setup
import os
from glob import glob

package_name = 'egguard_computer_vision'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Incluir archivos de lanzamiento
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        # Crear directorios para imágenes
        (os.path.join('share', package_name, 'test_images'), []),
        (os.path.join('share', package_name, 'output'), []),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@example.com',
    description='Detector de huevos basado en visión por computador',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'egg_detection_node = egguard_computer_vision.main_node:main',
            'test_image_publisher = egguard_computer_vision.test_image_publisher:main',
        ],
    },
)