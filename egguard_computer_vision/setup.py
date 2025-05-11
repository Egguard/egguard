from setuptools import setup
import os
from glob import glob

package_name = 'egguard_computer_vision'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'test_images'), glob('test_images/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ubuntu',
    maintainer_email='alexeltren@gmail.com',
    description='Computer vision package for egg detection',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'egg_detection_node = egguard_computer_vision.main_node:main',
            'test_image_publisher = egguard_computer_vision.test_image_publisher:main',
        ],
    },
)