from setuptools import setup
import os # incluir esta línea
from glob import glob # incluir esta línea

package_name = 'egguard_mode_manager'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*.launch.py')) # incluir
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='juan',
    maintainer_email='juanben10xlr99@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'initial_mode_publisher = egguard_mode_manager.initial_mode_publisher:main' #incluir
        ],
    },
)
