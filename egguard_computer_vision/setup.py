from setuptools import setup

package_name = 'egguard_computer_vision'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name, ['launch/egguard_vision.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ubuntu',
    maintainer_email='alexeltren@gmail.com',
    description='Egg detection system using computer vision',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'egguard_node=egguard_computer_vision.main:main',
            'egg_simulator = egguard_computer_vision.simulation:main'
        ],
    },
)