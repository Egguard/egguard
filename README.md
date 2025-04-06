# Egguard - Autonomous Chicken Farm Assistant

Egguard is an innovative ROS2-based robotic system designed to automate and enhance the safety of chicken farms. This intelligent assistant helps farmers by automating various tasks such as egg collection, predator detection, and farm production monitoring.

## Features

- **Autonomous Navigation**: Safely navigate through the chicken farm environment
- **Dual Control Modes**:
  - Manual control for direct human intervention
  - Autonomous mode for automated tasks
- **Safety Monitoring**: Detect potential threats and monitor farm conditions
- **Production Analytics**: Track and analyze farm production statistics
- **Emergency Response**: Quick response to critical situations

## Prerequisites

- ROS2 Galactic
- Gazebo
- Nav2 Stack
- Python 3.8+

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/egguard.git
cd egguard
```

2. Install dependencies:

```bash
rosdep install --from-paths src --ignore-src -r -y
```

## Building the Packages

The packages must be built in the following order:

```bash
# Build custom interfaces first
colcon build --packages-select egguard_custom_interfaces
source ./install/setup.bash

# Build mode manager
colcon build --packages-select egguard_mode_manager
source ./install/setup.bash

# Build navigation system
colcon build --packages-select egguard_nav2_system
source ./install/setup.bash

# Build world package
colcon build --packages-select my_world_egguard
source ./install/setup.bash
```

**Important**: After each build and when opening new terminals, always source the setup file:

```bash
source ./install/setup.bash
```

## Running the System

Follow these steps in order to launch the complete system:

1. **Launch the Simulation World**

```bash
# Set the Gazebo model path (adjust the path according to your setup)
export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:/path/to/your/egguard/my_world_egguard/models/model_editor_models

# Launch the world
ros2 launch my_world_egguard turtlebot3_egguard_my_world.launch.py
```

2. **Load the Map and Initialize Navigation**

```bash
# Launch the map provider with AMCL
ros2 launch egguard_nav2_system provide_map.launch.py use_sim_time:=True

# Load the map (adjust the path according to your setup)
ros2 service call /map_server/load_map nav2_msgs/srv/LoadMap "{map_url: /path/to/your/egguard/egguard_nav2_system/config/my_map.yaml}"

# Set initial robot pose
ros2 run egguard_nav2_system initial_pose_pub
```

3. **Launch Navigation and Control Systems**

```bash
# Start the autonomous navigation controller
ros2 launch egguard_nav2_system autonomous_controller.launch.py

# Set initial mode (autonomous)
ros2 launch egguard_mode_manager initial_mode_publisher.launch.py
```

4. **Control the Robot**

- For autonomous mode: The robot will follow predefined waypoints
  ```bash
  ros2 run egguard_nav2_system autonomous_controller
  ```
- For manual mode: Use the manual controller to directly control the robot
  ```bash
  ros2 run egguard_nav2_system manual_controller
  ```

## Project Structure

- `egguard_custom_interfaces/`: Custom message definitions for the project
- `egguard_mode_manager/`: Manages robot operational modes
- `egguard_nav2_system/`: Navigation and control system
- `my_world_egguard/`: Simulation environment and world models

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- ROS2 Community
- Nav2 Stack
- TurtleBot3 Team
