# Egguard World Simulation

This package provides the simulation environment for the Egguard robot using Gazebo and ROS2. It includes the world model, robot URDF, and necessary launch configurations.

## Purpose

The world simulation package provides:

- A custom Gazebo world for the Egguard robot
- Robot model and URDF definitions
- Launch configurations for simulation
- Integration with the navigation system

## Components

### World Models

Located in `world/` directory:

- `burger.model`: Main world model for the simulation
- `burger_pi.model`: Raspberry Pi-specific world model

### Robot Models

Located in `models/` directory:

- Contains custom models and meshes for the simulation environment

### URDF Files

Located in `urdf/` directory:

- Contains the Unified Robot Description Format files for the robot
- Supports different TurtleBot3 models (burger, waffle, waffle_pi)

## Launch Files

### Main Launch File

- `turtlebot3_egguard_my_world.launch.py`
  - Launches Gazebo server with the custom world
  - Launches Gazebo client for visualization
  - Sets up robot state publisher
  - Configures simulation time

## Environment Setup

The package sets up:

1. Gazebo environment with custom world
2. Robot model in the simulation
3. ROS2 integration for control and navigation
4. Simulation time synchronization

## Dependencies

- ROS2 Humble
- Gazebo
- turtlebot3_gazebo
- gazebo_ros
- ament_index_python

## Usage

To launch the simulation:

```bash
ros2 launch my_world_egguard turtlebot3_egguard_my_world.launch.py
```

## Design Considerations

1. **Modularity**

   - Separate world and robot models
   - Configurable launch parameters
   - Easy to extend with new models

2. **Integration**

   - Seamless integration with navigation system
   - Compatible with ROS2 control interfaces
   - Support for different robot models

3. **Performance**

   - Optimized world models
   - Efficient resource usage
   - Real-time simulation capability

4. **Extensibility**
   - Easy to add new models
   - Configurable environment parameters
   - Support for different simulation scenarios

## Directory Structure

```
my_world_egguard/
├── launch/              # Launch files
├── models/             # Custom Gazebo models
├── src/                # Source files
├── urdf/               # Robot URDF files
└── world/              # World model files
```

## Configuration

The simulation can be configured through:

- Environment variables (e.g., TURTLEBOT3_MODEL)
- Launch file parameters
- World model modifications
