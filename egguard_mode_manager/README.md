# Egguard Mode Manager

This package manages the operational mode of the Egguard robot, handling transitions between different modes and ensuring proper mode initialization.

## Purpose

The Mode Manager package is responsible for:

- Publishing the initial mode state of the robot
- Managing mode transitions between manual, autonomous, and emergency modes
- Ensuring reliable mode communication across the system

## Topics

### Published Topics

- `/mode` (egguard_custom_interfaces/msg/Mode)
  - QoS Profile: Transient Local, Keep Last (depth: 1)
  - Purpose: Broadcasts the current operational mode
  - Message Types: "manual", "autonomous", "emergency"

## Components

### Initial Mode Publisher

- Node name: `initial_mode_publisher`
- Function: Publishes the initial mode state when the system starts
- Default mode: "manual"
- Error handling: Includes comprehensive error handling for initialization and publishing

### QoS Configuration

- Implements Transient Local durability policy
- Maintains last message only (depth: 1)
- Ensures new subscribers receive the latest mode state

## Design Considerations

1. **Reliability**: Uses Transient Local QoS to ensure mode state is never lost
2. **Simplicity**: Single responsibility principle - focused on mode management
3. **Error Handling**: Comprehensive error handling for robustness
4. **Initialization**: Ensures system starts in a known state

## Dependencies

- ROS2 Humble
- egguard_custom_interfaces
- rclpy

## Usage

The mode manager is typically launched at system startup to establish the initial mode state. Other components can subscribe to the `/mode` topic to react to mode changes.

## Launch Files

Located in the `launch` directory, providing different launch configurations for various scenarios.
