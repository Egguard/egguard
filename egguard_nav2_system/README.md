# Egguard Navigation System

This package implements the navigation system for the Egguard robot, providing both manual and autonomous navigation capabilities using ROS2 Navigation2 (Nav2) stack.

## Purpose

The navigation system provides:

- Manual control through velocity commands
- Autonomous navigation using waypoints
- Mode-based control switching
- Integration with Nav2 stack for path planning and execution

## Topics

### Subscribed Topics

- `/mode` (egguard_custom_interfaces/msg/Mode)

  - Purpose: Receives the current operational mode
  - QoS: Transient Local, Keep Last (depth: 1)

- `/manual_nav` (egguard_custom_interfaces/msg/ManualNav)
  - Purpose: Receives manual navigation commands
  - QoS: Best Effort, Keep Last (depth: 1)
  - Only active in manual mode

### Published Topics

- `/cmd_vel` (geometry_msgs/msg/Twist)
  - Purpose: Publishes velocity commands to the robot
  - QoS: Best Effort, Keep Last (depth: 10)

### Action Clients

- `follow_waypoints` (nav2_msgs/action/FollowWaypoints)
  - Purpose: Sends waypoint sequences for autonomous navigation
  - Used by: AutonomousController

## Components

### Manual Controller

- Node name: `manual_controller`
- Function: Handles manual navigation commands
- Features:
  - Dynamic subscription management based on mode
  - Velocity mapping (0-100% to actual velocities)
  - Direction control (forward, left, right)
  - Emergency stop capability

### Autonomous Controller

- Node name: `autonomous_controller`
- Function: Manages autonomous navigation
- Features:
  - Waypoint-based navigation
  - Progress monitoring and feedback
  - Automatic mode switching
  - Predefined waypoint sequence

## Design Considerations

1. **Mode-Based Architecture**

   - Clear separation between manual and autonomous control
   - Dynamic subscription management for resource efficiency
   - Safe mode transitions

2. **Safety Features**

   - Emergency stop capability
   - Velocity limiting
   - Mode-based command validation

3. **Robustness**

   - Comprehensive error handling
   - Feedback monitoring
   - Graceful mode transitions

4. **Performance**
   - Optimized QoS profiles for different message types
   - Efficient resource management
   - Real-time command processing

## Dependencies

- ROS2 Humble
- Nav2 Stack
- egguard_custom_interfaces
- egguard_mode_manager
- geometry_msgs
- nav2_msgs

## Usage

The navigation system can be operated in two modes:

1. **Manual Mode**: Direct control through velocity commands
2. **Autonomous Mode**: Predefined waypoint navigation

Mode switching is handled through the `/mode` topic, and the system automatically adjusts its behavior accordingly.

## Launch Files

Located in the `launch` directory, providing different launch configurations for various navigation scenarios.
