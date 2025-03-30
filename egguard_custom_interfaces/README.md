# Egguard Custom Interfaces

This package contains custom message definitions used across the Egguard project for communication between different components.

## Messages

### ManualNav.msg

Used for manual navigation control of the robot.

- `velocity` (int32): Speed value ranging from 1 to 100, mapped to cmd_vel
- `direction` (string): Movement direction ("forward", "right", or "left")
- `stop_now` (bool): Emergency stop flag

### Mode.msg

Defines the operational mode of the robot.

- `mode` (string): Current mode of operation
  - "manual": Manual control mode
  - "autonomous": Autonomous navigation mode
  - "emergency": Emergency mode

## Design Considerations

1. **Message Simplicity**: Messages are kept simple and focused to ensure clear communication between components.
2. **Mode Management**: The Mode message enables centralized control of the robot's operational state.
3. **Safety First**: The ManualNav message includes an emergency stop flag for safety considerations.
4. **Extensibility**: The message structure allows for future expansion of functionality while maintaining backward compatibility.

## Usage

These interfaces are used by:

- Mode Manager package for state management
- Navigation system for movement control
- World simulation for environment interaction

## Dependencies

- ROS2 Humble
- Standard ROS2 message types
