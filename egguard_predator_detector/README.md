# EggGuard Predator Detector

A ROS2 package for detecting predators (eagles, raccoons, foxes) in poultry farms using ML inference APIs.

## Overview

Monitors camera feeds, sends images to ML API for classification, and triggers alerts when predators are detected above confidence threshold.

**Architecture**: Camera → ROS2 Node → ML API → Alert System

## Features

- Real-time predator detection from camera feeds
- Configurable confidence thresholds and detection intervals
- Alert system with backend API integration
- Cooldown protection to prevent spam alerts
- Thread-safe image processing

## Supported Predators

- `aguila` (Eagle)
- `mapache` (Raccoon)
- `zorro` (Fox)

## Configuration

| Parameter              | Default                        | Description                        |
| ---------------------- | ------------------------------ | ---------------------------------- |
| `confidence_threshold` | 0.70                           | Minimum confidence for detection   |
| `check_interval`       | 5.0                            | Seconds between processing cycles  |
| `cooldown`             | 10.0                           | Seconds between consecutive alerts |
| `ml_api_url`           | `http://localhost:8501`        | ML inference API URL               |
| `backend_url`          | `http://localhost:8081/api/v1` | Backend API URL                    |
| `backend_active`       | true                           | Enable/disable backend alerts      |

## Topics

### Subscribed Topics

- `/image` (sensor_msgs/Image): Camera image feed for processing

### Published Topics

None (alerts are sent via HTTP API)

## Installation

1. **Prerequisites**:

   ```bash
   # Ensure ROS2 is installed and sourced
   source /opt/ros/humble/setup.bash
   ```

2. **Dependencies**:

   ```bash
   sudo apt install python3-opencv python3-requests
   pip3 install cv-bridge
   ```

3. **Build the package**:
   ```bash
   cd ~/ros2_ws
   colcon build --packages-select egguard_predator_detector
   source install/setup.bash
   ```

## Usage

### Basic Launch

```bash
ros2 launch egguard_predator_detector predator_detector.launch.py
```

### Launch with Custom Parameters

```bash
ros2 run egguard_predator_detector predator_detector --ros-args \
  -p confidence_threshold:=0.8 \
  -p check_interval:=3.0 \
  -p backend_active:=false
```

### Testing Mode

For testing purposes, you can uncomment the test image loading section in the code:

```python
# Uncomment these lines in predator_detector.py for testing
test_image_path = os.path.join(get_package_share_directory('egguard_predator_detector'), 'testing', 'zorro.jpg')
frame_to_process = cv2.imread(test_image_path)
```

## API Integration

### ML Inference API

The node expects the ML API to be available at the configured endpoint with the following interface:

**Endpoint**: `POST /predict`

**Request**: Multipart form data with `file` field containing the image

**Response**:

```json
{
  "prediction": "zorro",
  "probability": 0.85
}
```

### Backend API

Alerts are sent to the backend API when predators are detected:

**Endpoint**: `POST /robots/{robot_id}/notifications`

**Request**: Multipart form data with:

- `notification`: JSON payload with message and severity
- `image`: JPEG image where predator was detected

## Testing Resources

The package includes test images in the `testing/` directory:

- `aguila.jpg` - Eagle test image
- `zorro.jpg` - Fox test image

These images are automatically installed with the package and can be used for testing the detection pipeline.

## Error Handling

The node handles various error conditions gracefully:

- **ML API Timeout**: Logs warning and continues processing
- **Connection Errors**: Logs warnings for both ML API and backend
- **Image Processing Errors**: Logs errors and skips problematic frames
- **Backend Communication**: Continues operation even if backend is unavailable

## Logging

The node provides detailed logging at different levels:

- **DEBUG**: Frame reception, ML API results, alert details
- **INFO**: Successful detections and alert confirmations
- **WARN**: API timeouts and connection issues
- **ERROR**: Critical failures in image processing or encoding

## Development

### File Structure

```
egguard_predator_detector/
├── egguard_predator_detector/
│   ├── __init__.py
│   └── predator_detector.py        # Main detection node
├── launch/
│   └── predator_detector.launch.py # Launch configuration
├── testing/
│   ├── aguila.jpg                  # Eagle test image
│   └── zorro.jpg                   # Fox test image
├── package.xml                     # Package metadata
├── setup.py                        # Package setup
└── README.md                       # This file
```

### Extending the System

To add new predator types:

1. Update the `predator_labels` list in `predator_detector.py`
2. Add corresponding entries to `predator_messages` dictionary
3. Update the ML model to recognize the new predator class

## Troubleshooting

### Common Issues

1. **No images received**:

   - Check camera topic is publishing: `ros2 topic echo /image`
   - Verify topic name matches subscription

2. **ML API connection failed**:

   - Ensure ML inference service is running
   - Check the `ml_api_url` parameter
   - Verify network connectivity

3. **No alerts sent**:

   - Check `backend_active` parameter is true
   - Verify backend URL and connectivity
   - Check detection confidence meets threshold

4. **High CPU usage**:
   - Increase `check_interval` to reduce processing frequency
   - Check image resolution and processing load

### Debug Commands

```bash
# Check node status
ros2 node info /predator_detector_node

# Monitor parameters
ros2 param list /predator_detector_node

# View logs
ros2 run rqt_console rqt_console
```

## Dependencies

- **ROS2**: Humble or later
- **OpenCV**: 4.x for image processing
- **cv_bridge**: ROS-OpenCV integration
- **requests**: HTTP API communication
- **threading**: Concurrent processing support

## License

This project is part of the EggGuard system. License details to be determined.

## Contributors

- Juan Diaz

## Related Packages

- `egguard_computer_vision`: Egg detection and analysis
- `egguard_mode_manager`: System mode coordination
- `predator_inference`: ML inference API service
