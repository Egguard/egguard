# Predator Inference API

A Flask-based machine learning inference service for predator detection in poultry farm environments. This API provides real-time image classification to identify potential threats including eagles, raccoons, and foxes.

## Overview

The Predator Inference API is a standalone machine learning service that processes images and returns predator classification results. It uses a TensorFlow/Keras model trained specifically for farm predator detection and provides HTTP endpoints for integration with ROS2 systems and other applications.

## Features

- **Real-time Inference**: Fast image classification using optimized TensorFlow model
- **Multi-class Detection**: Identifies eagles, raccoons, foxes, and non-predators
- **REST API**: Simple HTTP interface for easy integration
- **Docker Support**: Containerized deployment for consistent environments
- **Image Preprocessing**: Automatic image resizing and normalization
- **Error Handling**: Robust error handling for various input formats

## Supported Classes

The model can classify images into the following categories:

| Class ID | Label           | Spanish       | Description         |
| -------- | --------------- | ------------- | ------------------- |
| 0        | `aguila`        | Águila        | Eagle detection     |
| 1        | `mapache`       | Mapache       | Raccoon detection   |
| 2        | `no_depredador` | No Depredador | No predator present |
| 3        | `zorro`         | Zorro         | Fox detection       |

## API Endpoints

### POST /predict

Predator detection endpoint that processes uploaded images.

**Request**:

- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: File field named `file` containing image data

**Response**:

```json
{
  "prediction": "zorro",
  "probability": 0.8542
}
```

**Example Usage**:

```bash
curl -X POST -F "file=@test_image.jpg" http://localhost:8501/predict
```

### GET /view_image

Debug endpoint for viewing the last processed image.

**Request**:

- Method: `GET`
- No parameters required

**Response**:

- Content-Type: `image/jpeg`
- Body: JPEG image data

## Installation

### Option 1: Docker Deployment (Recommended)

1. **Build the Docker image**:

   ```bash
   cd predator_inference
   docker build -t predator-inference .
   ```

2. **Run the container**:

   ```bash
   docker run -p 8501:8501 predator-inference
   ```

3. **Test the API**:
   ```bash
   curl -X POST -F "file=@sillas.jpg" http://localhost:8501/predict
   ```

### Option 2: Local Installation

1. **Install Python dependencies**:

   ```bash
   pip install flask tensorflow opencv-python numpy requests
   ```

2. **Ensure model file is present**:

   ```bash
   # The predator_model.keras file should be in the same directory
   ls predator_model.keras
   ```

3. **Run the Flask application**:
   ```bash
   python app.py
   ```

## Usage Examples

### Python Client Example

```python
import requests

# Send image for prediction
with open("test_image.jpg", "rb") as f:
    files = {"file": f}
    response = requests.post("http://localhost:8501/predict", files=files)

if response.status_code == 200:
    result = response.json()
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['probability']:.2%}")
else:
    print(f"Error: {response.status_code}")
```

### Using the Included Test Client

```bash
python client_predict.py
```

This will test the API using the included `sillas.jpg` test image.

### cURL Examples

```bash
# Basic prediction
curl -X POST -F "file=@image.jpg" http://localhost:8501/predict

# With verbose output
curl -v -X POST -F "file=@image.jpg" http://localhost:8501/predict

# View last processed image
curl http://localhost:8501/view_image -o last_image.jpg
```

## Model Details

### Architecture

- **Base Model**: MobileNetV3-based architecture
- **Input Size**: 224x224 pixels
- **Color Space**: RGB
- **Preprocessing**: MobileNetV3 preprocessing pipeline

### Performance Characteristics

- **Inference Time**: ~50-200ms per image (CPU)
- **Memory Usage**: ~500MB RAM
- **Supported Formats**: JPEG, PNG, BMP, TIFF

### Model File

- **Filename**: `predator_model.keras`
- **Format**: Keras SavedModel format
- **Size**: Variable (depends on training)

## Configuration

### Environment Variables

| Variable     | Default                | Description         |
| ------------ | ---------------------- | ------------------- |
| `FLASK_HOST` | `0.0.0.0`              | Server host address |
| `FLASK_PORT` | `8501`                 | Server port number  |
| `MODEL_PATH` | `predator_model.keras` | Path to model file  |

### Flask Configuration

The API runs with the following default settings:

- **Host**: `0.0.0.0` (all interfaces)
- **Port**: `8501`
- **Debug**: `False` (production mode)

## Docker Configuration

### Dockerfile Details

The Docker image is based on:

- **Base Image**: `tensorflow/tensorflow:2.18.0`
- **Python Version**: 3.x (included in TensorFlow image)
- **Virtual Environment**: `/venv` for dependency isolation
- **Working Directory**: `/app`

### Exposed Ports

- **8501**: HTTP API port

### Volume Mounts (Optional)

```bash
# Mount custom model
docker run -v /path/to/model:/app/predator_model.keras -p 8501:8501 predator-inference

# Mount test images
docker run -v /path/to/images:/app/test_images -p 8501:8501 predator-inference
```

## File Structure

```
predator_inference/
├── app.py                    # Main Flask application
├── client_predict.py         # Test client script
├── predator_model.keras      # Trained TensorFlow model
├── Dockerfile               # Docker container configuration
├── sillas.jpg              # Test image (chairs - no predator)
├── running.txt             # Runtime notes/logs
├── testing images/         # Additional test images
└── README.md              # This file
```

## Error Handling

The API handles various error conditions:

### HTTP Status Codes

| Code | Condition    | Response                     |
| ---- | ------------ | ---------------------------- |
| 200  | Success      | Prediction result            |
| 400  | Bad Request  | Missing file or decode error |
| 500  | Server Error | Model inference failure      |

### Common Errors

1. **No file provided**:

   ```json
   { "error": "No file part" }
   ```

2. **Empty filename**:

   ```json
   { "error": "No selected file" }
   ```

3. **Image decode failure**:

   ```json
   { "error": "Failed to decode image" }
   ```

4. **Model inference error**:
   ```json
   { "error": "Model prediction failed: [details]" }
   ```

## Performance Optimization

### Recommendations

1. **Hardware**: Use GPU-enabled TensorFlow for faster inference
2. **Batch Processing**: Process multiple images in batches when possible
3. **Image Size**: Resize large images before sending to reduce network overhead
4. **Caching**: Cache model in memory for faster subsequent predictions

### Monitoring

Monitor the following metrics:

- Response times per request
- Memory usage patterns
- Error rates by error type
- Prediction confidence distributions

## Testing

### Test Images

The package includes test images:

- `sillas.jpg` - Test image with no predators (chairs)
- `testing images/` - Directory with additional test cases

### Validation

Test the API with various scenarios:

```bash
# Test with valid image
python client_predict.py

# Test with invalid file
curl -X POST http://localhost:8501/predict

# Test with corrupted image
curl -X POST -F "file=@corrupted.jpg" http://localhost:8501/predict
```

## Integration with ROS2

The API is designed to work with the `egguard_predator_detector` ROS2 package:

1. **Start the inference API**:

   ```bash
   python app.py
   ```

2. **Configure ROS2 node**:

   ```bash
   ros2 run egguard_predator_detector predator_detector --ros-args \
     -p ml_api_url:=http://localhost:8501
   ```

3. **Monitor predictions**:
   ```bash
   ros2 topic echo /egguard/predator_alerts
   ```

## Troubleshooting

### Common Issues

1. **Model not found**:

   - Ensure `predator_model.keras` is in the application directory
   - Check file permissions and accessibility

2. **Port already in use**:

   ```bash
   # Find process using port 8501
   sudo lsof -i :8501
   # Kill the process or use a different port
   ```

3. **Memory errors**:

   - Increase Docker memory limits
   - Use smaller batch sizes for inference

4. **Slow inference**:
   - Consider GPU acceleration
   - Optimize image preprocessing
   - Use model quantization techniques

### Debug Mode

Run in debug mode for detailed error information:

```python
# In app.py, change the last line to:
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8501, debug=True)
```

## Security Considerations

- **Input Validation**: Always validate uploaded file formats
- **File Size Limits**: Implement reasonable file size restrictions
- **Rate Limiting**: Consider implementing rate limiting for production use
- **HTTPS**: Use HTTPS in production environments
- **Authentication**: Add API authentication for production deployments

## Dependencies

### Python Packages

- `flask` - Web framework
- `tensorflow>=2.18.0` - Machine learning framework
- `opencv-python` - Image processing
- `numpy` - Numerical computations
- `requests` - HTTP client (for testing)

### System Dependencies

- `libgl1-mesa-glx` - OpenGL support
- `libglib2.0-0` - GLib library
- `libsm6`, `libxext6`, `libxrender1` - X11 support

## License

This project is part of the EggGuard system. License details to be determined.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## Related Projects

- **egguard_predator_detector**: ROS2 client package
- **egguard_computer_vision**: Egg detection system
- **egguard_mode_manager**: System coordination

## Changelog

### Version History

- **v1.0.0**: Initial release with basic predator detection
- **Latest**: Current development version

For detailed changes, see git commit history.
