# Posture Detection App Assets

Place the following files in this directory before building the application:

1. `posture_1dcnn_int8.tflite` (Primary INT8 deployment model)
2. `posture_1dcnn_float32.tflite` (Fallback/debug FLOAT32 model)
3. `normalization.json` (Sensor normalization coefficients: mean and std)

The app looks for these files by name to run real-time posture detection.
