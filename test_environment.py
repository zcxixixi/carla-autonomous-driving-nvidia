# -*- coding: utf-8 -*-
import sys
import os

print("=== Testing Deep Learning Environment ===")

# Test basic imports
try:
    import numpy as np
    print("NumPy import successful")
except ImportError as e:
    print(f"NumPy import failed: {e}")
    sys.exit(1)

try:
    import cv2
    print("OpenCV import successful")
except ImportError as e:
    print(f"OpenCV import failed: {e}")

try:
    import tensorflow as tf
    print(f"TensorFlow import successful - Version: {tf.__version__}")
except ImportError as e:
    print(f"TensorFlow import failed: {e}")
    sys.exit(1)

try:
    import carla
    print("CARLA import successful")
except ImportError as e:
    print(f"CARLA import failed: {e}")
    sys.exit(1)

# Test TensorFlow functionality
print("\n=== Testing TensorFlow Basic Functions ===")
try:
    from tensorflow import keras
    from tensorflow.keras import layers
    
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(784,)),
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print("TensorFlow model creation successful")
    
    # Test with dummy data
    import numpy as np
    X_test = np.random.random((10, 784))
    y_test = np.random.randint(0, 10, 10)
    
    predictions = model.predict(X_test, verbose=0)
    print(f"Model prediction successful - Output shape: {predictions.shape}")
    
except Exception as e:
    print(f"TensorFlow functionality test failed: {e}")
    sys.exit(1)

# Test CARLA connection
print("\n=== Testing CARLA Connection ===")
try:
    client = carla.Client('localhost', 2000)
    client.set_timeout(5.0)
    world = client.get_world()
    print("CARLA server connection successful")
except Exception as e:
    print(f"CARLA connection failed: {e}")
    print("Please make sure CARLA server is running")

print("\n=== Environment Test Complete ===")
print("All components working, ready for deep learning training!")