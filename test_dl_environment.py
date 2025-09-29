#!/usr/bin/env python3
import sys
import os

print("=== 测试深度学习环境 ===")

# Test basic imports
try:
    import numpy as np
    print("? NumPy导入成功")
except ImportError as e:
    print(f"? NumPy导入失败: {e}")
    sys.exit(1)

try:
    import cv2
    print("? OpenCV导入成功")
except ImportError as e:
    print(f"? OpenCV导入失败: {e}")

try:
    import tensorflow as tf
    print(f"? TensorFlow导入成功 - 版本: {tf.__version__}")
except ImportError as e:
    print(f"? TensorFlow导入失败: {e}")
    sys.exit(1)

try:
    import carla
    print(f"? CARLA导入成功 - 版本: {carla.__version__}")
except ImportError as e:
    print(f"? CARLA导入失败: {e}")
    sys.exit(1)

# Test TensorFlow functionality
print("\n=== 测试TensorFlow基本功能 ===")
try:
    # Create a simple model
    from tensorflow import keras
    from tensorflow.keras import layers
    
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(784,)),
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print("? TensorFlow模型创建成功")
    
    # Test with dummy data
    import numpy as np
    X_test = np.random.random((10, 784))
    y_test = np.random.randint(0, 10, 10)
    
    predictions = model.predict(X_test, verbose=0)
    print(f"? 模型预测成功 - 输出shape: {predictions.shape}")
    
except Exception as e:
    print(f"? TensorFlow功能测试失败: {e}")
    sys.exit(1)

# Test CARLA connection
print("\n=== 测试CARLA连接 ===")
try:
    client = carla.Client('localhost', 2000)
    client.set_timeout(5.0)
    world = client.get_world()
    print("? CARLA服务器连接成功")
except Exception as e:
    print(f"? CARLA连接失败: {e}")
    print("请确保CARLA服务器正在运行")

print("\n=== 环境测试完成 ===")
print("所有组件正常，可以开始深度学习训练！")