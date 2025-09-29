#!/usr/bin/env python3
"""
CARLA自动驾驶系统测试脚本 / CARLA Autonomous Driving System Test Script
张涔熙的科研项目 - 系统功能测试

This script provides comprehensive testing for the autonomous driving system.
"""

import logging
import sys
import unittest
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.perception.camera_perception import CameraPerception
from src.perception.lidar_perception import LidarPerception
from src.planning.path_planner import PathPlanner
from src.control.vehicle_controller import VehicleController
from src.data_collection.data_logger import DataLogger
from src.simulation.carla_simulator import CarlaSimulator

import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TestAutonomousDrivingSystem(unittest.TestCase):
    """自动驾驶系统测试类 / Autonomous Driving System Test Class"""
    
    def setUp(self):
        """测试初始化 / Test initialization"""
        logger.info("Setting up test environment...")
        
    def test_camera_perception(self):
        """测试相机感知模块 / Test camera perception module"""
        logger.info("Testing camera perception...")
        
        # Create dummy camera data
        dummy_image = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
        camera_data = {
            'data': dummy_image,
            'timestamp': 1234567890
        }
        
        # Initialize and test perception
        perception = CameraPerception()
        result = perception.process(camera_data)
        
        # Check results
        self.assertIsInstance(result, dict)
        self.assertIn('objects', result)
        self.assertIn('lanes', result)
        self.assertIn('traffic_lights', result)
        
        logger.info("✓ Camera perception test passed")
    
    def test_lidar_perception(self):
        """测试LiDAR感知模块 / Test LiDAR perception module"""
        logger.info("Testing LiDAR perception...")
        
        # Create dummy LiDAR data
        num_points = 1000
        dummy_points = np.random.randn(num_points, 4)
        lidar_data = {
            'data': dummy_points,
            'timestamp': 1234567890
        }
        
        # Initialize and test perception
        perception = LidarPerception()
        result = perception.process(lidar_data)
        
        # Check results
        self.assertIsInstance(result, dict)
        self.assertIn('objects', result)
        self.assertIn('obstacles', result)
        self.assertIn('depth_map', result)
        
        logger.info("✓ LiDAR perception test passed")
    
    def test_path_planner(self):
        """测试路径规划模块 / Test path planner module"""
        logger.info("Testing path planner...")
        
        # Create dummy data
        current_pose = {
            'location': [0.0, 0.0, 0.0],
            'rotation': [0.0, 0.0, 0.0]
        }
        
        perception_data = {
            'objects': [
                {'center': [10.0, 5.0, 0.0], 'type': 'car', 'size': [4.0, 2.0, 1.5]}
            ],
            'obstacles': []
        }
        
        goal = [50.0, 0.0]
        
        # Initialize and test planner
        planner = PathPlanner()
        result = planner.plan(current_pose, perception_data, goal)
        
        # Check results
        self.assertIsInstance(result, dict)
        self.assertIn('waypoints', result)
        self.assertIn('behavior', result)
        self.assertIn('valid', result)
        
        logger.info("✓ Path planner test passed")
    
    def test_vehicle_controller(self):
        """测试车辆控制模块 / Test vehicle controller module"""
        logger.info("Testing vehicle controller...")
        
        # Create dummy state and path
        current_state = {
            'speed': 10.0,
            'location': [0.0, 0.0, 0.0],
            'rotation': [0.0, 45.0, 0.0],
            'velocity': [7.07, 7.07, 0.0]
        }
        
        planned_path = {
            'waypoints': [
                {'position': [10.0, 10.0], 'speed': 15.0, 'time': 1.0}
            ],
            'behavior': 'cruise',
            'valid': True
        }
        
        # Initialize and test controller
        controller = VehicleController()
        result = controller.compute_control(current_state, planned_path)
        
        # Check results
        self.assertIsInstance(result, dict)
        self.assertIn('throttle', result)
        self.assertIn('brake', result)
        self.assertIn('steer', result)
        
        # Check control limits
        self.assertGreaterEqual(result['throttle'], 0.0)
        self.assertLessEqual(result['throttle'], 1.0)
        self.assertGreaterEqual(result['brake'], 0.0)
        self.assertLessEqual(result['brake'], 1.0)
        
        logger.info("✓ Vehicle controller test passed")
    
    def test_data_logger(self):
        """测试数据记录模块 / Test data logger module"""
        logger.info("Testing data logger...")
        
        # Initialize data logger
        logger_instance = DataLogger(log_dir="tests/test_logs")
        
        # Create dummy step data
        step_data = {
            'step': 1,
            'timestamp': 1234567890,
            'perception': {
                'objects': [{'class': 'car', 'confidence': 0.9}]
            },
            'control': {
                'throttle': 0.5,
                'brake': 0.0,
                'steer': 0.1
            }
        }
        
        # Test logging
        logger_instance.log_step(step_data)
        logger_instance.save_session()
        
        # Check session list
        sessions = logger_instance.get_session_list()
        self.assertIsInstance(sessions, list)
        
        logger.info("✓ Data logger test passed")
    
    def test_carla_simulator_initialization(self):
        """测试CARLA仿真器初始化 / Test CARLA simulator initialization"""
        logger.info("Testing CARLA simulator initialization...")
        
        # Note: This test only checks initialization, not actual connection
        # since CARLA server might not be running in test environment
        
        simulator = CarlaSimulator()
        
        # Check basic attributes
        self.assertEqual(simulator.host, 'localhost')
        self.assertEqual(simulator.port, 2000)
        self.assertIsNotNone(simulator.weather)
        
        logger.info("✓ CARLA simulator initialization test passed")


class IntegrationTests(unittest.TestCase):
    """集成测试类 / Integration Tests Class"""
    
    def test_perception_planning_integration(self):
        """测试感知和规划模块集成 / Test perception and planning integration"""
        logger.info("Testing perception-planning integration...")
        
        # Initialize modules
        camera_perception = CameraPerception()
        lidar_perception = LidarPerception()
        planner = PathPlanner()
        
        # Create dummy sensor data
        camera_data = {
            'data': np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8),
            'timestamp': 1234567890
        }
        
        lidar_data = {
            'data': np.random.randn(1000, 4),
            'timestamp': 1234567890
        }
        
        # Process perception
        camera_result = camera_perception.process(camera_data)
        lidar_result = lidar_perception.process(lidar_data)
        
        # Fuse perception results
        perception_data = {
            'objects': camera_result.get('objects', []) + lidar_result.get('objects', []),
            'obstacles': lidar_result.get('obstacles', []),
            'lanes': camera_result.get('lanes', [])
        }
        
        # Test planning with fused perception
        current_pose = {'location': [0.0, 0.0, 0.0], 'rotation': [0.0, 0.0, 0.0]}
        goal = [50.0, 0.0]
        
        plan_result = planner.plan(current_pose, perception_data, goal)
        
        # Check integration result
        self.assertIsInstance(plan_result, dict)
        self.assertIn('waypoints', plan_result)
        
        logger.info("✓ Perception-planning integration test passed")
    
    def test_planning_control_integration(self):
        """测试规划和控制模块集成 / Test planning and control integration"""
        logger.info("Testing planning-control integration...")
        
        # Initialize modules
        planner = PathPlanner()
        controller = VehicleController()
        
        # Create planning scenario
        current_pose = {'location': [0.0, 0.0, 0.0], 'rotation': [0.0, 0.0, 0.0]}
        perception_data = {'objects': [], 'obstacles': []}
        goal = [30.0, 0.0]
        
        # Plan path
        plan_result = planner.plan(current_pose, perception_data, goal)
        
        # Test control with planned path
        current_state = {
            'speed': 5.0,
            'location': [0.0, 0.0, 0.0],
            'rotation': [0.0, 0.0, 0.0]
        }
        
        control_result = controller.compute_control(current_state, plan_result)
        
        # Check integration result
        self.assertIsInstance(control_result, dict)
        self.assertIn('throttle', control_result)
        self.assertIn('steer', control_result)
        
        logger.info("✓ Planning-control integration test passed")


def run_performance_benchmark():
    """运行性能基准测试 / Run performance benchmark"""
    logger.info("Running performance benchmark...")
    
    import time
    
    # Test perception performance
    camera_perception = CameraPerception()
    dummy_image = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
    camera_data = {'data': dummy_image, 'timestamp': time.time()}
    
    start_time = time.time()
    for _ in range(10):
        camera_perception.process(camera_data)
    camera_time = (time.time() - start_time) / 10
    
    logger.info(f"Camera perception avg time: {camera_time:.3f}s per frame")
    
    # Test planning performance
    planner = PathPlanner()
    current_pose = {'location': [0.0, 0.0, 0.0], 'rotation': [0.0, 0.0, 0.0]}
    perception_data = {'objects': [], 'obstacles': []}
    goal = [50.0, 0.0]
    
    start_time = time.time()
    for _ in range(10):
        planner.plan(current_pose, perception_data, goal)
    planning_time = (time.time() - start_time) / 10
    
    logger.info(f"Path planning avg time: {planning_time:.3f}s per step")
    
    # Test control performance
    controller = VehicleController()
    current_state = {'speed': 10.0, 'location': [0.0, 0.0, 0.0], 'rotation': [0.0, 0.0, 0.0]}
    planned_path = {'waypoints': [{'position': [10.0, 0.0], 'speed': 15.0}], 'behavior': 'cruise', 'valid': True}
    
    start_time = time.time()
    for _ in range(100):
        controller.compute_control(current_state, planned_path)
    control_time = (time.time() - start_time) / 100
    
    logger.info(f"Vehicle control avg time: {control_time:.4f}s per step")


def main():
    """主测试函数 / Main test function"""
    logger.info("🚗 CARLA自动驾驶系统测试开始 / Starting CARLA Autonomous Driving System Tests")
    logger.info("张涔熙的科研项目 - 系统测试 / Zhang Cixi's Research Project - System Testing")
    
    # Run unit tests
    logger.info("\n=== 单元测试 / Unit Tests ===")
    unittest.main(argv=[''], module=__name__, exit=False, verbosity=2)
    
    # Run performance benchmark
    logger.info("\n=== 性能基准测试 / Performance Benchmark ===")
    run_performance_benchmark()
    
    logger.info("\n✅ 所有测试完成 / All tests completed!")


if __name__ == "__main__":
    main()