#!/usr/bin/env python3
"""
CARLA自动驾驶主程序 / CARLA Autonomous Driving Main Program
张涔熙的科研项目 - 自动驾驶系统核心模块

This module serves as the main entry point for the autonomous driving system,
integrating perception, planning, and control components with CARLA simulation.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import carla

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.simulation.carla_simulator import CarlaSimulator
from src.perception.camera_perception import CameraPerception
from src.perception.lidar_perception import LidarPerception
from src.planning.path_planner import PathPlanner
from src.control.vehicle_controller import VehicleController
from src.data_collection.data_logger import DataLogger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('autonomous_driving.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class AutonomousDrivingSystem:
    """
    自动驾驶系统主类 / Main Autonomous Driving System Class
    
    集成感知、规划、控制等模块，实现完整的自动驾驶功能
    Integrates perception, planning, and control modules for complete autonomous driving
    """
    
    def __init__(self, config_path: str = None):
        """
        初始化自动驾驶系统 / Initialize autonomous driving system
        
        Args:
            config_path: 配置文件路径 / Configuration file path
        """
        self.config_path = config_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize components
        self.simulator = None
        self.camera_perception = None
        self.lidar_perception = None
        self.path_planner = None
        self.vehicle_controller = None
        self.data_logger = None
        
        # System state
        self.is_running = False
        self.simulation_step = 0
        
    def initialize(self):
        """初始化所有系统组件 / Initialize all system components"""
        try:
            logger.info("Initializing autonomous driving system...")
            
            # Initialize CARLA simulator
            self.simulator = CarlaSimulator()
            self.simulator.connect()
            
            # Initialize perception modules
            self.camera_perception = CameraPerception(device=self.device)
            self.lidar_perception = LidarPerception(device=self.device)
            
            # Initialize planning and control
            self.path_planner = PathPlanner()
            self.vehicle_controller = VehicleController()
            
            # Initialize data logging
            self.data_logger = DataLogger()
            
            logger.info("System initialization completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize system: {e}")
            return False
    
    def run_autonomous_driving(self, duration: float = 60.0):
        """
        运行自动驾驶主循环 / Run autonomous driving main loop
        
        Args:
            duration: 运行时长(秒) / Duration in seconds
        """
        if not self.is_running:
            logger.error("System not initialized. Call initialize() first.")
            return
        
        logger.info(f"Starting autonomous driving for {duration} seconds...")
        start_time = time.time()
        
        try:
            while time.time() - start_time < duration:
                self.simulation_step += 1
                step_start = time.time()
                
                # Get sensor data from CARLA
                sensor_data = self.simulator.get_sensor_data()
                
                if sensor_data is None:
                    logger.warning("No sensor data received, skipping step")
                    continue
                
                # Perception: Process camera and LiDAR data
                camera_output = self.camera_perception.process(sensor_data.get('camera'))
                lidar_output = self.lidar_perception.process(sensor_data.get('lidar'))
                
                # Fusion of perception results
                perception_result = self._fuse_perception(camera_output, lidar_output)
                
                # Path planning based on perception
                planned_path = self.path_planner.plan(
                    current_pose=self.simulator.get_vehicle_pose(),
                    perception_data=perception_result,
                    goal=self.simulator.get_destination()
                )
                
                # Vehicle control
                control_command = self.vehicle_controller.compute_control(
                    current_state=self.simulator.get_vehicle_state(),
                    planned_path=planned_path
                )
                
                # Apply control to vehicle
                self.simulator.apply_control(control_command)
                
                # Log data for analysis
                self.data_logger.log_step({
                    'step': self.simulation_step,
                    'timestamp': time.time(),
                    'sensor_data': sensor_data,
                    'perception': perception_result,
                    'planned_path': planned_path,
                    'control': control_command
                })
                
                # Control loop timing
                step_time = time.time() - step_start
                if step_time < 0.05:  # Target 20 FPS
                    time.sleep(0.05 - step_time)
                
                # Print progress
                if self.simulation_step % 20 == 0:
                    logger.info(f"Step {self.simulation_step}, "
                              f"Time: {step_time:.3f}s, "
                              f"Speed: {self.simulator.get_vehicle_speed():.2f} m/s")
        
        except KeyboardInterrupt:
            logger.info("Autonomous driving interrupted by user")
        except Exception as e:
            logger.error(f"Error during autonomous driving: {e}")
        finally:
            logger.info("Autonomous driving session completed")
    
    def _fuse_perception(self, camera_output, lidar_output):
        """
        融合感知结果 / Fuse perception results
        
        Args:
            camera_output: 相机感知结果 / Camera perception output
            lidar_output: LiDAR感知结果 / LiDAR perception output
        
        Returns:
            融合后的感知结果 / Fused perception result
        """
        # Simple fusion strategy - can be improved with more sophisticated methods
        return {
            'objects': camera_output.get('objects', []) + lidar_output.get('objects', []),
            'lanes': camera_output.get('lanes', []),
            'depth': lidar_output.get('depth_map'),
            'semantic': camera_output.get('semantic_map')
        }
    
    def start(self):
        """启动系统 / Start the system"""
        if self.initialize():
            self.is_running = True
            logger.info("Autonomous driving system started successfully")
        else:
            logger.error("Failed to start autonomous driving system")
    
    def stop(self):
        """停止系统 / Stop the system"""
        self.is_running = False
        if self.simulator:
            self.simulator.disconnect()
        if self.data_logger:
            self.data_logger.save_session()
        logger.info("Autonomous driving system stopped")


def main():
    """主函数 / Main function"""
    parser = argparse.ArgumentParser(description='CARLA Autonomous Driving System')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Configuration file path')
    parser.add_argument('--duration', type=float, default=60.0,
                       help='Simulation duration in seconds')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Create and run autonomous driving system
    ads = AutonomousDrivingSystem(config_path=args.config)
    
    try:
        ads.start()
        if ads.is_running:
            ads.run_autonomous_driving(duration=args.duration)
    finally:
        ads.stop()


if __name__ == "__main__":
    main()