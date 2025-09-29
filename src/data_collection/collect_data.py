#!/usr/bin/env python3
"""
数据收集脚本 / Data Collection Script
张涔熙的科研项目 - 自动驾驶数据收集工具

This script collects data from CARLA simulation for training and analysis.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.simulation.carla_simulator import CarlaSimulator
from src.data_collection.data_logger import DataLogger

logger = logging.getLogger(__name__)


class DataCollector:
    """数据收集器类 / Data Collector Class"""
    
    def __init__(self, output_dir: str = "data/collected"):
        """
        初始化数据收集器 / Initialize data collector
        
        Args:
            output_dir: 输出目录 / Output directory
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.simulator = None
        self.data_logger = None
        
    def setup(self):
        """设置收集环境 / Setup collection environment"""
        logger.info("Setting up data collection environment...")
        
        # Initialize CARLA simulator
        self.simulator = CarlaSimulator()
        if not self.simulator.connect():
            raise Exception("Failed to connect to CARLA server")
        
        # Spawn vehicle and setup sensors
        if not self.simulator.spawn_vehicle():
            raise Exception("Failed to spawn vehicle")
        
        if not self.simulator.setup_sensors():
            raise Exception("Failed to setup sensors")
        
        # Initialize data logger
        self.data_logger = DataLogger(log_dir=str(self.output_dir))
        
        logger.info("Data collection environment ready")
    
    def collect_data(self, duration: float = 300.0, scenario: str = "free_roam"):
        """
        收集数据 / Collect data
        
        Args:
            duration: 收集时长(秒) / Collection duration in seconds
            scenario: 场景类型 / Scenario type
        """
        logger.info(f"Starting data collection for {duration} seconds...")
        logger.info(f"Scenario: {scenario}")
        
        start_time = time.time()
        step_count = 0
        
        try:
            while time.time() - start_time < duration:
                step_count += 1
                step_start = time.time()
                
                # Get sensor data
                sensor_data = self.simulator.get_sensor_data()
                vehicle_state = self.simulator.get_vehicle_state()
                
                if sensor_data and vehicle_state:
                    # Create data entry
                    data_entry = {
                        'step': step_count,
                        'timestamp': time.time(),
                        'scenario': scenario,
                        'sensor_data': sensor_data,
                        'vehicle_state': vehicle_state,
                        'vehicle_pose': self.simulator.get_vehicle_pose()
                    }
                    
                    # Log data
                    self.data_logger.log_step(data_entry)
                    
                    # Save raw sensor data periodically
                    if step_count % 50 == 0:
                        if 'camera' in sensor_data:
                            self.data_logger.save_sensor_data(
                                'camera', 
                                sensor_data['camera']['data'],
                                {'frame': sensor_data['camera'].get('frame', 0)}
                            )
                        
                        if 'lidar' in sensor_data:
                            self.data_logger.save_sensor_data(
                                'lidar',
                                sensor_data['lidar']['data'],
                                {'frame': sensor_data['lidar'].get('frame', 0)}
                            )
                
                # Simple autonomous behavior for data collection
                control_command = self._generate_collection_behavior(vehicle_state, scenario)
                self.simulator.apply_control(control_command)
                
                # Control timing
                step_time = time.time() - step_start
                if step_time < 0.05:  # 20 FPS
                    time.sleep(0.05 - step_time)
                
                # Progress update
                if step_count % 100 == 0:
                    elapsed = time.time() - start_time
                    remaining = duration - elapsed
                    logger.info(f"Collected {step_count} frames, {remaining:.1f}s remaining")
        
        except KeyboardInterrupt:
            logger.info("Data collection interrupted by user")
        except Exception as e:
            logger.error(f"Error during data collection: {e}")
        finally:
            # Save session
            self.data_logger.save_session()
            logger.info(f"Data collection completed. {step_count} frames collected.")
    
    def _generate_collection_behavior(self, vehicle_state: dict, scenario: str) -> dict:
        """
        生成收集行为 / Generate collection behavior
        
        Args:
            vehicle_state: 车辆状态 / Vehicle state
            scenario: 场景类型 / Scenario type
            
        Returns:
            控制命令 / Control command
        """
        current_speed = vehicle_state.get('speed', 0.0)
        
        if scenario == "free_roam":
            # Simple cruise control
            target_speed = 10.0  # m/s
            
            if current_speed < target_speed:
                return {'throttle': 0.5, 'brake': 0.0, 'steer': 0.0}
            else:
                return {'throttle': 0.0, 'brake': 0.2, 'steer': 0.0}
        
        elif scenario == "stop_go":
            # Alternating stop and go
            import math
            cycle_time = 20.0  # 20 second cycle
            current_time = time.time() % cycle_time
            
            if current_time < 10.0:  # Go phase
                return {'throttle': 0.6, 'brake': 0.0, 'steer': 0.0}
            else:  # Stop phase
                return {'throttle': 0.0, 'brake': 0.8, 'steer': 0.0}
        
        elif scenario == "turning":
            # Turning behavior
            import math
            steer_angle = 0.3 * math.sin(time.time() * 0.5)
            return {'throttle': 0.4, 'brake': 0.0, 'steer': steer_angle}
        
        else:
            # Default behavior
            return {'throttle': 0.3, 'brake': 0.0, 'steer': 0.0}
    
    def cleanup(self):
        """清理资源 / Cleanup resources"""
        if self.simulator:
            self.simulator.disconnect()
        logger.info("Data collection cleanup completed")


def main():
    """主函数 / Main function"""
    parser = argparse.ArgumentParser(description='CARLA Autonomous Driving Data Collection')
    parser.add_argument('--duration', type=float, default=300.0,
                       help='Data collection duration in seconds (default: 300)')
    parser.add_argument('--scenario', type=str, default='free_roam',
                       choices=['free_roam', 'stop_go', 'turning'],
                       help='Data collection scenario')
    parser.add_argument('--output', type=str, default='data/collected',
                       help='Output directory for collected data')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('data_collection.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger.info("🚗 CARLA自动驾驶数据收集开始")
    logger.info("CARLA Autonomous Driving Data Collection")
    logger.info("张涔熙的科研项目 / Zhang Cixi's Research Project")
    logger.info(f"Scenario: {args.scenario}, Duration: {args.duration}s")
    
    collector = DataCollector(args.output)
    
    try:
        collector.setup()
        collector.collect_data(args.duration, args.scenario)
    except Exception as e:
        logger.error(f"Data collection failed: {e}")
        sys.exit(1)
    finally:
        collector.cleanup()
    
    logger.info("✅ Data collection completed successfully!")


if __name__ == "__main__":
    main()