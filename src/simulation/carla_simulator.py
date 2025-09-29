#!/usr/bin/env python3
"""
CARLA仿真器接口 / CARLA Simulator Interface
张涔熙的科研项目 - CARLA仿真环境管理模块

This module provides an interface to interact with the CARLA simulator,
including vehicle spawning, sensor setup, and data collection.
"""

import logging
import random
import time
from typing import Dict, List, Optional, Tuple

import carla
import numpy as np

logger = logging.getLogger(__name__)


class CarlaSimulator:
    """
    CARLA仿真器管理类 / CARLA Simulator Manager Class
    
    负责管理CARLA仿真环境，包括车辆生成、传感器配置、数据采集等
    Manages CARLA simulation environment including vehicle spawning, sensor setup, data collection
    """
    
    def __init__(self, host: str = 'localhost', port: int = 2000, timeout: float = 10.0):
        """
        初始化CARLA仿真器 / Initialize CARLA simulator
        
        Args:
            host: CARLA服务器地址 / CARLA server host
            port: CARLA服务器端口 / CARLA server port
            timeout: 连接超时时间 / Connection timeout
        """
        self.host = host
        self.port = port
        self.timeout = timeout
        
        # CARLA objects
        self.client = None
        self.world = None
        self.blueprint_library = None
        self.vehicle = None
        self.sensors = {}
        self.sensor_data = {}
        
        # Simulation settings
        self.weather = carla.WeatherParameters.ClearNoon
        self.map_name = 'Town01'
        
        # Vehicle state
        self.spawn_point = None
        self.destination = None
        
    def connect(self) -> bool:
        """
        连接到CARLA服务器 / Connect to CARLA server
        
        Returns:
            连接是否成功 / Whether connection is successful
        """
        try:
            logger.info(f"Connecting to CARLA server at {self.host}:{self.port}")
            
            # Create client and connect
            self.client = carla.Client(self.host, self.port)
            self.client.set_timeout(self.timeout)
            
            # Load world and get blueprint library
            self.world = self.client.get_world()
            self.blueprint_library = self.world.get_blueprint_library()
            
            # Set weather
            self.world.set_weather(self.weather)
            
            logger.info(f"Connected to CARLA successfully. Map: {self.world.get_map().name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to CARLA: {e}")
            return False
    
    def disconnect(self):
        """断开CARLA连接 / Disconnect from CARLA"""
        try:
            if self.vehicle:
                self.vehicle.destroy()
            
            for sensor in self.sensors.values():
                if sensor.is_alive:
                    sensor.destroy()
            
            logger.info("Disconnected from CARLA")
        except Exception as e:
            logger.error(f"Error during disconnection: {e}")
    
    def spawn_vehicle(self, vehicle_type: str = 'model3') -> bool:
        """
        生成车辆 / Spawn vehicle
        
        Args:
            vehicle_type: 车辆类型 / Vehicle type
            
        Returns:
            生成是否成功 / Whether spawning is successful
        """
        try:
            # Get vehicle blueprint
            if vehicle_type == 'model3':
                vehicle_bp = self.blueprint_library.filter('tesla.model3')[0]
            else:
                vehicle_bp = self.blueprint_library.filter('vehicle.*')[0]
            
            # Get spawn points
            spawn_points = self.world.get_map().get_spawn_points()
            self.spawn_point = random.choice(spawn_points)
            
            # Spawn vehicle
            self.vehicle = self.world.spawn_actor(vehicle_bp, self.spawn_point)
            
            # Set destination (random for now)
            self.destination = random.choice(spawn_points).location
            
            logger.info(f"Vehicle spawned at {self.spawn_point.location}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to spawn vehicle: {e}")
            return False
    
    def setup_sensors(self) -> bool:
        """
        设置传感器 / Setup sensors
        
        Returns:
            设置是否成功 / Whether setup is successful
        """
        if not self.vehicle:
            logger.error("No vehicle spawned. Call spawn_vehicle() first.")
            return False
        
        try:
            # Camera sensor
            camera_bp = self.blueprint_library.find('sensor.camera.rgb')
            camera_bp.set_attribute('image_size_x', '800')
            camera_bp.set_attribute('image_size_y', '600')
            camera_bp.set_attribute('fov', '90')
            
            camera_transform = carla.Transform(
                carla.Location(x=2.0, z=1.4),
                carla.Rotation(pitch=-15)
            )
            camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
            camera.listen(lambda data: self._camera_callback(data))
            self.sensors['camera'] = camera
            
            # LiDAR sensor
            lidar_bp = self.blueprint_library.find('sensor.lidar.ray_cast')
            lidar_bp.set_attribute('channels', '64')
            lidar_bp.set_attribute('points_per_second', '100000')
            lidar_bp.set_attribute('rotation_frequency', '20')
            lidar_bp.set_attribute('range', '100')
            
            lidar_transform = carla.Transform(carla.Location(x=0, z=2.4))
            lidar = self.world.spawn_actor(lidar_bp, lidar_transform, attach_to=self.vehicle)
            lidar.listen(lambda data: self._lidar_callback(data))
            self.sensors['lidar'] = lidar
            
            # Collision sensor
            collision_bp = self.blueprint_library.find('sensor.other.collision')
            collision = self.world.spawn_actor(collision_bp, carla.Transform(), attach_to=self.vehicle)
            collision.listen(lambda event: self._collision_callback(event))
            self.sensors['collision'] = collision
            
            logger.info("Sensors setup completed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup sensors: {e}")
            return False
    
    def _camera_callback(self, image):
        """相机数据回调 / Camera data callback"""
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]  # Remove alpha channel
        self.sensor_data['camera'] = {
            'data': array,
            'timestamp': image.timestamp,
            'frame': image.frame
        }
    
    def _lidar_callback(self, lidar_data):
        """LiDAR数据回调 / LiDAR data callback"""
        points = np.frombuffer(lidar_data.raw_data, dtype=np.dtype('f4'))
        points = np.reshape(points, (int(points.shape[0] / 4), 4))
        self.sensor_data['lidar'] = {
            'data': points,
            'timestamp': lidar_data.timestamp,
            'frame': lidar_data.frame
        }
    
    def _collision_callback(self, event):
        """碰撞检测回调 / Collision detection callback"""
        logger.warning(f"Collision detected with {event.other_actor.type_id}")
        self.sensor_data['collision'] = {
            'actor': event.other_actor.type_id,
            'timestamp': event.timestamp,
            'impulse': [event.normal_impulse.x, event.normal_impulse.y, event.normal_impulse.z]
        }
    
    def get_sensor_data(self) -> Optional[Dict]:
        """
        获取传感器数据 / Get sensor data
        
        Returns:
            传感器数据字典 / Sensor data dictionary
        """
        return self.sensor_data.copy() if self.sensor_data else None
    
    def get_vehicle_pose(self) -> Optional[carla.Transform]:
        """
        获取车辆位姿 / Get vehicle pose
        
        Returns:
            车辆变换矩阵 / Vehicle transform
        """
        if self.vehicle:
            return self.vehicle.get_transform()
        return None
    
    def get_vehicle_state(self) -> Optional[Dict]:
        """
        获取车辆状态 / Get vehicle state
        
        Returns:
            车辆状态字典 / Vehicle state dictionary
        """
        if not self.vehicle:
            return None
        
        transform = self.vehicle.get_transform()
        velocity = self.vehicle.get_velocity()
        acceleration = self.vehicle.get_acceleration()
        
        return {
            'location': [transform.location.x, transform.location.y, transform.location.z],
            'rotation': [transform.rotation.pitch, transform.rotation.yaw, transform.rotation.roll],
            'velocity': [velocity.x, velocity.y, velocity.z],
            'acceleration': [acceleration.x, acceleration.y, acceleration.z],
            'speed': np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        }
    
    def get_vehicle_speed(self) -> float:
        """
        获取车辆速度 / Get vehicle speed
        
        Returns:
            车辆速度(m/s) / Vehicle speed in m/s
        """
        if self.vehicle:
            velocity = self.vehicle.get_velocity()
            return np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        return 0.0
    
    def get_destination(self) -> Optional[carla.Location]:
        """
        获取目标位置 / Get destination
        
        Returns:
            目标位置 / Destination location
        """
        return self.destination
    
    def apply_control(self, control_command: Dict):
        """
        应用控制命令 / Apply control command
        
        Args:
            control_command: 控制命令字典 / Control command dictionary
        """
        if not self.vehicle:
            return
        
        control = carla.VehicleControl(
            throttle=control_command.get('throttle', 0.0),
            steer=control_command.get('steer', 0.0),
            brake=control_command.get('brake', 0.0),
            hand_brake=control_command.get('hand_brake', False),
            reverse=control_command.get('reverse', False)
        )
        
        self.vehicle.apply_control(control)
    
    def set_weather(self, weather_name: str):
        """
        设置天气 / Set weather
        
        Args:
            weather_name: 天气名称 / Weather name
        """
        weather_presets = {
            'clear': carla.WeatherParameters.ClearNoon,
            'cloudy': carla.WeatherParameters.CloudyNoon,
            'wet': carla.WeatherParameters.WetNoon,
            'rain': carla.WeatherParameters.HardRainNoon,
            'fog': carla.WeatherParameters.ClearSunset
        }
        
        if weather_name in weather_presets:
            self.weather = weather_presets[weather_name]
            if self.world:
                self.world.set_weather(self.weather)
            logger.info(f"Weather set to {weather_name}")
        else:
            logger.warning(f"Unknown weather preset: {weather_name}")
    
    def load_map(self, map_name: str):
        """
        加载地图 / Load map
        
        Args:
            map_name: 地图名称 / Map name
        """
        try:
            if self.client:
                self.world = self.client.load_world(map_name)
                self.map_name = map_name
                logger.info(f"Loaded map: {map_name}")
            else:
                logger.error("Not connected to CARLA server")
        except Exception as e:
            logger.error(f"Failed to load map {map_name}: {e}")


def main():
    """测试函数 / Test function"""
    simulator = CarlaSimulator()
    
    if simulator.connect():
        if simulator.spawn_vehicle():
            if simulator.setup_sensors():
                logger.info("CARLA simulator setup completed successfully")
                
                # Run for a short test
                for i in range(100):
                    data = simulator.get_sensor_data()
                    state = simulator.get_vehicle_state()
                    
                    if data and state:
                        logger.info(f"Step {i}: Speed = {state['speed']:.2f} m/s")
                    
                    # Simple forward movement
                    simulator.apply_control({'throttle': 0.3, 'steer': 0.0})
                    time.sleep(0.1)
            
        simulator.disconnect()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()