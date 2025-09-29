"""
CARLA工具函数
提供常用的CARLA操作辅助函数
"""

import sys
import os
from typing import List, Optional, Tuple
import random

# CARLA导入
try:
    carla_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'PythonAPI', 'carla')
    if os.path.exists(carla_path):
        sys.path.insert(0, carla_path)
    import carla
except ImportError:
    print("警告: CARLA模块未找到")


def connect_to_carla(host='localhost', port=2000, timeout=10.0) -> Tuple[carla.Client, carla.World]:
    """连接到CARLA服务器"""
    client = carla.Client(host, port)
    client.set_timeout(timeout)
    world = client.get_world()
    return client, world


def get_random_spawn_point(world: carla.World) -> carla.Transform:
    """获取随机生成点"""
    spawn_points = world.get_map().get_spawn_points()
    return random.choice(spawn_points)


def spawn_random_vehicle(world: carla.World) -> Optional[carla.Actor]:
    """生成随机车辆"""
    blueprint_library = world.get_blueprint_library()
    vehicle_blueprints = blueprint_library.filter('vehicle.*')
    
    vehicle_bp = random.choice(vehicle_blueprints)
    spawn_point = get_random_spawn_point(world)
    
    try:
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        return vehicle
    except Exception as e:
        print(f"生成车辆失败: {e}")
        return None


def cleanup_actors(actors: List[carla.Actor]):
    """清理角色列表"""
    for actor in actors:
        try:
            if actor.is_alive:
                actor.destroy()
        except Exception as e:
            print(f"清理角色时出错: {e}")


class CarlaLogger:
    """CARLA日志记录器"""
    
    @staticmethod
    def info(message: str):
        print(f"[INFO] {message}")
    
    @staticmethod
    def error(message: str):
        print(f"[ERROR] {message}")
    
    @staticmethod
    def warning(message: str):
        print(f"[WARNING] {message}")


class SafeCarlaClient:
    """安全的CARLA客户端包装器"""
    
    def __init__(self, host='localhost', port=2000):
        self.client = None
        self.world = None
        self.actors = []
        
        try:
            self.client, self.world = connect_to_carla(host, port)
            CarlaLogger.info(f"连接到CARLA服务器 {host}:{port}")
        except Exception as e:
            CarlaLogger.error(f"连接失败: {e}")
            raise
    
    def spawn_vehicle(self, blueprint_id=None) -> Optional[carla.Actor]:
        """安全生成车辆"""
        try:
            vehicle = spawn_random_vehicle(self.world)
            if vehicle:
                self.actors.append(vehicle)
                CarlaLogger.info(f"生成车辆 ID: {vehicle.id}")
            return vehicle
        except Exception as e:
            CarlaLogger.error(f"生成车辆失败: {e}")
            return None
    
    def cleanup(self):
        """清理所有资源"""
        CarlaLogger.info("清理CARLA资源...")
        cleanup_actors(self.actors)
        self.actors.clear()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()