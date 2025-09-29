# -*- coding: utf-8 -*-
"""
CARLA Vehicle Spawn Test Script
Test spawning and controlling vehicles in CARLA world
"""

import sys
import os
import time
import random
from typing import List, Optional

# Import CARLA (already installed in conda environment)
try:
    import carla
    print("? CARLA module imported successfully")
except ImportError as e:
    print(f"? CARLA module import failed: {e}")
    sys.exit(1)


class VehicleTestManager:
    """Vehicle Test Manager"""
    
    def __init__(self, host='localhost', port=2000):
        self.client = None
        self.world = None
        self.spawned_actors: List[carla.Actor] = []
        
        try:
            self.client = carla.Client(host, port)
            self.client.set_timeout(10.0)
            self.world = self.client.get_world()
            print(f"? 连接到CARLA服务器 {host}:{port}")
        except Exception as e:
            print(f"? 连接失败: {e}")
            raise
    
    def spawn_random_vehicle(self) -> Optional[carla.Actor]:
        """生成一个随机车辆"""
        try:
            # 获取车辆蓝图
            blueprint_library = self.world.get_blueprint_library()
            vehicle_blueprints = blueprint_library.filter('vehicle.*')
            
            # 随机选择车辆
            vehicle_bp = random.choice(vehicle_blueprints)
            
            # 设置随机颜色（如果支持）
            if vehicle_bp.has_attribute('color'):
                color = random.choice(vehicle_bp.get_attribute('color').recommended_values)
                vehicle_bp.set_attribute('color', color)
            
            # 获取生成点
            spawn_points = self.world.get_map().get_spawn_points()
            spawn_point = random.choice(spawn_points)
            
            # 生成车辆
            vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
            self.spawned_actors.append(vehicle)
            
            print(f"? 生成车辆: {vehicle_bp.id} 在位置 ({spawn_point.location.x:.1f}, {spawn_point.location.y:.1f})")
            return vehicle
            
        except Exception as e:
            print(f"? 生成车辆失败: {e}")
            return None
    
    def enable_autopilot(self, vehicle: carla.Actor):
        """启用自动驾驶"""
        try:
            vehicle.set_autopilot(True)
            print(f"? 为车辆 {vehicle.id} 启用自动驾驶")
        except Exception as e:
            print(f"? 启用自动驾驶失败: {e}")
    
    def test_sensor_data(self, vehicle: carla.Actor):
        """测试传感器数据收集"""
        try:
            blueprint_library = self.world.get_blueprint_library()
            
            # 添加RGB相机
            camera_bp = blueprint_library.find('sensor.camera.rgb')
            camera_bp.set_attribute('image_size_x', '800')
            camera_bp.set_attribute('image_size_y', '600')
            
            # 相机变换（安装在车顶）
            camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
            
            # 生成相机
            camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
            self.spawned_actors.append(camera)
            
            # 设置数据回调
            def save_image(image):
                image.save_to_disk(f'data/camera_{image.frame}.png')
                print(f"? 保存图像: camera_{image.frame}.png")
            
            camera.listen(save_image)
            print("? RGB相机传感器已设置")
            
            return camera
            
        except Exception as e:
            print(f"? 传感器设置失败: {e}")
            return None
    
    def cleanup(self):
        """清理生成的角色"""
        print(f"正在清理 {len(self.spawned_actors)} 个生成的角色...")
        for actor in self.spawned_actors:
            try:
                actor.destroy()
            except Exception as e:
                print(f"清理角色时出错: {e}")
        self.spawned_actors.clear()
        print("? 清理完成")


def main():
    """主测试函数"""
    print("=" * 50)
    print("CARLA车辆生成和传感器测试")
    print("=" * 50)
    
    manager = None
    
    try:
        # 创建测试管理器
        manager = VehicleTestManager()
        
        # 生成3辆测试车辆
        print("\n1. 生成测试车辆...")
        vehicles = []
        for i in range(3):
            vehicle = manager.spawn_random_vehicle()
            if vehicle:
                vehicles.append(vehicle)
        
        print(f"? 成功生成 {len(vehicles)} 辆车辆")
        
        # 为车辆启用自动驾驶
        print("\n2. 启用自动驾驶...")
        for vehicle in vehicles:
            manager.enable_autopilot(vehicle)
        
        # 为第一辆车添加传感器
        if vehicles:
            print("\n3. 设置传感器...")
            camera = manager.test_sensor_data(vehicles[0])
        
        # 运行测试一段时间
        print("\n4. 运行仿真测试 (10秒)...")
        print("观察车辆自动驾驶行为...")
        
        for i in range(10):
            time.sleep(1)
            print(f"  测试进度: {i+1}/10 秒")
        
        print("\n? 测试完成!")
        
    except KeyboardInterrupt:
        print("\n用户中断测试")
    except Exception as e:
        print(f"\n? 测试过程中出错: {e}")
    finally:
        # 清理资源
        if manager:
            manager.cleanup()


if __name__ == '__main__':
    main()