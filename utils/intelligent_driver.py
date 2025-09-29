# -*- coding: utf-8 -*-

import carla
import numpy as np
import math
import random

class IntelligentDriver:
    """智能驾驶控制器"""
    
    def __init__(self, vehicle, world):
        self.vehicle = vehicle
        self.world = world
        self.map = world.get_map()
        
        # 驾驶参数
        self.target_speed = 30.0  # 目标速度 km/h
        self.max_steer = 1.0
        self.max_throttle = 0.8
        self.max_brake = 0.3
        
        # 路径规划
        self.waypoints = []
        self.current_waypoint_index = 0
        self.waypoint_distance = 2.0  # 路径点间距
        
        # 避障参数
        self.detection_distance = 20.0
        self.safe_distance = 5.0
        
        print("Intelligent Driver initialized")
        
    def get_current_waypoint(self):
        """获取当前位置的路径点"""
        location = self.vehicle.get_location()
        waypoint = self.map.get_waypoint(location)
        return waypoint
        
    def generate_route(self, distance=200):
        """生成前方路径"""
        current_waypoint = self.get_current_waypoint()
        
        self.waypoints = []
        next_waypoint = current_waypoint
        
        for i in range(int(distance / self.waypoint_distance)):
            if next_waypoint:
                self.waypoints.append(next_waypoint)
                next_waypoints = next_waypoint.next(self.waypoint_distance)
                
                if next_waypoints:
                    # 随机选择方向增加探索性
                    if len(next_waypoints) > 1 and random.random() > 0.8:
                        next_waypoint = random.choice(next_waypoints)
                    else:
                        next_waypoint = next_waypoints[0]
                else:
                    break
                    
        print(f"Generated route with {len(self.waypoints)} waypoints")
        
    def get_target_waypoint(self):
        """获取目标路径点"""
        if not self.waypoints:
            return None
            
        vehicle_location = self.vehicle.get_location()
        
        # 找到最近的路径点
        min_distance = float('inf')
        closest_index = 0
        
        for i, waypoint in enumerate(self.waypoints):
            distance = vehicle_location.distance(waypoint.transform.location)
            if distance < min_distance:
                min_distance = distance
                closest_index = i
                
        # 选择前方的路径点作为目标
        target_index = min(closest_index + 3, len(self.waypoints) - 1)
        return self.waypoints[target_index]
        
    def detect_obstacles(self):
        """检测前方障碍物"""
        vehicle_location = self.vehicle.get_location()
        vehicle_transform = self.vehicle.get_transform()
        
        # 获取前方向量
        forward_vector = vehicle_transform.get_forward_vector()
        
        obstacles = []
        
        # 检测其他车辆
        for actor in self.world.get_actors().filter('vehicle.*'):
            if actor.id == self.vehicle.id:
                continue
                
            actor_location = actor.get_location()
            distance = vehicle_location.distance(actor_location)
            
            if distance < self.detection_distance:
                # 检查是否在前方
                to_actor = actor_location - vehicle_location
                dot_product = (forward_vector.x * to_actor.x + 
                             forward_vector.y * to_actor.y)
                
                if dot_product > 0:  # 在前方
                    obstacles.append({
                        'actor': actor,
                        'distance': distance,
                        'location': actor_location
                    })
                    
        return obstacles
        
    def calculate_steering(self, target_waypoint):
        """计算转向角度"""
        if not target_waypoint:
            return 0.0
            
        vehicle_transform = self.vehicle.get_transform()
        vehicle_location = vehicle_transform.location
        
        # 计算目标方向
        target_location = target_waypoint.transform.location
        target_vector = target_location - vehicle_location
        
        # 当前朝向
        forward_vector = vehicle_transform.get_forward_vector()
        
        # 计算角度差
        angle = math.atan2(target_vector.y, target_vector.x) - math.atan2(forward_vector.y, forward_vector.x)
        
        # 标准化角度到 [-π, π]
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
            
        # 转换为转向值 [-1, 1]
        steer = angle / math.pi
        return np.clip(steer, -self.max_steer, self.max_steer)
        
    def calculate_throttle_brake(self, obstacles):
        """计算油门和刹车"""
        velocity = self.vehicle.get_velocity()
        current_speed = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)  # km/h
        
        throttle = 0.0
        brake = 0.0
        
        # 检查前方障碍物
        if obstacles:
            closest_obstacle = min(obstacles, key=lambda x: x['distance'])
            if closest_obstacle['distance'] < self.safe_distance:
                # 紧急刹车
                brake = self.max_brake
                throttle = 0.0
            elif closest_obstacle['distance'] < self.safe_distance * 2:
                # 减速
                brake = 0.1
                throttle = 0.0
            else:
                # 正常驾驶但保持谨慎
                if current_speed < self.target_speed * 0.7:
                    throttle = 0.4
        else:
            # 无障碍物，根据目标速度调整
            if current_speed < self.target_speed:
                throttle = min(0.6, (self.target_speed - current_speed) / 10.0)
            elif current_speed > self.target_speed * 1.2:
                brake = 0.2
                
        return throttle, brake
        
    def get_control(self):
        """获取智能驾驶控制命令"""
        # 生成或更新路径
        if not self.waypoints or len(self.waypoints) < 10:
            self.generate_route()
            
        # 获取目标点
        target_waypoint = self.get_target_waypoint()
        
        # 检测障碍物
        obstacles = self.detect_obstacles()
        
        # 计算控制命令
        steer = self.calculate_steering(target_waypoint)
        throttle, brake = self.calculate_throttle_brake(obstacles)
        
        # 创建控制对象
        control = carla.VehicleControl()
        control.steer = steer
        control.throttle = throttle
        control.brake = brake
        control.hand_brake = False
        control.reverse = False
        
        return control, len(obstacles)

class DrivingModes:
    """不同的驾驶模式"""
    
    @staticmethod
    def random_drive():
        """随机驾驶（当前使用的）"""
        control = carla.VehicleControl()
        control.throttle = np.random.uniform(0.3, 0.6)
        control.steer = np.random.uniform(-0.3, 0.3)
        return control
        
    @staticmethod
    def cruise_drive(vehicle, world, target_speed=25.0):
        """巡航驾驶 - 沿道路行驶"""
        map_obj = world.get_map()
        current_waypoint = map_obj.get_waypoint(vehicle.get_location())
        
        if current_waypoint:
            next_waypoints = current_waypoint.next(5.0)
            if next_waypoints:
                target_waypoint = next_waypoints[0]
                
                # 计算转向
                vehicle_transform = vehicle.get_transform()
                target_location = target_waypoint.transform.location
                vehicle_location = vehicle_transform.location
                
                target_vector = target_location - vehicle_location
                forward_vector = vehicle_transform.get_forward_vector()
                
                angle = math.atan2(target_vector.y, target_vector.x) - math.atan2(forward_vector.y, forward_vector.x)
                while angle > math.pi:
                    angle -= 2 * math.pi
                while angle < -math.pi:
                    angle += 2 * math.pi
                    
                steer = angle / math.pi * 0.5
                
                # 速度控制
                velocity = vehicle.get_velocity()
                current_speed = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2)
                
                if current_speed < target_speed:
                    throttle = 0.5
                    brake = 0.0
                else:
                    throttle = 0.2
                    brake = 0.0
                    
                control = carla.VehicleControl()
                control.steer = np.clip(steer, -0.5, 0.5)
                control.throttle = throttle
                control.brake = brake
                
                return control
                
        # 回退到随机驾驶
        return DrivingModes.random_drive()
        
    @staticmethod
    def exploration_drive(vehicle, world):
        """探索驾驶 - 定期改变方向探索不同区域"""
        map_obj = world.get_map()
        current_waypoint = map_obj.get_waypoint(vehicle.get_location())
        
        if current_waypoint and random.random() > 0.95:  # 5%概率改变方向
            # 随机选择不同的路径
            next_waypoints = current_waypoint.next(10.0)
            if len(next_waypoints) > 1:
                target_waypoint = random.choice(next_waypoints)
            elif next_waypoints:
                target_waypoint = next_waypoints[0]
            else:
                return DrivingModes.random_drive()
                
            # 计算朝向目标的控制
            vehicle_transform = vehicle.get_transform()
            target_location = target_waypoint.transform.location
            vehicle_location = vehicle_transform.location
            
            target_vector = target_location - vehicle_location
            forward_vector = vehicle_transform.get_forward_vector()
            
            angle = math.atan2(target_vector.y, target_vector.x) - math.atan2(forward_vector.y, forward_vector.x)
            steer = angle / math.pi * 0.3
            
            control = carla.VehicleControl()
            control.steer = np.clip(steer, -0.4, 0.4)
            control.throttle = np.random.uniform(0.4, 0.7)
            control.brake = 0.0
            
            return control
            
        return DrivingModes.cruise_drive(vehicle, world)

# 使用示例
def get_intelligent_control(vehicle, world, driver=None):
    """获取智能驾驶控制"""
    if driver is None:
        # 使用探索驾驶模式
        return DrivingModes.exploration_drive(vehicle, world)
    else:
        # 使用完整的智能驾驶系统
        control, obstacle_count = driver.get_control()
        return control