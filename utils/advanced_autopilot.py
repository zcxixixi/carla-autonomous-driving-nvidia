# -*- coding: utf-8 -*-

import carla
import numpy as np
import math
import random
from collections import deque
import time

class CARLAAutopilot:
    """集成CARLA官方和改进的自动驾驶控制器"""
    
    def __init__(self, vehicle, world):
        self.vehicle = vehicle
        self.world = world
        self.map = world.get_map()
        
        # 使用CARLA内置的自动驾驶
        self.use_builtin_autopilot = False
        
        # PID控制器参数
        self.speed_controller = PIDController(Kp=1.0, Ki=0.1, Kd=0.01)
        self.steering_controller = PIDController(Kp=1.0, Ki=0.0, Kd=0.1)
        
        # 目标参数
        self.target_speed = 25.0  # km/h
        self.waypoint_buffer = deque(maxlen=50)
        
        # 状态记录
        self.prev_error = 0.0
        self.integral = 0.0
        
        print("CARLA Autopilot System initialized")
        
    def enable_builtin_autopilot(self):
        """启用CARLA内置自动驾驶"""
        self.vehicle.set_autopilot(True)
        self.use_builtin_autopilot = True
        print("CARLA built-in autopilot enabled")
        
    def disable_builtin_autopilot(self):
        """禁用CARLA内置自动驾驶"""
        self.vehicle.set_autopilot(False)
        self.use_builtin_autopilot = False
        print("CARLA built-in autopilot disabled")
        
    def get_waypoint_path(self, distance=50):
        """获取前方路径点"""
        if not self.waypoint_buffer:
            current_waypoint = self.map.get_waypoint(self.vehicle.get_location())
            if current_waypoint:
                next_waypoint = current_waypoint
                for _ in range(int(distance / 2)):
                    if next_waypoint:
                        self.waypoint_buffer.append(next_waypoint)
                        next_waypoints = next_waypoint.next(2.0)
                        if next_waypoints:
                            next_waypoint = next_waypoints[0]
                        else:
                            break
                            
        return list(self.waypoint_buffer)

class PIDController:
    """PID控制器"""
    
    def __init__(self, Kp=1.0, Ki=0.0, Kd=0.0):
        self.Kp = Kp
        self.Ki = Ki  
        self.Kd = Kd
        self.prev_error = 0.0
        self.integral = 0.0
        
    def update(self, error, dt=0.1):
        """更新PID控制"""
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        
        return output

class ModernDrivingController:
    """现代化自动驾驶控制器 - 集成多种先进算法"""
    
    @staticmethod
    def pure_pursuit_control(vehicle, world, lookahead_distance=10.0):
        """Pure Pursuit算法 - 经典路径跟踪算法"""
        try:
            # 获取当前位置和朝向
            vehicle_transform = vehicle.get_transform()
            vehicle_location = vehicle_transform.location
            vehicle_yaw = math.radians(vehicle_transform.rotation.yaw)
            
            # 获取目标路径点
            map_obj = world.get_map()
            current_waypoint = map_obj.get_waypoint(vehicle_location)
            
            if current_waypoint:
                # 寻找lookahead距离处的目标点
                target_waypoint = current_waypoint
                accumulated_distance = 0
                
                while accumulated_distance < lookahead_distance:
                    next_waypoints = target_waypoint.next(2.0)
                    if next_waypoints:
                        target_waypoint = next_waypoints[0]
                        accumulated_distance += 2.0
                    else:
                        break
                
                # Pure Pursuit计算
                target_location = target_waypoint.transform.location
                dx = target_location.x - vehicle_location.x
                dy = target_location.y - vehicle_location.y
                
                # 转换到车辆坐标系
                cos_yaw = math.cos(vehicle_yaw)
                sin_yaw = math.sin(vehicle_yaw)
                local_x = dx * cos_yaw + dy * sin_yaw
                local_y = -dx * sin_yaw + dy * cos_yaw
                
                # 计算曲率
                curvature = 2 * local_y / (lookahead_distance ** 2)
                
                # 转换为转向角
                steering_angle = math.atan(curvature * 2.7)  # 2.7是车辆轴距
                steer = steering_angle / (math.pi / 6)  # 标准化到[-1, 1]
                
                # 速度控制
                velocity = vehicle.get_velocity()
                current_speed = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2)
                target_speed = 25.0
                
                if abs(curvature) > 0.1:  # 转弯
                    target_speed *= 0.7
                    
                if current_speed < target_speed:
                    throttle = 0.4
                    brake = 0.0
                else:
                    throttle = 0.1
                    brake = 0.0
                
                control = carla.VehicleControl()
                control.steer = np.clip(steer, -0.5, 0.5)
                control.throttle = throttle
                control.brake = brake
                
                return control
                
        except Exception as e:
            print(f"Pure Pursuit error: {e}")
            
        return ModernDrivingController.stanley_control(vehicle, world)
    
    @staticmethod
    def stanley_control(vehicle, world):
        """Stanley控制算法 - 斯坦福自动驾驶算法"""
        try:
            # 获取车辆状态
            vehicle_transform = vehicle.get_transform()
            vehicle_location = vehicle_transform.location
            vehicle_yaw = math.radians(vehicle_transform.rotation.yaw)
            
            velocity = vehicle.get_velocity()
            current_speed = math.sqrt(velocity.x**2 + velocity.y**2)  # m/s
            
            # 获取最近路径点
            map_obj = world.get_map()
            current_waypoint = map_obj.get_waypoint(vehicle_location)
            
            if current_waypoint:
                # 计算横向误差
                waypoint_location = current_waypoint.transform.location
                waypoint_yaw = math.radians(current_waypoint.transform.rotation.yaw)
                
                # 车辆到路径点的向量
                dx = waypoint_location.x - vehicle_location.x
                dy = waypoint_location.y - vehicle_location.y
                
                # 计算横向误差（垂直于路径的距离）
                cross_track_error = -math.sin(waypoint_yaw) * dx + math.cos(waypoint_yaw) * dy
                
                # 计算航向误差
                heading_error = waypoint_yaw - vehicle_yaw
                
                # 标准化航向误差到[-π, π]
                while heading_error > math.pi:
                    heading_error -= 2 * math.pi
                while heading_error < -math.pi:
                    heading_error += 2 * math.pi
                
                # Stanley控制律
                k_e = 0.5  # 横向误差增益
                k_soft = 1.0  # 软化参数
                
                cross_track_steer = math.atan2(k_e * cross_track_error, k_soft + current_speed)
                steer = heading_error + cross_track_steer
                
                # 速度控制
                target_speed = 22.0  # km/h
                speed_kmh = current_speed * 3.6
                
                # 根据转向角调整速度
                if abs(steer) > 0.2:
                    target_speed *= 0.8
                
                if speed_kmh < target_speed:
                    throttle = 0.4
                    brake = 0.0
                else:
                    throttle = 0.1
                    brake = 0.0
                
                control = carla.VehicleControl()
                control.steer = np.clip(steer, -0.4, 0.4)
                control.throttle = throttle
                control.brake = brake
                
                return control
                
        except Exception as e:
            print(f"Stanley control error: {e}")
            
        return ModernDrivingController.mpc_control(vehicle, world)
    
    @staticmethod 
    def mpc_control(vehicle, world):
        """简化MPC控制 - 模型预测控制"""
        try:
            # 获取车辆状态
            vehicle_transform = vehicle.get_transform()
            velocity = vehicle.get_velocity()
            current_speed = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2)
            
            # 获取预测路径
            map_obj = world.get_map()
            current_waypoint = map_obj.get_waypoint(vehicle.get_location())
            
            if current_waypoint:
                # 预测多个时间步的路径点
                prediction_horizon = 5
                waypoints = []
                next_waypoint = current_waypoint
                
                for i in range(prediction_horizon):
                    if next_waypoint:
                        waypoints.append(next_waypoint)
                        next_wps = next_waypoint.next(3.0)  # 3米间距
                        if next_wps:
                            next_waypoint = next_wps[0]
                
                if len(waypoints) >= 2:
                    # 简化的MPC - 计算最优转向角
                    target_waypoint = waypoints[min(2, len(waypoints)-1)]
                    
                    vehicle_location = vehicle_transform.location
                    target_location = target_waypoint.transform.location
                    
                    # 计算目标方向
                    dx = target_location.x - vehicle_location.x
                    dy = target_location.y - vehicle_location.y
                    target_yaw = math.atan2(dy, dx)
                    
                    vehicle_yaw = math.radians(vehicle_transform.rotation.yaw)
                    yaw_error = target_yaw - vehicle_yaw
                    
                    # 标准化角度
                    while yaw_error > math.pi:
                        yaw_error -= 2 * math.pi
                    while yaw_error < -math.pi:
                        yaw_error += 2 * math.pi
                    
                    # MPC控制律（简化版）
                    steer = yaw_error * 0.5
                    
                    # 自适应速度控制
                    curvature = abs(yaw_error)
                    if curvature > 0.3:
                        target_speed = 18.0
                    elif curvature > 0.1:
                        target_speed = 24.0
                    else:
                        target_speed = 28.0
                    
                    if current_speed < target_speed:
                        throttle = 0.5
                        brake = 0.0
                    elif current_speed > target_speed * 1.1:
                        throttle = 0.0
                        brake = 0.2
                    else:
                        throttle = 0.2
                        brake = 0.0
                    
                    control = carla.VehicleControl()
                    control.steer = np.clip(steer, -0.3, 0.3)
                    control.throttle = throttle
                    control.brake = brake
                    
                    return control
                    
        except Exception as e:
            print(f"MPC control error: {e}")
        
        # 最后的备选方案
        control = carla.VehicleControl()
        control.throttle = 0.3
        control.steer = 0.0
        return control

# 使用示例函数
def get_advanced_control(vehicle, world, algorithm='pure_pursuit'):
    """获取先进的自动驾驶控制"""
    
    if algorithm == 'pure_pursuit':
        return ModernDrivingController.pure_pursuit_control(vehicle, world)
    elif algorithm == 'stanley':
        return ModernDrivingController.stanley_control(vehicle, world)
    elif algorithm == 'mpc':
        return ModernDrivingController.mpc_control(vehicle, world)
    elif algorithm == 'builtin':
        # 使用CARLA内置自动驾驶
        vehicle.set_autopilot(True)
        return None  # 不需要手动控制
    else:
        return ModernDrivingController.pure_pursuit_control(vehicle, world)