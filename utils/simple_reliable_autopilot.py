# -*- coding: utf-8 -*-
"""
Simple but Reliable Autopilot - No More Junction Circles!
Anti-Circle Junction Navigation System
"""

import carla
import numpy as np
import math
import time
from collections import deque

class SimpleReliableAutopilot:
    """Simple Reliable Autopilot - Anti-Circle Junction Navigation"""
    
    def __init__(self, vehicle, world):
        self.vehicle = vehicle
        self.world = world
        self.map = world.get_map()
        
        # Simple parameters
        self.target_speed = 25.0  # km/h, conservative speed
        self.junction_speed = 10.0  # km/h, very slow in junctions
        
        # Junction detection and anti-circle
        self.in_junction = False
        self.junction_entry_time = None
        self.junction_timeout = 10.0  # max junction stay time
        self.last_locations = deque(maxlen=20)  # location history
        self.stuck_counter = 0
        self.forced_direction = None
        
        # Simple waypoint following
        self.current_waypoint = None
        self.target_waypoint = None
        
        print("Simple Reliable Autopilot initialized - Anti-Circle Mode!")
        
    def detect_stuck_in_circle(self):
        """Detect if stuck in junction circle"""
        if len(self.last_locations) < 10:
            return False
            
        # Calculate center of recent 10 positions
        recent_positions = list(self.last_locations)[-10:]
        center_x = sum(pos.x for pos in recent_positions) / len(recent_positions)
        center_y = sum(pos.y for pos in recent_positions) / len(recent_positions)
        
        # Check if circling around center
        distances_from_center = []
        for pos in recent_positions:
            dist = math.sqrt((pos.x - center_x)**2 + (pos.y - center_y)**2)
            distances_from_center.append(dist)
            
        # If all distances are close to center, we're circling
        avg_distance = sum(distances_from_center) / len(distances_from_center)
        if avg_distance < 5.0:  # 5m radius circle
            return True
            
        return False
        
    def escape_junction_circle(self):
        """Force escape from junction circle"""
        current_location = self.vehicle.get_location()
        current_waypoint = self.map.get_waypoint(current_location)
        
        if not current_waypoint:
            return None
            
        # Get all possible next waypoints
        next_waypoints = current_waypoint.next(5.0)
        
        if not next_waypoints:
            return current_waypoint
            
        # Force select a direction and stick to it
        if not self.forced_direction:
            # Choose direction, prefer straight
            vehicle_yaw = self.vehicle.get_transform().rotation.yaw
            
            best_waypoint = None
            min_angle_diff = float('inf')
            
            for wp in next_waypoints:
                wp_yaw = wp.transform.rotation.yaw
                angle_diff = abs(self._normalize_angle(wp_yaw - vehicle_yaw))
                
                # Prefer straight direction
                if angle_diff < min_angle_diff:
                    min_angle_diff = angle_diff
                    best_waypoint = wp
                    
            self.forced_direction = best_waypoint
            print("Anti-Circle: Forcing direction to escape junction!")
            
        return self.forced_direction
        
    def _normalize_angle(self, angle):
        """Normalize angle to [-180, 180]"""
        while angle > 180:
            angle -= 360
        while angle < -180:
            angle += 360
        return angle
        
    def compute_control(self):
        """Main control logic - Anti-circle junction navigation"""
        current_location = self.vehicle.get_location()
        self.last_locations.append(current_location)
        
        # 获取当前waypoint
        current_waypoint = self.map.get_waypoint(current_location)
        if not current_waypoint:
            return self._emergency_stop()
            
        # Detect junction state
        was_in_junction = self.in_junction
        self.in_junction = current_waypoint.is_junction
        
        # Junction entry logic
        if self.in_junction and not was_in_junction:
            self.junction_entry_time = time.time()
            self.forced_direction = None  # Reset forced direction
            print("Entering junction")
            
        # Junction exit logic
        elif not self.in_junction and was_in_junction:
            self.junction_entry_time = None
            self.forced_direction = None
            self.stuck_counter = 0
            print("Exited junction successfully")
            
        # Detect junction timeout or circle
        if self.in_junction:
            # Timeout detection
            if (self.junction_entry_time and 
                time.time() - self.junction_entry_time > self.junction_timeout):
                print("Junction timeout! Forcing escape...")
                target_waypoint = self.escape_junction_circle()
            # Circle detection
            elif self.detect_stuck_in_circle():
                self.stuck_counter += 1
                if self.stuck_counter > 5:
                    print("Circle detected! Forcing escape...")
                    target_waypoint = self.escape_junction_circle()
                else:
                    target_waypoint = self._get_normal_target_waypoint(current_waypoint)
            else:
                target_waypoint = self._get_normal_target_waypoint(current_waypoint)
        else:
            # Normal road driving
            target_waypoint = self._get_normal_target_waypoint(current_waypoint)
            
        if not target_waypoint:
            return self._emergency_stop()
            
        # Compute control commands
        target_speed = self.junction_speed if self.in_junction else self.target_speed
        
        # If in forced escape mode, increase speed
        if self.forced_direction:
            target_speed = min(target_speed * 1.5, 20.0)
            
        return self._compute_vehicle_control(target_waypoint, target_speed)
        
    def _get_normal_target_waypoint(self, current_waypoint):
        """Get normal target waypoint"""
        next_waypoints = current_waypoint.next(3.0)
        
        if not next_waypoints:
            return current_waypoint
            
        if len(next_waypoints) == 1:
            return next_waypoints[0]
            
        # Multiple choices, prefer straight
        vehicle_yaw = self.vehicle.get_transform().rotation.yaw
        
        best_waypoint = next_waypoints[0]
        min_angle_diff = float('inf')
        
        for wp in next_waypoints:
            wp_yaw = wp.transform.rotation.yaw
            angle_diff = abs(self._normalize_angle(wp_yaw - vehicle_yaw))
            
            if angle_diff < min_angle_diff:
                min_angle_diff = angle_diff
                best_waypoint = wp
                
        return best_waypoint
        
    def _compute_vehicle_control(self, target_waypoint, target_speed):
        """计算车辆控制指令"""
        # 转向控制
        steering = self._compute_steering(target_waypoint)
        
        # 速度控制
        throttle, brake = self._compute_speed_control(target_speed)
        
        control = carla.VehicleControl()
        control.steer = np.clip(steering, -0.5, 0.5)  # 限制转向角度
        control.throttle = np.clip(throttle, 0.0, 0.8)  # 限制油门
        control.brake = np.clip(brake, 0.0, 1.0)
        
        return control
        
    def _compute_steering(self, target_waypoint):
        """计算转向"""
        vehicle_transform = self.vehicle.get_transform()
        target_location = target_waypoint.transform.location
        
        # 向量计算
        dx = target_location.x - vehicle_transform.location.x
        dy = target_location.y - vehicle_transform.location.y
        
        # 车辆朝向
        yaw = math.radians(vehicle_transform.rotation.yaw)
        
        # 转换到车辆坐标系
        local_x = dx * math.cos(yaw) + dy * math.sin(yaw)
        local_y = -dx * math.sin(yaw) + dy * math.cos(yaw)
        
        # Pure pursuit控制
        lookahead_distance = math.sqrt(local_x**2 + local_y**2)
        if lookahead_distance < 0.1:
            return 0.0
            
        curvature = 2.0 * local_y / (lookahead_distance**2)
        
        # 转向角度
        wheelbase = 2.7
        steering_angle = math.atan(curvature * wheelbase)
        
        # 归一化
        max_steering = math.radians(25)  # 减小最大转向角
        return steering_angle / max_steering
        
    def _compute_speed_control(self, target_speed_kmh):
        """计算速度控制"""
        # 当前速度
        velocity = self.vehicle.get_velocity()
        current_speed_ms = math.sqrt(velocity.x**2 + velocity.y**2)
        current_speed_kmh = current_speed_ms * 3.6
        
        # 速度误差
        speed_error = target_speed_kmh - current_speed_kmh
        
        # 简单PID控制
        if speed_error > 2.0:
            throttle = min(0.5, speed_error * 0.05)  # 温和加速
            brake = 0.0
        elif speed_error < -2.0:
            throttle = 0.0
            brake = min(0.5, abs(speed_error) * 0.1)  # 温和制动
        else:
            throttle = 0.2  # 维持速度
            brake = 0.0
            
        return throttle, brake
        
    def _emergency_stop(self):
        """紧急停车"""
        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 0.0
        control.brake = 1.0
        return control

# 主接口函数
def get_simple_reliable_control(vehicle, world):
    """获取简单可靠的控制指令"""
    if not hasattr(vehicle, 'simple_reliable_autopilot'):
        vehicle.simple_reliable_autopilot = SimpleReliableAutopilot(vehicle, world)
        
    return vehicle.simple_reliable_autopilot.compute_control()