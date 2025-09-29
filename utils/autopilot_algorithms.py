# -*- coding: utf-8 -*-

import carla
import numpy as np
import math
import random

class AdvancedAutopilot:
    """Advanced autopilot algorithms collection"""
    
    @staticmethod
    def pure_pursuit_control(vehicle, world, lookahead_distance=8.0):
        """Pure Pursuit Algorithm - Classic path tracking"""
        try:
            # Get vehicle state
            vehicle_transform = vehicle.get_transform()
            vehicle_location = vehicle_transform.location
            vehicle_yaw = math.radians(vehicle_transform.rotation.yaw)
            
            # Get target waypoint
            map_obj = world.get_map()
            current_waypoint = map_obj.get_waypoint(vehicle_location)
            
            if current_waypoint:
                # Find lookahead target
                target_waypoint = current_waypoint
                accumulated_distance = 0
                
                while accumulated_distance < lookahead_distance:
                    next_waypoints = target_waypoint.next(2.0)
                    if next_waypoints:
                        target_waypoint = next_waypoints[0]
                        accumulated_distance += 2.0
                    else:
                        break
                
                # Pure Pursuit calculation
                target_location = target_waypoint.transform.location
                dx = target_location.x - vehicle_location.x
                dy = target_location.y - vehicle_location.y
                
                # Transform to vehicle coordinate system
                cos_yaw = math.cos(vehicle_yaw)
                sin_yaw = math.sin(vehicle_yaw)
                local_x = dx * cos_yaw + dy * sin_yaw
                local_y = -dx * sin_yaw + dy * cos_yaw
                
                # Calculate curvature
                curvature = 2 * local_y / (lookahead_distance ** 2)
                
                # Convert to steering angle
                steering_angle = math.atan(curvature * 2.5)  # 2.5m wheelbase
                steer = steering_angle / (math.pi / 4)  # Normalize
                
                # Speed control
                velocity = vehicle.get_velocity()
                current_speed = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2)
                target_speed = 24.0
                
                # Reduce speed in turns
                if abs(curvature) > 0.1:
                    target_speed *= 0.75
                    
                if current_speed < target_speed:
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
            print(f"Pure Pursuit error: {e}")
            
        return AdvancedAutopilot.stanley_control(vehicle, world)
    
    @staticmethod
    def stanley_control(vehicle, world):
        """Stanley Control Algorithm"""
        try:
            # Get vehicle state
            vehicle_transform = vehicle.get_transform()
            vehicle_location = vehicle_transform.location
            vehicle_yaw = math.radians(vehicle_transform.rotation.yaw)
            
            velocity = vehicle.get_velocity()
            current_speed = math.sqrt(velocity.x**2 + velocity.y**2)  # m/s
            
            # Get nearest waypoint
            map_obj = world.get_map()
            current_waypoint = map_obj.get_waypoint(vehicle_location)
            
            if current_waypoint:
                # Calculate cross track error
                waypoint_location = current_waypoint.transform.location
                waypoint_yaw = math.radians(current_waypoint.transform.rotation.yaw)
                
                # Vector from vehicle to waypoint
                dx = waypoint_location.x - vehicle_location.x
                dy = waypoint_location.y - vehicle_location.y
                
                # Cross track error (perpendicular distance to path)
                cross_track_error = -math.sin(waypoint_yaw) * dx + math.cos(waypoint_yaw) * dy
                
                # Heading error
                heading_error = waypoint_yaw - vehicle_yaw
                
                # Normalize heading error
                while heading_error > math.pi:
                    heading_error -= 2 * math.pi
                while heading_error < -math.pi:
                    heading_error += 2 * math.pi
                
                # Stanley control law
                k_e = 0.4  # Cross track error gain
                k_soft = 1.0  # Softening parameter
                
                cross_track_steer = math.atan2(k_e * cross_track_error, k_soft + current_speed)
                steer = heading_error + cross_track_steer
                
                # Speed control
                target_speed = 22.0  # km/h
                speed_kmh = current_speed * 3.6
                
                # Adjust speed based on steering
                if abs(steer) > 0.2:
                    target_speed *= 0.8
                
                if speed_kmh < target_speed:
                    throttle = 0.4
                    brake = 0.0
                else:
                    throttle = 0.1
                    brake = 0.0
                
                control = carla.VehicleControl()
                control.steer = np.clip(steer, -0.35, 0.35)
                control.throttle = throttle
                control.brake = brake
                
                return control
                
        except Exception as e:
            print(f"Stanley control error: {e}")
            
        return AdvancedAutopilot.basic_waypoint_follow(vehicle, world)
    
    @staticmethod
    def basic_waypoint_follow(vehicle, world):
        """Basic waypoint following (fallback)"""
        try:
            # Simple waypoint following
            map_obj = world.get_map()
            current_waypoint = map_obj.get_waypoint(vehicle.get_location())
            
            if current_waypoint:
                next_waypoints = current_waypoint.next(5.0)
                if next_waypoints:
                    target_waypoint = next_waypoints[0]
                    
                    # Calculate steering
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
                        
                    steer = angle / math.pi * 0.3
                    
                    # Speed control
                    velocity = vehicle.get_velocity()
                    current_speed = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2)
                    target_speed = 25.0
                    
                    if current_speed < target_speed:
                        throttle = 0.4
                        brake = 0.0
                    else:
                        throttle = 0.1
                        brake = 0.0
                        
                    control = carla.VehicleControl()
                    control.steer = np.clip(steer, -0.3, 0.3)
                    control.throttle = throttle
                    control.brake = brake
                    
                    return control
        except:
            pass
                    
        # Final fallback
        control = carla.VehicleControl()
        control.throttle = 0.3
        control.steer = 0.0
        return control

def get_advanced_control(vehicle, world, algorithm='pure_pursuit'):
    """Get advanced autopilot control"""
    
    if algorithm == 'pure_pursuit':
        return AdvancedAutopilot.pure_pursuit_control(vehicle, world)
    elif algorithm == 'stanley':
        return AdvancedAutopilot.stanley_control(vehicle, world)
    elif algorithm == 'basic':
        return AdvancedAutopilot.basic_waypoint_follow(vehicle, world)
    else:
        return AdvancedAutopilot.pure_pursuit_control(vehicle, world)