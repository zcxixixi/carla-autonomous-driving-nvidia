# -*- coding: utf-8 -*-

import carla
import numpy as np
import math
import random

class DrivingModes:
    """Different driving modes for CARLA vehicles"""
    
    @staticmethod
    def random_drive():
        """Random driving (original mode)"""
        control = carla.VehicleControl()
        control.throttle = np.random.uniform(0.3, 0.6)
        control.steer = np.random.uniform(-0.3, 0.3)
        return control
        
    @staticmethod
    def cruise_drive(vehicle, world, target_speed=25.0):
        """Cruise driving - follow roads"""
        try:
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
                        
                    steer = angle / math.pi * 0.5
                    
                    # Speed control
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
        except:
            pass
                    
        # Fallback to random driving
        return DrivingModes.random_drive()
        
    @staticmethod
    def exploration_drive(vehicle, world):
        """Exploration driving - change direction periodically"""
        try:
            map_obj = world.get_map()
            current_waypoint = map_obj.get_waypoint(vehicle.get_location())
            
            if current_waypoint and random.random() > 0.95:  # 5% chance to change direction
                # Random path selection
                next_waypoints = current_waypoint.next(10.0)
                if len(next_waypoints) > 1:
                    target_waypoint = random.choice(next_waypoints)
                elif next_waypoints:
                    target_waypoint = next_waypoints[0]
                else:
                    return DrivingModes.random_drive()
                    
                # Calculate control towards target
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
        except:
            pass
                
        return DrivingModes.cruise_drive(vehicle, world)
        
    @staticmethod
    def smooth_drive(vehicle, world):
        """Ultra smooth driving with conservative turning and stable straight-line driving"""
        try:
            # Get current velocity and position
            velocity = vehicle.get_velocity()
            current_speed = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2)
            
            # Use waypoints for direction
            map_obj = world.get_map()
            current_waypoint = map_obj.get_waypoint(vehicle.get_location())
            
            if current_waypoint:
                # Look ahead further for smoother planning
                next_waypoints = current_waypoint.next(12.0)  # Increased from 8m to 12m
                
                if next_waypoints:
                    target_waypoint = next_waypoints[0]
                    
                    # Calculate steering angle
                    vehicle_transform = vehicle.get_transform()
                    target_location = target_waypoint.transform.location
                    vehicle_location = vehicle_transform.location
                    
                    target_vector = target_location - vehicle_location
                    forward_vector = vehicle_transform.get_forward_vector()
                    
                    # Calculate angle difference
                    angle = math.atan2(target_vector.y, target_vector.x) - math.atan2(forward_vector.y, forward_vector.x)
                    
                    # Normalize angle
                    while angle > math.pi:
                        angle -= 2 * math.pi
                    while angle < -math.pi:
                        angle += 2 * math.pi
                    
                    # Much more conservative steering with dead zone for straight driving
                    abs_angle = abs(angle)
                    
                    # Dead zone for straight driving - ignore very small angles
                    if abs_angle < 0.05:  # ~3 degrees dead zone
                        steer = 0.0
                    else:
                        # Reduced steering sensitivity and maximum angle
                        steer = angle / math.pi * 0.15  # Reduced from 0.3 to 0.15
                    
                    # Determine if we're turning (for speed control)
                    is_turning = abs_angle > 0.1  # ~6 degrees threshold
                    is_sharp_turn = abs_angle > 0.3  # ~17 degrees threshold
                    
                    # Speed control with turning consideration
                    if is_sharp_turn:
                        # Sharp turn - significant speed reduction
                        target_speed = 15.0  # km/h
                    elif is_turning:
                        # Gentle turn - moderate speed reduction  
                        target_speed = 22.0  # km/h
                    else:
                        # Straight driving - normal speed
                        target_speed = 28.0  # km/h
                    
                    # Smooth speed control
                    speed_diff = target_speed - current_speed
                    
                    if speed_diff > 5.0:
                        throttle = 0.4  # Gentle acceleration
                        brake = 0.0
                    elif speed_diff > 2.0:
                        throttle = 0.25  # Light acceleration
                        brake = 0.0
                    elif speed_diff < -5.0:
                        throttle = 0.0
                        brake = 0.2  # Gentle braking
                    elif speed_diff < -2.0:
                        throttle = 0.0
                        brake = 0.1  # Light braking
                    else:
                        # Maintain speed
                        throttle = 0.15  # Minimal throttle to maintain speed
                        brake = 0.0
                    
                    # Apply even more conservative steering limits
                    steer = np.clip(steer, -0.15, 0.15)  # Max ¡À15 degrees
                    
                    control = carla.VehicleControl()
                    control.steer = steer
                    control.throttle = throttle
                    control.brake = brake
                    
                    return control
        except:
            pass
        
        # Fallback
        return DrivingModes.cruise_drive(vehicle, world)
        
    @staticmethod
    def stable_drive(vehicle, world):
        """Ultra-stable driving with minimal steering and progressive speed control"""
        try:
            # Get current velocity
            velocity = vehicle.get_velocity()
            current_speed = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2)
            
            # Use waypoints for direction
            map_obj = world.get_map()
            current_waypoint = map_obj.get_waypoint(vehicle.get_location())
            
            if current_waypoint:
                # Look very far ahead for ultra-smooth planning
                next_waypoints = current_waypoint.next(15.0)  # 15 meters ahead
                
                if next_waypoints:
                    target_waypoint = next_waypoints[0]
                    
                    # Calculate steering angle
                    vehicle_transform = vehicle.get_transform()
                    target_location = target_waypoint.transform.location
                    vehicle_location = vehicle_transform.location
                    
                    target_vector = target_location - vehicle_location
                    forward_vector = vehicle_transform.get_forward_vector()
                    
                    # Calculate angle difference
                    angle = math.atan2(target_vector.y, target_vector.x) - math.atan2(forward_vector.y, forward_vector.x)
                    
                    # Normalize angle
                    while angle > math.pi:
                        angle -= 2 * math.pi
                    while angle < -math.pi:
                        angle += 2 * math.pi
                    
                    abs_angle = abs(angle)
                    
                    # Larger dead zone for straight driving
                    if abs_angle < 0.08:  # ~4.5 degrees dead zone
                        steer = 0.0
                    else:
                        # Ultra-conservative steering
                        steer = angle / math.pi * 0.08  # Very small steering coefficient
                    
                    # Progressive speed control based on curvature
                    if abs_angle > 0.4:  # Sharp turn (~23 degrees)
                        target_speed = 12.0  # Very slow for sharp turns
                    elif abs_angle > 0.2:  # Medium turn (~11 degrees)  
                        target_speed = 18.0  # Moderate speed
                    elif abs_angle > 0.1:  # Gentle turn (~6 degrees)
                        target_speed = 24.0  # Slightly reduced speed
                    else:
                        target_speed = 26.0  # Normal straight-line speed
                    
                    # Very smooth throttle/brake control
                    speed_error = target_speed - current_speed
                    
                    if speed_error > 8.0:
                        throttle = 0.3
                        brake = 0.0
                    elif speed_error > 3.0:
                        throttle = 0.2
                        brake = 0.0
                    elif speed_error > 1.0:
                        throttle = 0.1
                        brake = 0.0
                    elif speed_error < -8.0:
                        throttle = 0.0
                        brake = 0.15
                    elif speed_error < -3.0:
                        throttle = 0.0
                        brake = 0.08
                    elif speed_error < -1.0:
                        throttle = 0.0
                        brake = 0.03
                    else:
                        # Maintain current speed
                        throttle = 0.05
                        brake = 0.0
                    
                    # Ultra-conservative steering limits
                    steer = np.clip(steer, -0.1, 0.1)  # Max ¡À10 degrees
                    
                    control = carla.VehicleControl()
                    control.steer = steer
                    control.throttle = throttle
                    control.brake = brake
                    
                    return control
        except:
            pass
        
        # Fallback to smooth drive
        return DrivingModes.smooth_drive(vehicle, world)