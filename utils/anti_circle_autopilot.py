# -*- coding: utf-8 -*-
"""
Anti-Circle Junction Navigation for CARLA
Simple and reliable autopilot that never gets stuck in junctions
"""

import carla
import numpy as np
import math
import time
from collections import deque

class AntiCircleAutopilot:
    """Anti-circle autopilot that never gets stuck at junctions"""
    
    def __init__(self, vehicle, world):
        self.vehicle = vehicle
        self.world = world
        self.map = world.get_map()
        
        # Speed parameters
        self.target_speed = 30.0  # km/h
        self.junction_speed = 15.0  # km/h
        
        # Anti-circle system
        self.in_junction = False
        self.junction_entry_time = None
        self.junction_timeout = 8.0  # seconds
        self.location_history = deque(maxlen=15)
        self.stuck_counter = 0
        self.escape_mode = False
        self.escape_direction = None
        
        print("Anti-Circle Autopilot initialized")
        
    def is_stuck_in_circle(self):
        """Detect if vehicle is stuck in a circular motion"""
        if len(self.location_history) < 10:
            return False
            
        # Get recent positions
        recent_positions = list(self.location_history)[-10:]
        
        # Calculate center point
        center_x = sum(pos.x for pos in recent_positions) / len(recent_positions)
        center_y = sum(pos.y for pos in recent_positions) / len(recent_positions)
        
        # Check if all positions are close to center (circular motion)
        distances = []
        for pos in recent_positions:
            dist = math.sqrt((pos.x - center_x)**2 + (pos.y - center_y)**2)
            distances.append(dist)
            
        avg_distance = sum(distances) / len(distances)
        max_distance = max(distances)
        
        # If average distance is small and positions are clustered, we're circling
        return avg_distance < 4.0 and max_distance < 8.0
        
    def get_escape_waypoint(self, current_waypoint):
        """Get waypoint to escape junction circle"""
        if not current_waypoint:
            return None
            
        # Get all possible next waypoints
        next_waypoints = current_waypoint.next(4.0)
        if not next_waypoints:
            return current_waypoint
            
        # If we already have an escape direction, stick to it
        if self.escape_direction:
            return self.escape_direction
            
        # Choose the straightest path
        vehicle_yaw = self.vehicle.get_transform().rotation.yaw
        
        best_waypoint = next_waypoints[0]
        min_angle_diff = float('inf')
        
        for wp in next_waypoints:
            wp_yaw = wp.transform.rotation.yaw
            angle_diff = abs(self.normalize_angle(wp_yaw - vehicle_yaw))
            
            if angle_diff < min_angle_diff:
                min_angle_diff = angle_diff
                best_waypoint = wp
                
        self.escape_direction = best_waypoint
        print("ESCAPE: Forcing direction to exit junction")
        return best_waypoint
        
    def normalize_angle(self, angle):
        """Normalize angle to [-180, 180] range"""
        while angle > 180:
            angle -= 360
        while angle < -180:
            angle += 360
        return angle
        
    def get_target_waypoint(self, current_waypoint):
        """Get target waypoint with intelligent junction handling"""
        if not current_waypoint:
            return None
            
        # Normal waypoint selection
        next_waypoints = current_waypoint.next(3.0)
        if not next_waypoints:
            return current_waypoint
            
        if len(next_waypoints) == 1:
            return next_waypoints[0]
            
        # Multiple options - prefer straight path
        vehicle_yaw = self.vehicle.get_transform().rotation.yaw
        
        best_waypoint = next_waypoints[0]
        min_angle_diff = float('inf')
        
        for wp in next_waypoints:
            wp_yaw = wp.transform.rotation.yaw
            angle_diff = abs(self.normalize_angle(wp_yaw - vehicle_yaw))
            
            if angle_diff < min_angle_diff:
                min_angle_diff = angle_diff
                best_waypoint = wp
                
        return best_waypoint
        
    def compute_control(self):
        """Main control computation with anti-circle logic"""
        current_location = self.vehicle.get_location()
        self.location_history.append(current_location)
        
        # Get current waypoint
        current_waypoint = self.map.get_waypoint(current_location)
        if not current_waypoint:
            return self.emergency_stop()
            
        # Track junction state
        was_in_junction = self.in_junction
        self.in_junction = current_waypoint.is_junction
        
        # Junction entry
        if self.in_junction and not was_in_junction:
            self.junction_entry_time = time.time()
            self.escape_mode = False
            self.escape_direction = None
            print("Entering junction")
            
        # Junction exit
        elif not self.in_junction and was_in_junction:
            self.junction_entry_time = None
            self.escape_mode = False
            self.escape_direction = None
            self.stuck_counter = 0
            print("Successfully exited junction")
            
        # Anti-circle logic for junctions
        if self.in_junction:
            # Check for timeout
            if (self.junction_entry_time and 
                time.time() - self.junction_entry_time > self.junction_timeout):
                print("Junction timeout - activating escape mode")
                self.escape_mode = True
                
            # Check for circular motion
            elif self.is_stuck_in_circle():
                self.stuck_counter += 1
                if self.stuck_counter > 3:
                    print("Circle detected - activating escape mode")
                    self.escape_mode = True
                    
            # Get waypoint based on mode
            if self.escape_mode:
                target_waypoint = self.get_escape_waypoint(current_waypoint)
                target_speed = min(self.junction_speed * 1.5, 25.0)  # Faster escape
            else:
                target_waypoint = self.get_target_waypoint(current_waypoint)
                target_speed = self.junction_speed
        else:
            # Normal road driving
            target_waypoint = self.get_target_waypoint(current_waypoint)
            target_speed = self.target_speed
            
        if not target_waypoint:
            return self.emergency_stop()
            
        return self.compute_vehicle_control(target_waypoint, target_speed)
        
    def compute_vehicle_control(self, target_waypoint, target_speed):
        """Compute vehicle control commands"""
        # Steering control
        steering = self.compute_steering(target_waypoint)
        
        # Speed control
        throttle, brake = self.compute_speed_control(target_speed)
        
        # Create control
        control = carla.VehicleControl()
        control.steer = np.clip(steering, -0.6, 0.6)  # Limited steering
        control.throttle = np.clip(throttle, 0.0, 0.7)  # Limited throttle
        control.brake = np.clip(brake, 0.0, 1.0)
        
        return control
        
    def compute_steering(self, target_waypoint):
        """Compute steering using pure pursuit"""
        vehicle_transform = self.vehicle.get_transform()
        target_location = target_waypoint.transform.location
        
        # Vector from vehicle to target
        dx = target_location.x - vehicle_transform.location.x
        dy = target_location.y - vehicle_transform.location.y
        
        # Vehicle heading
        yaw = math.radians(vehicle_transform.rotation.yaw)
        
        # Transform to vehicle coordinate frame
        local_x = dx * math.cos(yaw) + dy * math.sin(yaw)
        local_y = -dx * math.sin(yaw) + dy * math.cos(yaw)
        
        # Pure pursuit algorithm
        lookahead_distance = math.sqrt(local_x**2 + local_y**2)
        if lookahead_distance < 0.1:
            return 0.0
            
        curvature = 2.0 * local_y / (lookahead_distance**2)
        
        # Convert to steering angle
        wheelbase = 2.7  # Typical car wheelbase
        steering_angle = math.atan(curvature * wheelbase)
        
        # Normalize to [-1, 1]
        max_steering_angle = math.radians(25)  # 25 degrees max
        return steering_angle / max_steering_angle
        
    def compute_speed_control(self, target_speed_kmh):
        """Compute throttle and brake using simple PID"""
        # Current speed
        velocity = self.vehicle.get_velocity()
        current_speed_ms = math.sqrt(velocity.x**2 + velocity.y**2)
        current_speed_kmh = current_speed_ms * 3.6
        
        # Speed error
        speed_error = target_speed_kmh - current_speed_kmh
        
        # Simple proportional control
        if speed_error > 3.0:
            throttle = min(0.6, speed_error * 0.05)
            brake = 0.0
        elif speed_error < -3.0:
            throttle = 0.0
            brake = min(0.6, abs(speed_error) * 0.08)
        else:
            throttle = 0.25  # Maintain speed
            brake = 0.0
            
        return throttle, brake
        
    def emergency_stop(self):
        """Emergency stop control"""
        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 0.0
        control.brake = 1.0
        return control

# Main interface
def get_anti_circle_control(vehicle, world):
    """Get anti-circle autopilot control"""
    if not hasattr(vehicle, 'anti_circle_autopilot'):
        vehicle.anti_circle_autopilot = AntiCircleAutopilot(vehicle, world)
        
    return vehicle.anti_circle_autopilot.compute_control()