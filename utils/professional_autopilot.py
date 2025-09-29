# -*- coding: utf-8 -*-
"""
Professional Autonomous Driving System
Based on CARLA official examples and industry best practices
"""

import carla
import numpy as np
import math
import random
from enum import Enum
from collections import deque

class DrivingBehavior(Enum):
    AGGRESSIVE = "aggressive"
    NORMAL = "normal"
    CAUTIOUS = "cautious"

class VehicleState(Enum):
    DRIVING = "driving"
    STOPPED = "stopped"
    EMERGENCY = "emergency"

class ProfessionalAutopilot:
    """Professional autonomous driving system with junction intelligence"""
    
    def __init__(self, vehicle, world, behavior=DrivingBehavior.NORMAL):
        self.vehicle = vehicle
        self.world = world
        self.map = world.get_map()
        self.behavior = behavior
        
        # Waypoint management
        self.waypoint_buffer = deque(maxlen=10)
        self.current_waypoint = None
        self.target_waypoint = None
        
        # State management
        self.state = VehicleState.DRIVING
        self.last_junction_id = None
        self.junction_entry_location = None
        
        # Behavior parameters
        self._setup_behavior_parameters()
        
        # Control history for smoothing
        self.steering_history = deque(maxlen=5)
        self.throttle_history = deque(maxlen=5)
        
        # Initialize waypoints
        self._initialize_waypoints()
        
        print(f"Professional Autopilot initialized - Behavior: {behavior.value}")
        
    def _setup_behavior_parameters(self):
        """Setup parameters based on driving behavior"""
        if self.behavior == DrivingBehavior.AGGRESSIVE:
            self.target_speed = 50.0  # km/h
            self.junction_speed = 25.0
            self.follow_distance = 5.0
            self.lane_change_threshold = 0.3
        elif self.behavior == DrivingBehavior.CAUTIOUS:
            self.target_speed = 25.0  # km/h
            self.junction_speed = 10.0
            self.follow_distance = 15.0
            self.lane_change_threshold = 0.8
        else:  # NORMAL
            self.target_speed = 35.0  # km/h
            self.junction_speed = 15.0
            self.follow_distance = 10.0
            self.lane_change_threshold = 0.5
            
    def _initialize_waypoints(self):
        """Initialize waypoint buffer"""
        current_location = self.vehicle.get_location()
        self.current_waypoint = self.map.get_waypoint(current_location)
        
        if self.current_waypoint:
            self._populate_waypoint_buffer()
            
    def _populate_waypoint_buffer(self):
        """Populate waypoint buffer with upcoming waypoints"""
        self.waypoint_buffer.clear()
        
        waypoint = self.current_waypoint
        for i in range(10):
            if not waypoint:
                break
                
            self.waypoint_buffer.append(waypoint)
            
            # Get next waypoint with intelligent junction handling
            next_waypoints = waypoint.next(3.0)
            if next_waypoints:
                # Choose best next waypoint
                waypoint = self._choose_best_waypoint(waypoint, next_waypoints)
            else:
                break
                
    def _choose_best_waypoint(self, current_waypoint, next_waypoints):
        """Intelligent waypoint selection for junctions"""
        if len(next_waypoints) == 1:
            return next_waypoints[0]
            
        # If multiple options (junction), make intelligent decision
        if current_waypoint.is_junction:
            return self._handle_junction_choice(current_waypoint, next_waypoints)
        else:
            # Prefer straight path when not in junction
            return self._prefer_straight_path(current_waypoint, next_waypoints)
            
    def _handle_junction_choice(self, current_waypoint, next_waypoints):
        """Handle junction waypoint selection"""
        vehicle_location = self.vehicle.get_location()
        
        # Strategy 1: Continue in same general direction
        current_direction = current_waypoint.transform.rotation.yaw
        
        best_waypoint = None
        min_angle_diff = float('inf')
        
        for wp in next_waypoints:
            # Calculate angle difference
            wp_direction = wp.transform.rotation.yaw
            angle_diff = abs(self._normalize_angle(wp_direction - current_direction))
            
            # Prefer smaller angle changes (continue straight when possible)
            if angle_diff < min_angle_diff:
                min_angle_diff = angle_diff
                best_waypoint = wp
                
        # Strategy 2: If no clear straight path, prefer right turns (traffic rules)
        if min_angle_diff > 45:  # No clear straight path
            for wp in next_waypoints:
                wp_direction = wp.transform.rotation.yaw
                turn_angle = self._normalize_angle(wp_direction - current_direction)
                
                # Prefer right turns (-90 to 0 degrees)
                if -90 <= turn_angle <= 0:
                    best_waypoint = wp
                    break
                    
        return best_waypoint if best_waypoint else next_waypoints[0]
        
    def _prefer_straight_path(self, current_waypoint, next_waypoints):
        """Prefer straight path when multiple options available"""
        current_direction = current_waypoint.transform.rotation.yaw
        
        best_waypoint = next_waypoints[0]
        min_angle_diff = float('inf')
        
        for wp in next_waypoints:
            wp_direction = wp.transform.rotation.yaw
            angle_diff = abs(self._normalize_angle(wp_direction - current_direction))
            
            if angle_diff < min_angle_diff:
                min_angle_diff = angle_diff
                best_waypoint = wp
                
        return best_waypoint
        
    def _normalize_angle(self, angle):
        """Normalize angle to [-180, 180] range"""
        while angle > 180:
            angle -= 360
        while angle < -180:
            angle += 360
        return angle
        
    def update_waypoints(self):
        """Update waypoint buffer based on vehicle position"""
        vehicle_location = self.vehicle.get_location()
        current_waypoint = self.map.get_waypoint(vehicle_location)
        
        if not current_waypoint:
            return
            
        # Check if we need to update waypoints
        if (not self.current_waypoint or 
            current_waypoint.transform.location.distance(
                self.current_waypoint.transform.location) > 5.0):
            
            self.current_waypoint = current_waypoint
            self._populate_waypoint_buffer()
            
    def get_target_waypoint(self):
        """Get appropriate target waypoint"""
        if len(self.waypoint_buffer) < 2:
            self.update_waypoints()
            
        if len(self.waypoint_buffer) >= 2:
            # Look ahead based on speed
            velocity = self.vehicle.get_velocity()
            speed_ms = math.sqrt(velocity.x**2 + velocity.y**2)
            
            # Dynamic lookahead distance
            if speed_ms < 5.0:  # Low speed
                lookahead_index = 1
            elif speed_ms < 15.0:  # Medium speed
                lookahead_index = min(3, len(self.waypoint_buffer) - 1)
            else:  # High speed
                lookahead_index = min(5, len(self.waypoint_buffer) - 1)
                
            return self.waypoint_buffer[lookahead_index]
        else:
            return self.current_waypoint
            
    def check_traffic_lights(self):
        """Check for traffic light states"""
        if not self.current_waypoint:
            return carla.TrafficLightState.Green
            
        # Check if approaching traffic light
        if self.current_waypoint.is_junction:
            vehicle_location = self.vehicle.get_location()
            
            # Get all traffic lights
            traffic_lights = self.world.get_actors().filter('traffic.traffic_light*')
            
            for tl in traffic_lights:
                # Check if traffic light is relevant to current position
                distance = tl.get_location().distance(vehicle_location)
                if distance < 20.0:
                    # Check if traffic light affects current lane
                    tl_waypoint = self.map.get_waypoint(tl.get_location())
                    if (tl_waypoint and 
                        tl_waypoint.road_id == self.current_waypoint.road_id):
                        return tl.get_state()
                        
        return carla.TrafficLightState.Green
        
    def check_obstacles(self):
        """Check for obstacles and other vehicles"""
        vehicle_location = self.vehicle.get_location()
        vehicle_forward = self.vehicle.get_transform().get_forward_vector()
        
        # Get all vehicles
        vehicles = self.world.get_actors().filter('vehicle.*')
        
        min_distance = float('inf')
        obstacle_detected = False
        
        for other_vehicle in vehicles:
            if other_vehicle.id == self.vehicle.id:
                continue
                
            other_location = other_vehicle.get_location()
            distance = vehicle_location.distance(other_location)
            
            # Check if vehicle is in front
            if distance < 30.0:  # Within detection range
                # Vector to other vehicle
                to_other = other_location - vehicle_location
                
                # Check if in front (dot product > 0)
                dot_product = (to_other.x * vehicle_forward.x + 
                              to_other.y * vehicle_forward.y)
                
                if dot_product > 0 and distance < min_distance:
                    min_distance = distance
                    obstacle_detected = True
                    
        return obstacle_detected, min_distance
        
    def compute_control(self):
        """Main control computation"""
        # Update waypoints
        self.update_waypoints()
        
        # Get target waypoint
        target_waypoint = self.get_target_waypoint()
        if not target_waypoint:
            return self._emergency_stop()
            
        # Check traffic conditions
        traffic_state = self.check_traffic_lights()
        obstacle_detected, obstacle_distance = self.check_obstacles()
        
        # Determine target speed based on conditions
        target_speed = self._compute_target_speed(
            target_waypoint, traffic_state, obstacle_detected, obstacle_distance)
            
        # Compute control commands
        steering = self._compute_steering(target_waypoint)
        throttle, brake = self._compute_speed_control(target_speed)
        
        # Apply control smoothing
        steering = self._smooth_steering(steering)
        throttle = self._smooth_throttle(throttle)
        
        # Create control
        control = carla.VehicleControl()
        control.steer = np.clip(steering, -1.0, 1.0)
        control.throttle = np.clip(throttle, 0.0, 1.0)
        control.brake = np.clip(brake, 0.0, 1.0)
        
        return control
        
    def _compute_target_speed(self, target_waypoint, traffic_state, 
                             obstacle_detected, obstacle_distance):
        """Compute target speed based on conditions"""
        base_speed = self.target_speed
        
        # Adjust for junction
        if target_waypoint and target_waypoint.is_junction:
            base_speed = self.junction_speed
            
        # Adjust for traffic lights
        if traffic_state == carla.TrafficLightState.Red:
            return 0.0
        elif traffic_state == carla.TrafficLightState.Yellow:
            return min(base_speed, 15.0)
            
        # Adjust for obstacles
        if obstacle_detected:
            if obstacle_distance < self.follow_distance:
                return 0.0  # Stop if too close
            elif obstacle_distance < self.follow_distance * 2:
                return base_speed * 0.5  # Slow down
                
        return base_speed
        
    def _compute_steering(self, target_waypoint):
        """Compute steering using enhanced pure pursuit"""
        if not target_waypoint:
            return 0.0
            
        vehicle_transform = self.vehicle.get_transform()
        target_location = target_waypoint.transform.location
        
        # Pure pursuit algorithm
        dx = target_location.x - vehicle_transform.location.x
        dy = target_location.y - vehicle_transform.location.y
        
        # Vehicle heading
        yaw = math.radians(vehicle_transform.rotation.yaw)
        
        # Transform to vehicle coordinate frame
        local_x = dx * math.cos(yaw) + dy * math.sin(yaw)
        local_y = -dx * math.sin(yaw) + dy * math.cos(yaw)
        
        # Pure pursuit control
        lookahead_distance = math.sqrt(local_x**2 + local_y**2)
        if lookahead_distance < 0.1:
            return 0.0
            
        curvature = 2.0 * local_y / (lookahead_distance**2)
        
        # Convert to steering angle
        wheelbase = 2.7  # Vehicle wheelbase
        steering_angle = math.atan(curvature * wheelbase)
        
        # Normalize
        max_steering_angle = math.radians(30)
        steering = steering_angle / max_steering_angle
        
        return steering
        
    def _compute_speed_control(self, target_speed_kmh):
        """Compute speed control using PID"""
        # Current speed
        velocity = self.vehicle.get_velocity()
        current_speed_ms = math.sqrt(velocity.x**2 + velocity.y**2)
        current_speed_kmh = current_speed_ms * 3.6
        
        # Speed error
        speed_error = target_speed_kmh - current_speed_kmh
        
        # PID parameters (tuned for different behaviors)
        if self.behavior == DrivingBehavior.AGGRESSIVE:
            kp = 0.4
        elif self.behavior == DrivingBehavior.CAUTIOUS:
            kp = 0.15
        else:
            kp = 0.25
            
        if speed_error > 1.0:
            throttle = min(1.0, kp * speed_error / 10.0)
            brake = 0.0
        elif speed_error < -1.0:
            throttle = 0.0
            brake = min(1.0, abs(speed_error) * kp / 15.0)
        else:
            throttle = 0.1  # Maintain speed
            brake = 0.0
            
        return throttle, brake
        
    def _smooth_steering(self, steering):
        """Apply steering smoothing"""
        self.steering_history.append(steering)
        if len(self.steering_history) > 1:
            # Simple moving average
            return sum(self.steering_history) / len(self.steering_history)
        return steering
        
    def _smooth_throttle(self, throttle):
        """Apply throttle smoothing"""
        self.throttle_history.append(throttle)
        if len(self.throttle_history) > 1:
            return sum(self.throttle_history) / len(self.throttle_history)
        return throttle
        
    def _emergency_stop(self):
        """Emergency stop"""
        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 0.0
        control.brake = 1.0
        return control

# Main interface functions
def get_professional_autopilot_control(vehicle, world, behavior="normal"):
    """Get professional autopilot control"""
    if not hasattr(vehicle, 'professional_autopilot'):
        behavior_enum = {
            "aggressive": DrivingBehavior.AGGRESSIVE,
            "normal": DrivingBehavior.NORMAL,
            "cautious": DrivingBehavior.CAUTIOUS
        }.get(behavior, DrivingBehavior.NORMAL)
        
        vehicle.professional_autopilot = ProfessionalAutopilot(
            vehicle, world, behavior_enum)
        
    return vehicle.professional_autopilot.compute_control()

def get_carla_native_autopilot_control(vehicle):
    """Use CARLA's built-in autopilot as reference"""
    # Enable CARLA's native autopilot
    vehicle.set_autopilot(True)
    
    # Return empty control (autopilot handles everything)
    return carla.VehicleControl()