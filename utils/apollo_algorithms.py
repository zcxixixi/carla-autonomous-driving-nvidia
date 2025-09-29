# -*- coding: utf-8 -*-

import carla
import numpy as np
import math
from collections import deque
from enum import Enum
import time

class DrivingState(Enum):
    """Apollo-style driving states"""
    FOLLOW_LANE = "follow_lane"
    CHANGE_LANE = "change_lane"  
    STOP = "stop"
    EMERGENCY_STOP = "emergency_stop"
    PARK = "park"

class ApolloStylePlanner:
    """Apollo-inspired path planning and control system"""
    
    def __init__(self, vehicle, world):
        self.vehicle = vehicle
        self.world = world
        self.map = world.get_map()
        
        # Planning parameters (Apollo-style)
        self.planning_horizon = 50.0  # meters
        self.planning_resolution = 1.0  # meters
        self.max_speed = 25.0  # km/h (reduced for safety)
        self.comfort_decel = -2.0  # m/s^2
        self.emergency_decel = -4.0  # m/s^2
        
        # State management
        self.current_state = DrivingState.FOLLOW_LANE
        self.target_waypoints = deque(maxlen=100)
        self.reference_line = []
        
        # Control parameters (Apollo PID-like)
        self.lateral_controller = LateralController()
        self.longitudinal_controller = LongitudinalController()
        
        # Safety initialization
        self.initialization_frames = 0
        self.is_initialized = False
        self.initial_location = vehicle.get_location()
        
        print("Apollo-style Planner initialized")
        
    def generate_reference_line(self):
        """Generate Apollo-style reference line"""
        current_waypoint = self.map.get_waypoint(self.vehicle.get_location())
        
        if not current_waypoint:
            return
            
        self.reference_line = []
        waypoint = current_waypoint
        
        # Generate smooth reference line
        for i in range(int(self.planning_horizon / self.planning_resolution)):
            if waypoint:
                self.reference_line.append({
                    'waypoint': waypoint,
                    'speed_limit': 30.0,  # km/h
                    'curvature': self.calculate_curvature(waypoint),
                    's': i * self.planning_resolution  # frenet s coordinate
                })
                
                next_waypoints = waypoint.next(self.planning_resolution)
                if next_waypoints:
                    waypoint = next_waypoints[0]
                else:
                    break
                    
    def calculate_curvature(self, waypoint):
        """Calculate road curvature (Apollo method)"""
        if not waypoint:
            return 0.0
            
        # Get neighboring waypoints
        prev_wps = waypoint.previous(2.0)
        next_wps = waypoint.next(2.0)
        
        if not prev_wps or not next_wps:
            return 0.0
            
        prev_wp = prev_wps[0]
        next_wp = next_wps[0]
        
        # Calculate curvature using three points
        p1 = prev_wp.transform.location
        p2 = waypoint.transform.location  
        p3 = next_wp.transform.location
        
        # Curvature = 1/R where R is radius
        a = math.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2)
        b = math.sqrt((p3.x - p2.x)**2 + (p3.y - p2.y)**2)
        c = math.sqrt((p3.x - p1.x)**2 + (p3.y - p1.y)**2)
        
        if a == 0 or b == 0 or c == 0:
            return 0.0
            
        # Area using Heron's formula
        s = (a + b + c) / 2
        area = math.sqrt(max(0, s * (s - a) * (s - b) * (s - c)))
        
        # Curvature = 4 * Area / (a * b * c)
        if area == 0:
            return 0.0
            
        curvature = 4 * area / (a * b * c)
        return curvature
        
    def plan_trajectory(self):
        """Apollo-style trajectory planning"""
        if not self.reference_line:
            self.generate_reference_line()
            
        # State machine for different driving scenarios
        if self.current_state == DrivingState.FOLLOW_LANE:
            return self.plan_lane_follow()
        elif self.current_state == DrivingState.STOP:
            return self.plan_stop()
        else:
            return self.plan_lane_follow()
            
    def plan_lane_follow(self):
        """Lane following trajectory (Apollo EM Planner style)"""
        trajectory_points = []
        
        # Safety check for initialization
        if not self.is_initialized:
            self.initialization_frames += 1
            if self.initialization_frames < 10:  # Wait 10 frames
                # Return safe stop trajectory
                current_location = self.vehicle.get_location()
                current_rotation = self.vehicle.get_transform().rotation
                trajectory_points.append({
                    'location': current_location,
                    'rotation': current_rotation,
                    'speed': 0.0,
                    'curvature': 0.0
                })
                return trajectory_points
            else:
                self.is_initialized = True
                print("Apollo: Initialization complete, starting safe driving")
        
        for ref_point in self.reference_line[:15]:  # Reduced lookahead
            waypoint = ref_point['waypoint']
            curvature = ref_point['curvature']
            
            # Conservative speed planning (Apollo method)
            if curvature > 0.1:
                target_speed = max(10.0, self.max_speed * (1 - curvature * 6))
            elif curvature > 0.05:
                target_speed = max(15.0, self.max_speed * (1 - curvature * 4))
            else:
                target_speed = self.max_speed * 0.8  # Conservative max speed
                
            trajectory_points.append({
                'location': waypoint.transform.location,
                'rotation': waypoint.transform.rotation,
                'speed': target_speed,
                'curvature': curvature
            })
            
        return trajectory_points
        
    def plan_stop(self):
        """Emergency stop planning"""
        current_location = self.vehicle.get_location()
        current_rotation = self.vehicle.get_transform().rotation
        
        # Simple stop trajectory - current position with zero speed
        return [{
            'location': current_location,
            'rotation': current_rotation, 
            'speed': 0.0,
            'curvature': 0.0
        }]
        
    def execute_control(self, trajectory):
        """Execute Apollo-style vehicle control"""
        if not trajectory:
            return self.emergency_control()
            
        # Safety check during initialization
        if not self.is_initialized:
            control = carla.VehicleControl()
            control.steer = 0.0
            control.throttle = 0.0
            control.brake = 0.5  # Gentle brake during init
            return control
            
        target_point = trajectory[0]  # First trajectory point
        
        # Lateral control (Apollo LQR-like)
        steer = self.lateral_controller.compute_control(
            self.vehicle, target_point)
            
        # Longitudinal control (Apollo PID-like)
        throttle, brake = self.longitudinal_controller.compute_control(
            self.vehicle, target_point['speed'])
            
        control = carla.VehicleControl()
        control.steer = steer
        control.throttle = throttle
        control.brake = brake
        
        return control
        
    def emergency_control(self):
        """Emergency control when no valid trajectory"""
        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 0.0
        control.brake = 0.5
        return control

class LateralController:
    """Apollo-style lateral controller (LQR-inspired)"""
    
    def __init__(self):
        self.kp_heading = 0.6  # Reduced for gentler steering
        self.kp_lateral = 0.3  # Reduced for stability
        self.previous_error = 0.0
        
    def compute_control(self, vehicle, target_point):
        """Compute lateral control command"""
        vehicle_transform = vehicle.get_transform()
        vehicle_location = vehicle_transform.location
        vehicle_yaw = math.radians(vehicle_transform.rotation.yaw)
        
        # Target point
        target_location = target_point['location']
        target_yaw = math.radians(target_point['rotation'].yaw)
        
        # Lateral error (cross track error)
        dx = target_location.x - vehicle_location.x
        dy = target_location.y - vehicle_location.y
        
        # Transform to vehicle coordinate frame
        lateral_error = -dx * math.sin(vehicle_yaw) + dy * math.cos(vehicle_yaw)
        
        # Heading error
        heading_error = target_yaw - vehicle_yaw
        while heading_error > math.pi:
            heading_error -= 2 * math.pi
        while heading_error < -math.pi:
            heading_error += 2 * math.pi
            
        # Apollo-style control law
        steer_command = (self.kp_heading * heading_error + 
                        self.kp_lateral * lateral_error)
        
        # Limit steering angle (more conservative)
        steer_command = np.clip(steer_command, -0.3, 0.3)
        
        return steer_command

class LongitudinalController:
    """Apollo-style longitudinal controller (PID-inspired)"""
    
    def __init__(self):
        self.kp = 0.3
        self.ki = 0.02
        self.kd = 0.1
        self.integral_error = 0.0
        self.previous_error = 0.0
        
    def compute_control(self, vehicle, target_speed_kmh):
        """Compute throttle and brake commands"""
        # Current speed
        velocity = vehicle.get_velocity()
        current_speed_ms = math.sqrt(velocity.x**2 + velocity.y**2)
        current_speed_kmh = current_speed_ms * 3.6
        
        # Speed error
        speed_error = target_speed_kmh - current_speed_kmh
        
        # PID control
        self.integral_error += speed_error * 0.1  # dt = 0.1s
        derivative_error = (speed_error - self.previous_error) / 0.1
        
        # Control signal
        control_signal = (self.kp * speed_error + 
                         self.ki * self.integral_error +
                         self.kd * derivative_error)
        
        self.previous_error = speed_error
        
        # Convert to throttle/brake
        if control_signal > 0:
            throttle = min(0.6, control_signal / 10.0)
            brake = 0.0
        else:
            throttle = 0.0
            brake = min(0.4, -control_signal / 10.0)
            
        return throttle, brake

class ApolloStyleDriver:
    """Apollo-inspired autonomous driver"""
    
    @staticmethod
    def apollo_control(vehicle, world):
        """Main Apollo-style control function"""
        try:
            # Initialize planner if not exists
            if not hasattr(vehicle, 'apollo_planner'):
                vehicle.apollo_planner = ApolloStylePlanner(vehicle, world)
                
            planner = vehicle.apollo_planner
            
            # Planning phase
            trajectory = planner.plan_trajectory()
            
            # Control phase  
            control = planner.execute_control(trajectory)
            
            return control
            
        except Exception as e:
            print(f"Apollo control error: {e}")
            # Emergency fallback
            control = carla.VehicleControl()
            control.steer = 0.0
            control.throttle = 0.0
            control.brake = 0.3
            return control

# Main interface function
def get_apollo_control(vehicle, world):
    """Get Apollo-style autonomous driving control"""
    return ApolloStyleDriver.apollo_control(vehicle, world)