# -*- coding: utf-8 -*-
"""
Enhanced CARLA Autopilot with Junction Navigation
Based on CARLA's built-in autopilot but with enhanced junction handling
"""

import carla
import numpy as np
import math
import random
from enum import Enum

class NavigationState(Enum):
    LANE_FOLLOW = "lane_follow"
    APPROACHING_JUNCTION = "approaching_junction" 
    IN_JUNCTION = "in_junction"
    JUNCTION_EXIT = "junction_exit"

class CarlaEnhancedAutopilot:
    """Enhanced autopilot using CARLA's navigation system"""
    
    def __init__(self, vehicle, world, destination=None):
        self.vehicle = vehicle
        self.world = world
        self.map = world.get_map()
        
        # Navigation
        self.destination = destination
        self.route_waypoints = []
        self.current_waypoint_index = 0
        
        # State
        self.navigation_state = NavigationState.LANE_FOLLOW
        self.junction_decision = None
        
        # Control parameters
        self.target_speed = 30.0  # km/h
        self.junction_speed = 15.0  # km/h
        self.following_distance = 10.0  # meters
        
        # Traffic light detection
        self.traffic_light_state = None
        
        print("Enhanced CARLA Autopilot initialized")
        
    def set_destination(self, destination_location):
        """Set navigation destination and compute route"""
        self.destination = destination_location
        
        # Get current and destination waypoints
        current_waypoint = self.map.get_waypoint(self.vehicle.get_location())
        destination_waypoint = self.map.get_waypoint(destination_location)
        
        if current_waypoint and destination_waypoint:
            # Use CARLA's built-in route planner
            self.route_waypoints = self._compute_route(current_waypoint, destination_waypoint)
            self.current_waypoint_index = 0
            print(f"Route computed: {len(self.route_waypoints)} waypoints")
            
    def _compute_route(self, start_waypoint, end_waypoint):
        """Compute route using CARLA's global route planner"""
        try:
            # Use CARLA's GlobalRoutePlanner if available
            from agents.navigation.global_route_planner import GlobalRoutePlanner
            
            grp = GlobalRoutePlanner(self.map, 2.0)
            route = grp.trace_route(start_waypoint.transform.location, 
                                  end_waypoint.transform.location)
            
            return [waypoint for waypoint, _ in route]
        except ImportError:
            # Fallback to simple waypoint following
            print("GlobalRoutePlanner not available, using simple navigation")
            return self._simple_route_planning(start_waypoint, end_waypoint)
            
    def _simple_route_planning(self, start_waypoint, end_waypoint):
        """Simple route planning as fallback"""
        route = []
        current_wp = start_waypoint
        
        # Generate route toward destination
        for _ in range(100):  # Max 100 waypoints
            if not current_wp:
                break
                
            route.append(current_wp)
            
            # Check if we're close to destination
            distance_to_dest = current_wp.transform.location.distance(
                end_waypoint.transform.location)
            if distance_to_dest < 5.0:
                route.append(end_waypoint)
                break
                
            # Get next waypoints
            next_waypoints = current_wp.next(3.0)
            if not next_waypoints:
                break
                
            # Choose waypoint closest to destination
            best_wp = None
            best_distance = float('inf')
            
            for wp in next_waypoints:
                dist = wp.transform.location.distance(end_waypoint.transform.location)
                if dist < best_distance:
                    best_distance = dist
                    best_wp = wp
                    
            current_wp = best_wp
            
        return route
        
    def detect_junction_state(self):
        """Detect if vehicle is approaching or in a junction"""
        current_waypoint = self.map.get_waypoint(self.vehicle.get_location())
        
        if not current_waypoint:
            return NavigationState.LANE_FOLLOW
            
        # Check if current waypoint is in junction
        if current_waypoint.is_junction:
            return NavigationState.IN_JUNCTION
            
        # Check if approaching junction (look ahead)
        future_waypoints = current_waypoint.next(10.0)
        for wp in future_waypoints:
            if wp.is_junction:
                return NavigationState.APPROACHING_JUNCTION
                
        return NavigationState.LANE_FOLLOW
        
    def make_junction_decision(self, current_waypoint):
        """Make intelligent junction decisions based on route"""
        if not self.route_waypoints or self.current_waypoint_index >= len(self.route_waypoints):
            # No route, use CARLA's autopilot decision
            next_waypoints = current_waypoint.next(5.0)
            if next_waypoints:
                return next_waypoints[0]  # Go straight by default
            return current_waypoint
            
        # Follow the planned route
        target_waypoint = self.route_waypoints[self.current_waypoint_index]
        
        # Update waypoint index if we're close to current target
        vehicle_location = self.vehicle.get_location()
        distance_to_target = vehicle_location.distance(target_waypoint.transform.location)
        
        if distance_to_target < 3.0:
            self.current_waypoint_index = min(
                self.current_waypoint_index + 1, 
                len(self.route_waypoints) - 1
            )
            if self.current_waypoint_index < len(self.route_waypoints):
                target_waypoint = self.route_waypoints[self.current_waypoint_index]
                
        return target_waypoint
        
    def get_traffic_light_state(self):
        """Get traffic light state affecting the vehicle"""
        vehicle_waypoint = self.map.get_waypoint(self.vehicle.get_location())
        
        if vehicle_waypoint and vehicle_waypoint.is_junction:
            # Get traffic light affecting this junction
            traffic_lights = self.world.get_actors().filter('traffic.traffic_light*')
            
            for traffic_light in traffic_lights:
                if traffic_light.get_location().distance(self.vehicle.get_location()) < 15.0:
                    return traffic_light.get_state()
                    
        return carla.TrafficLightState.Green
        
    def compute_control(self):
        """Main control computation"""
        # Update navigation state
        self.navigation_state = self.detect_junction_state()
        
        # Get current waypoint
        current_waypoint = self.map.get_waypoint(self.vehicle.get_location())
        if not current_waypoint:
            return self._emergency_stop()
            
        # Determine target waypoint based on state
        if self.navigation_state in [NavigationState.APPROACHING_JUNCTION, NavigationState.IN_JUNCTION]:
            target_waypoint = self.make_junction_decision(current_waypoint)
            target_speed = self.junction_speed
        else:
            # Normal lane following
            next_waypoints = current_waypoint.next(5.0)
            target_waypoint = next_waypoints[0] if next_waypoints else current_waypoint
            target_speed = self.target_speed
            
        # Check traffic lights
        traffic_state = self.get_traffic_light_state()
        if traffic_state == carla.TrafficLightState.Red:
            target_speed = 0.0
        elif traffic_state == carla.TrafficLightState.Yellow:
            target_speed = min(target_speed, 15.0)
            
        # Compute control commands
        return self._compute_vehicle_control(target_waypoint, target_speed)
        
    def _compute_vehicle_control(self, target_waypoint, target_speed):
        """Compute vehicle control commands"""
        # Get vehicle state
        vehicle_transform = self.vehicle.get_transform()
        vehicle_location = vehicle_transform.location
        vehicle_rotation = vehicle_transform.rotation
        
        # Target state
        target_location = target_waypoint.transform.location
        target_rotation = target_waypoint.transform.rotation
        
        # Compute steering (lateral control)
        steering = self._compute_steering(vehicle_transform, target_waypoint.transform)
        
        # Compute speed control (longitudinal control)
        throttle, brake = self._compute_speed_control(target_speed)
        
        # Create control command
        control = carla.VehicleControl()
        control.steer = np.clip(steering, -1.0, 1.0)
        control.throttle = np.clip(throttle, 0.0, 1.0)
        control.brake = np.clip(brake, 0.0, 1.0)
        
        return control
        
    def _compute_steering(self, vehicle_transform, target_transform):
        """Compute steering angle using pure pursuit"""
        # Vector from vehicle to target
        dx = target_transform.location.x - vehicle_transform.location.x
        dy = target_transform.location.y - vehicle_transform.location.y
        
        # Vehicle heading
        vehicle_yaw = math.radians(vehicle_transform.rotation.yaw)
        
        # Transform to vehicle coordinate frame
        local_x = dx * math.cos(vehicle_yaw) + dy * math.sin(vehicle_yaw)
        local_y = -dx * math.sin(vehicle_yaw) + dy * math.cos(vehicle_yaw)
        
        # Pure pursuit steering
        lookahead_distance = math.sqrt(local_x**2 + local_y**2)
        if lookahead_distance < 0.1:
            return 0.0
            
        # Curvature calculation
        curvature = 2.0 * local_y / (lookahead_distance**2)
        
        # Convert to steering angle (simplified bicycle model)
        wheelbase = 2.7  # Approximate vehicle wheelbase
        steering_angle = math.atan(curvature * wheelbase)
        
        # Normalize to [-1, 1]
        steering = steering_angle / math.radians(30)  # Max 30 degrees
        
        return steering
        
    def _compute_speed_control(self, target_speed_kmh):
        """Compute throttle and brake using PID control"""
        # Current speed
        velocity = self.vehicle.get_velocity()
        current_speed_ms = math.sqrt(velocity.x**2 + velocity.y**2)
        current_speed_kmh = current_speed_ms * 3.6
        
        # Speed error
        speed_error = target_speed_kmh - current_speed_kmh
        
        # PID control (simplified)
        kp = 0.2
        
        if speed_error > 0:
            # Need to accelerate
            throttle = min(1.0, kp * speed_error / 10.0)
            brake = 0.0
        else:
            # Need to decelerate
            throttle = 0.0
            brake = min(1.0, abs(speed_error) * kp / 10.0)
            
        return throttle, brake
        
    def _emergency_stop(self):
        """Emergency stop control"""
        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 0.0
        control.brake = 1.0
        return control

# Main interface function
def get_enhanced_autopilot_control(vehicle, world, destination=None):
    """Get enhanced autopilot control with junction navigation"""
    if not hasattr(vehicle, 'enhanced_autopilot'):
        vehicle.enhanced_autopilot = CarlaEnhancedAutopilot(vehicle, world, destination)
        
    autopilot = vehicle.enhanced_autopilot
    
    # Set destination if provided
    if destination and not autopilot.destination:
        autopilot.set_destination(destination)
        
    return autopilot.compute_control()