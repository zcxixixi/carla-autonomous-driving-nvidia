# -*- coding: utf-8 -*-
"""
Enhanced CARLA Native Autopilot with Smooth Curve Handling
Optimized for smooth cornering and improved driving experience
"""

import carla
import math
import time

class SmoothCarlaAutopilot:
    """Enhanced CARLA autopilot with smooth curve handling"""
    
    def __init__(self, vehicle, world):
        self.vehicle = vehicle
        self.world = world
        
        # Get Traffic Manager for enhanced control
        self.client = carla.Client('localhost', 2000)
        self.traffic_manager = self.client.get_trafficmanager()
        
        # Smooth driving parameters
        self.setup_smooth_driving()
        
        # Enable autopilot with Traffic Manager
        try:
            tm_port = self.traffic_manager.get_port()
            self.vehicle.set_autopilot(True, tm_port)
        except:
            # Fallback to basic autopilot
            self.vehicle.set_autopilot(True)
            print("Using basic CARLA autopilot")
        
        print("Enhanced CARLA Autopilot initialized with smooth curve handling")
        
    def setup_smooth_driving(self):
        """Setup Traffic Manager for smooth driving"""
        try:
            # Reduce speed for smoother curves (percentage below speed limit)
            self.traffic_manager.global_percentage_speed_difference(20.0)
            
            # Set vehicle-specific parameters for smoother driving
            # Smoother lane changes and turns
            self.traffic_manager.distance_to_leading_vehicle(self.vehicle, 5.0)  # Following distance
            self.traffic_manager.vehicle_percentage_speed_difference(self.vehicle, 25.0)  # Speed reduction
            
            # Improved collision detection
            self.traffic_manager.collision_detection(self.vehicle, True)
            
            # Smoother lane changes
            self.traffic_manager.auto_lane_change(self.vehicle, True)
            
            # Respect traffic lights and signs
            self.traffic_manager.ignore_lights_percentage(self.vehicle, 0.0)  # Always respect lights
            self.traffic_manager.ignore_signs_percentage(self.vehicle, 0.0)   # Always respect signs
            
            # Keep right lane preference for smoother traffic flow
            self.traffic_manager.force_lane_change(self.vehicle, False)
            
            print("Traffic Manager configured for smooth driving")
            
        except Exception as e:
            print(f"Traffic Manager setup warning: {e}")
            print("Using basic autopilot without advanced traffic management")

def get_enhanced_carla_autopilot_control(vehicle, world):
    """Get enhanced CARLA autopilot with smooth curve handling"""
    if not hasattr(vehicle, 'smooth_carla_autopilot'):
        vehicle.smooth_carla_autopilot = SmoothCarlaAutopilot(vehicle, world)
        
    # Return empty control since autopilot handles everything
    return None  # Let autopilot handle control