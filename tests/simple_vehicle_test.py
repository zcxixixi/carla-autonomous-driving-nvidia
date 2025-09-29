# -*- coding: utf-8 -*-
"""
Simple CARLA Vehicle Test
Basic vehicle spawning and control test for CARLA
"""

import carla
import random
import time
import sys

def main():
    print("=" * 50)
    print("CARLA Vehicle Spawn Test")
    print("=" * 50)
    
    actors_list = []
    
    try:
        # Connect to CARLA
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        world = client.get_world()
        
        print(f"? Connected to CARLA server")
        print(f"  Current map: {world.get_map().name}")
        
        # Get blueprint library
        blueprint_library = world.get_blueprint_library()
        
        # Get vehicle blueprints
        vehicle_blueprints = blueprint_library.filter('vehicle.*')
        print(f"? Found {len(vehicle_blueprints)} vehicle types")
        
        # Get spawn points
        spawn_points = world.get_map().get_spawn_points()
        print(f"? Found {len(spawn_points)} spawn points")
        
        # Spawn a few vehicles
        for i in range(3):
            # Pick random vehicle and spawn point
            vehicle_bp = random.choice(vehicle_blueprints)
            spawn_point = random.choice(spawn_points)
            
            try:
                # Try to spawn vehicle
                vehicle = world.spawn_actor(vehicle_bp, spawn_point)
                actors_list.append(vehicle)
                
                # Enable autopilot
                vehicle.set_autopilot(True)
                
                print(f"? Spawned vehicle {i+1}: {vehicle_bp.id} at location ({spawn_point.location.x:.1f}, {spawn_point.location.y:.1f})")
                
                # Remove spawn point to avoid collision
                spawn_points.remove(spawn_point)
                
            except Exception as e:
                print(f"? Failed to spawn vehicle {i+1}: {e}")
        
        print(f"? Successfully spawned {len(actors_list)} vehicles")
        
        if actors_list:
            print("\n Running simulation for 10 seconds...")
            print("Watch the vehicles drive autonomously!")
            
            for i in range(10):
                time.sleep(1)
                print(f"  Time: {i+1}/10 seconds")
            
            print("? Test completed successfully!")
        
    except Exception as e:
        print(f"? Test failed: {e}")
        return 1
        
    finally:
        # Clean up
        print(f"\nCleaning up {len(actors_list)} spawned vehicles...")
        for actor in actors_list:
            try:
                actor.destroy()
            except:
                pass
        print("? Cleanup completed")
        
    return 0

if __name__ == '__main__':
    sys.exit(main())