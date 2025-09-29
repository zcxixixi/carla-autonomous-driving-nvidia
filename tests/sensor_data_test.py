# -*- coding: utf-8 -*-
"""
CARLA Sensor Data Collection Test
Test collecting camera and LiDAR data from vehicles
"""

import carla
import random
import time
import sys
import os
import numpy as np

def main():
    print("=" * 50)
    print("CARLA Sensor Data Collection Test")
    print("=" * 50)
    
    actors_list = []
    
    try:
        # Connect to CARLA
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        world = client.get_world()
        
        print(f"? Connected to CARLA server")
        
        # Create data directory
        data_dir = "data"
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            print(f"? Created data directory: {data_dir}")
        
        # Get blueprint library
        blueprint_library = world.get_blueprint_library()
        
        # Spawn a vehicle
        vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
        spawn_points = world.get_map().get_spawn_points()
        spawn_point = random.choice(spawn_points)
        
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        actors_list.append(vehicle)
        vehicle.set_autopilot(True)
        
        print(f"? Spawned vehicle: Tesla Model 3")
        
        # Add RGB camera
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '800')
        camera_bp.set_attribute('image_size_y', '600')
        camera_bp.set_attribute('sensor_tick', '0.1')
        
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        actors_list.append(camera)
        
        # Add depth camera
        depth_bp = blueprint_library.find('sensor.camera.depth')
        depth_bp.set_attribute('image_size_x', '800')
        depth_bp.set_attribute('image_size_y', '600')
        depth_bp.set_attribute('sensor_tick', '0.1')
        
        depth_camera = world.spawn_actor(depth_bp, camera_transform, attach_to=vehicle)
        actors_list.append(depth_camera)
        
        # Add LiDAR
        lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('channels', '32')
        lidar_bp.set_attribute('points_per_second', '90000')
        lidar_bp.set_attribute('rotation_frequency', '10')
        lidar_bp.set_attribute('range', '50')
        
        lidar_transform = carla.Transform(carla.Location(x=0, z=2.5))
        lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)
        actors_list.append(lidar)
        
        print(f"? Added sensors: RGB camera, depth camera, LiDAR")
        
        # Data counters
        image_count = 0
        depth_count = 0
        lidar_count = 0
        
        # Set up data callbacks
        def save_rgb_image(image):
            nonlocal image_count
            image.save_to_disk(f'{data_dir}/rgb_{image_count:04d}.png')
            image_count += 1
            print(f"  Saved RGB image: {image_count}")
        
        def save_depth_image(image):
            nonlocal depth_count
            image.save_to_disk(f'{data_dir}/depth_{depth_count:04d}.png', carla.ColorConverter.LogarithmicDepth)
            depth_count += 1
            print(f"  Saved depth image: {depth_count}")
        
        def save_lidar_data(point_cloud):
            nonlocal lidar_count
            data = np.frombuffer(point_cloud.raw_data, dtype=np.dtype('f4'))
            data = np.reshape(data, (int(data.shape[0] / 4), 4))
            np.save(f'{data_dir}/lidar_{lidar_count:04d}.npy', data)
            lidar_count += 1
            print(f"  Saved LiDAR data: {lidar_count}")
        
        # Start listening to sensors
        camera.listen(save_rgb_image)
        depth_camera.listen(save_depth_image)
        lidar.listen(save_lidar_data)
        
        print("? Sensors are now collecting data...")
        print("  Running data collection for 15 seconds...")
        
        # Collect data for 15 seconds
        for i in range(15):
            time.sleep(1)
            print(f"  Progress: {i+1}/15 seconds")
        
        print(f"? Data collection completed!")
        print(f"  Collected {image_count} RGB images")
        print(f"  Collected {depth_count} depth images") 
        print(f"  Collected {lidar_count} LiDAR point clouds")
        
    except Exception as e:
        print(f"? Test failed: {e}")
        return 1
        
    finally:
        # Clean up
        print(f"\nCleaning up {len(actors_list)} actors...")
        for actor in actors_list:
            try:
                if actor.is_listening:
                    actor.stop()
                actor.destroy()
            except:
                pass
        print("? Cleanup completed")
        
    return 0

if __name__ == '__main__':
    sys.exit(main())