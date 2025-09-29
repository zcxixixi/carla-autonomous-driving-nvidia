# -*- coding: utf-8 -*-
"""
Extended Data Collection for Research
Collect large-scale datasets for research purposes
"""

import carla
import random
import time
import sys
import os
import json
import math
import numpy as np

def collect_research_dataset(duration_minutes=30, scenario_count=5):
    """收集研究用数据集"""
    print("=" * 60)
    print("CARLA Research Dataset Collection")
    print("=" * 60)
    
    actors_list = []
    
    try:
        # Connect to CARLA
        client = carla.Client('localhost', 2000)
        client.set_timeout(15.0)
        world = client.get_world()
        
        # Set synchronous mode for deterministic results
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.1  # 10 FPS
        world.apply_settings(settings)
        
        print(f"? Connected to CARLA server in synchronous mode")
        
        # Create data directories
        base_dir = "research_data"
        scenarios = []
        
        for i in range(scenario_count):
            scenario_dir = os.path.join(base_dir, f"scenario_{i:02d}")
            os.makedirs(os.path.join(scenario_dir, "rgb"), exist_ok=True)
            os.makedirs(os.path.join(scenario_dir, "depth"), exist_ok=True)
            os.makedirs(os.path.join(scenario_dir, "lidar"), exist_ok=True)
            os.makedirs(os.path.join(scenario_dir, "metadata"), exist_ok=True)
            scenarios.append(scenario_dir)
        
        print(f"? Created {scenario_count} scenario directories")
        
        # Run collection scenarios
        for scenario_idx, scenario_dir in enumerate(scenarios):
            print(f"\n? Collecting Scenario {scenario_idx + 1}/{scenario_count}")
            
            # Spawn ego vehicle with sensors
            ego_vehicle, sensors = spawn_ego_vehicle_with_sensors(world, scenario_dir)
            actors_list.extend([ego_vehicle] + sensors)
            
            # Add traffic (other vehicles and pedestrians)
            traffic_actors = spawn_traffic(world, num_vehicles=20, num_walkers=30)
            actors_list.extend(traffic_actors)
            
            # Collect data for this scenario
            collect_scenario_data(world, ego_vehicle, duration_minutes//scenario_count * 60, scenario_dir)
            
            # Clean up for next scenario
            cleanup_actors(actors_list)
            actors_list = []
            
            print(f"? Scenario {scenario_idx + 1} completed")
        
        print(f"\n? Dataset collection completed!")
        print(f"   Total scenarios: {scenario_count}")
        print(f"   Estimated data size: ~{scenario_count * duration_minutes * 50}MB")
        
    except Exception as e:
        print(f"? Collection failed: {e}")
    finally:
        cleanup_actors(actors_list)
        
        # Reset to async mode
        settings = world.get_settings() 
        settings.synchronous_mode = False
        world.apply_settings(settings)

def spawn_ego_vehicle_with_sensors(world, scenario_dir):
    """生成配备传感器的主车辆"""
    blueprint_library = world.get_blueprint_library()
    
    # Spawn ego vehicle
    vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
    spawn_points = world.get_map().get_spawn_points()
    spawn_point = random.choice(spawn_points)
    
    ego_vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    ego_vehicle.set_autopilot(True)
    
    sensors = []
    
    # RGB Camera
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '1920')
    camera_bp.set_attribute('image_size_y', '1080')
    camera_bp.set_attribute('sensor_tick', '0.1')
    
    camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=ego_vehicle)
    
    # Set up camera callback
    camera.listen(lambda image: image.save_to_disk(
        f'{scenario_dir}/rgb/frame_{image.frame:06d}.png'))
    sensors.append(camera)
    
    # Depth Camera
    depth_bp = blueprint_library.find('sensor.camera.depth')
    depth_bp.set_attribute('image_size_x', '1920') 
    depth_bp.set_attribute('image_size_y', '1080')
    depth_bp.set_attribute('sensor_tick', '0.1')
    
    depth_camera = world.spawn_actor(depth_bp, camera_transform, attach_to=ego_vehicle)
    depth_camera.listen(lambda image: image.save_to_disk(
        f'{scenario_dir}/depth/frame_{image.frame:06d}.png', carla.ColorConverter.LogarithmicDepth))
    sensors.append(depth_camera)
    
    # LiDAR
    lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('channels', '64')  # Higher resolution
    lidar_bp.set_attribute('points_per_second', '120000')
    lidar_bp.set_attribute('rotation_frequency', '10')
    lidar_bp.set_attribute('range', '100')  # Longer range
    
    lidar_transform = carla.Transform(carla.Location(x=0, z=2.5))
    lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=ego_vehicle)
    
    def save_lidar_data(point_cloud):
        data = np.frombuffer(point_cloud.raw_data, dtype=np.dtype('f4'))
        data = np.reshape(data, (int(data.shape[0] / 4), 4))
        np.save(f'{scenario_dir}/lidar/frame_{point_cloud.frame:06d}.npy', data)
        
        # Also save metadata
        metadata = {
            'frame': point_cloud.frame,
            'timestamp': point_cloud.timestamp,
            'transform': {
                'location': [point_cloud.transform.location.x, 
                           point_cloud.transform.location.y, 
                           point_cloud.transform.location.z],
                'rotation': [point_cloud.transform.rotation.pitch,
                           point_cloud.transform.rotation.yaw, 
                           point_cloud.transform.rotation.roll]
            }
        }
        np.save(f'{scenario_dir}/metadata/frame_{point_cloud.frame:06d}.npy', metadata)
    
    lidar.listen(save_lidar_data)
    sensors.append(lidar)
    
    return ego_vehicle, sensors

def spawn_traffic(world, num_vehicles=20, num_walkers=30):
    """生成交通参与者"""
    actors = []
    blueprint_library = world.get_blueprint_library()
    
    # Spawn vehicles
    vehicle_bps = blueprint_library.filter('vehicle.*')
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)
    
    for i in range(min(num_vehicles, len(spawn_points))):
        try:
            vehicle_bp = random.choice(vehicle_bps)
            vehicle = world.spawn_actor(vehicle_bp, spawn_points[i])
            vehicle.set_autopilot(True)
            actors.append(vehicle)
        except Exception:
            continue
    
    # Spawn walkers
    walker_bps = blueprint_library.filter('walker.pedestrian.*')
    walker_controller_bp = blueprint_library.find('controller.ai.walker')
    
    for i in range(num_walkers):
        try:
            spawn_point = world.get_random_location_from_navigation()
            walker_bp = random.choice(walker_bps)
            walker = world.spawn_actor(walker_bp, carla.Transform(spawn_point))
            actors.append(walker)
            
            walker_controller = world.spawn_actor(walker_controller_bp, carla.Transform(), walker)
            walker_controller.start()
            walker_controller.go_to_location(world.get_random_location_from_navigation())
            actors.append(walker_controller)
        except Exception:
            continue
    
    return actors

def collect_scenario_data(world, ego_vehicle, duration_seconds, scenario_dir):
    """Collect scenario data and write per-frame labels according to schema.

    This function runs the synchronous simulation loop and writes a JSON
    metadata file per frame into `scenario_dir/metadata/frame_XXXXXX.json`.
    The metadata includes ego pose, velocity, nearest vehicle distance, a
    simple time-to-collision (TTC) estimate and placeholders for lane offset
    and traffic light state (these can be improved with additional sensors).
    """
    print(f"   Collecting data for {duration_seconds} seconds...")

    for step in range(duration_seconds * 10):  # 10 FPS since fixed_delta_seconds=0.1
        world.tick()  # Advance simulation deterministically

        # Snapshot and frame id
        snapshot = world.get_snapshot()
        frame = snapshot.frame

        # Progress update
        if step % 50 == 0:  # every 5 seconds
            progress = (step / (duration_seconds * 10)) * 100
            print(f"   Progress: {progress:.1f}% (frame {frame})")

        # Ego telemetry
        transform = ego_vehicle.get_transform()
        velocity = ego_vehicle.get_velocity()
        speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)

        ego_loc = transform.location
        ego_forward = transform.get_forward_vector()

        # Find nearest vehicle in front
        min_dist = float('inf')
        nearest_id = None
        nearest_rel_speed = None

        for actor in world.get_actors().filter('vehicle.*'):
            try:
                if actor.id == ego_vehicle.id:
                    continue
                other_loc = actor.get_transform().location
                dx = other_loc.x - ego_loc.x
                dy = other_loc.y - ego_loc.y
                dz = other_loc.z - ego_loc.z
                dist = math.sqrt(dx*dx + dy*dy + dz*dz)

                # Is the other vehicle roughly in front? use dot product
                dot = dx*ego_forward.x + dy*ego_forward.y + dz*ego_forward.z
                if dot > 0 and dist < min_dist:
                    min_dist = dist
                    nearest_id = actor.id
                    other_vel = actor.get_velocity()
                    other_speed = math.sqrt(other_vel.x**2 + other_vel.y**2 + other_vel.z**2)
                    nearest_rel_speed = max(speed - other_speed, 0.0)
            except Exception:
                continue

        if nearest_id is None:
            front_vehicle_distance = None
            ttc = None
        else:
            front_vehicle_distance = min_dist
            # Simple TTC estimate (avoid div by zero)
            if nearest_rel_speed > 0.1:
                ttc = front_vehicle_distance / nearest_rel_speed
            else:
                ttc = None

        # Placeholder lane offset and traffic light state — can be improved
        lane_offset = None
        traffic_light = None

        # risk_score placeholder: simple heuristic (higher risk if TTC small)
        risk_score = 0.0
        if ttc is not None:
            if ttc < 1.0:
                risk_score = 1.0
            else:
                risk_score = max(0.0, 1.0 - (ttc / 5.0))

        metadata = {
            'frame': frame,
            'timestamp': snapshot.timestamp,
            'ego': {
                'location': [ego_loc.x, ego_loc.y, ego_loc.z],
                'rotation': [transform.rotation.pitch, transform.rotation.yaw, transform.rotation.roll],
                'speed_m_s': speed
            },
            'front_vehicle_distance_m': front_vehicle_distance,
            'time_to_collision_s': ttc,
            'lane_offset_m': lane_offset,
            'traffic_light_state': traffic_light,
            'high_level_command': 'straight',
            'risk_score': risk_score
        }

        # Write metadata JSON per frame
        meta_path = os.path.join(scenario_dir, 'metadata', f'frame_{frame:06d}.json')
        try:
            with open(meta_path, 'w') as f:
                json.dump(metadata, f)
        except Exception as e:
            print(f"   Warning: failed to write metadata for frame {frame}: {e}")

        time.sleep(0.001)  # very small delay to yield

def cleanup_actors(actors):
    """清理生成的角色"""
    for actor in actors:
        try:
            if hasattr(actor, 'is_listening') and actor.is_listening:
                actor.stop()
            if hasattr(actor, 'is_alive') and actor.is_alive:
                actor.destroy()
        except:
            pass

if __name__ == '__main__':
    collect_research_dataset(duration_minutes=15, scenario_count=3)