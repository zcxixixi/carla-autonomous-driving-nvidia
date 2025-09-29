#!/usr/bin/env python3
"""
Clean CARLA Data Collection Script
Collect driving data with affordance labels for training
"""

import carla
import random
import time
import os
import json
import math
import numpy as np


def collect_clean_dataset(duration_minutes=2, scenario_count=1):
    """Collect a clean research dataset"""
    print("=" * 60)
    print("CARLA Clean Dataset Collection")
    print("=" * 60)
    
    actors_list = []
    
    try:
        # Connect to CARLA
        client = carla.Client('localhost', 2000)
        client.set_timeout(15.0)
        world = client.get_world()
        
        # Set synchronous mode
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.1  # 10 FPS
        world.apply_settings(settings)
        
        print("Connected to CARLA server in synchronous mode")
        
        # Create data directory
        base_dir = "clean_research_data"
        scenario_dir = os.path.join(base_dir, "scenario_00")
        os.makedirs(os.path.join(scenario_dir, "rgb"), exist_ok=True)
        os.makedirs(os.path.join(scenario_dir, "metadata"), exist_ok=True)
        
        print(f"Created directory: {scenario_dir}")
        
        # Spawn ego vehicle with camera
        ego_vehicle, camera = spawn_ego_with_camera(world, scenario_dir)
        actors_list.extend([ego_vehicle, camera])
        
        # Add some traffic
        traffic_actors = spawn_basic_traffic(world, num_vehicles=10)
        actors_list.extend(traffic_actors)
        
        # Collect data
        collect_data_loop(world, ego_vehicle, duration_minutes * 60, scenario_dir)
        
        print("Dataset collection completed!")
        
    except Exception as e:
        print(f"Collection failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cleanup_actors(actors_list)
        
        # Reset to async mode
        settings = world.get_settings() 
        settings.synchronous_mode = False
        world.apply_settings(settings)


def spawn_ego_with_camera(world, scenario_dir):
    """Spawn ego vehicle with RGB camera"""
    blueprint_library = world.get_blueprint_library()
    
    # Spawn ego vehicle
    vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
    spawn_points = world.get_map().get_spawn_points()
    spawn_point = random.choice(spawn_points)
    
    ego_vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    ego_vehicle.set_autopilot(True)
    print("Spawned ego vehicle with autopilot")
    
    # RGB Camera
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '640')
    camera_bp.set_attribute('image_size_y', '480')
    camera_bp.set_attribute('sensor_tick', '0.1')
    
    camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=ego_vehicle)
    
    # Set up camera callback
    camera.listen(lambda image: image.save_to_disk(
        f'{scenario_dir}/rgb/frame_{image.frame:06d}.png'))
    
    print("Spawned camera sensor")
    return ego_vehicle, camera


def spawn_basic_traffic(world, num_vehicles=10):
    """Spawn basic traffic"""
    actors = []
    blueprint_library = world.get_blueprint_library()
    
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
    
    print(f"Spawned {len(actors)} traffic vehicles")
    return actors


def collect_data_loop(world, ego_vehicle, duration_seconds, scenario_dir):
    """Main data collection loop"""
    print(f"Collecting data for {duration_seconds} seconds...")
    
    for step in range(int(duration_seconds * 10)):  # 10 FPS
        world.tick()
        
        # Get snapshot
        snapshot = world.get_snapshot()
        frame = snapshot.frame
        
        # Progress update
        if step % 50 == 0:  # every 5 seconds
            progress = (step / (duration_seconds * 10)) * 100
            print(f"Progress: {progress:.1f}% (frame {frame})")
        
        # Collect affordance labels
        metadata = collect_affordance_labels(world, ego_vehicle, snapshot)
        
        # Save metadata
        meta_path = os.path.join(scenario_dir, 'metadata', f'frame_{frame:06d}.json')
        try:
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            print(f"Warning: failed to write metadata for frame {frame}: {e}")
        
        time.sleep(0.001)


def collect_affordance_labels(world, ego_vehicle, snapshot):
    """Collect affordance labels for current frame"""
    # Ego telemetry
    transform = ego_vehicle.get_transform()
    velocity = ego_vehicle.get_velocity()
    speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
    
    ego_loc = transform.location
    ego_forward = transform.get_forward_vector()
    
    # Find nearest vehicle in front
    min_dist = float('inf')
    nearest_rel_speed = 0.0
    
    for actor in world.get_actors().filter('vehicle.*'):
        try:
            if actor.id == ego_vehicle.id:
                continue
            other_loc = actor.get_transform().location
            dx = other_loc.x - ego_loc.x
            dy = other_loc.y - ego_loc.y
            dz = other_loc.z - ego_loc.z
            dist = math.sqrt(dx*dx + dy*dy + dz*dz)
            
            # Check if in front
            dot = dx*ego_forward.x + dy*ego_forward.y + dz*ego_forward.z
            if dot > 0 and dist < min_dist:
                min_dist = dist
                other_vel = actor.get_velocity()
                other_speed = math.sqrt(other_vel.x**2 + other_vel.y**2 + other_vel.z**2)
                nearest_rel_speed = max(speed - other_speed, 0.0)
        except Exception:
            continue
    
    # Calculate affordances
    front_vehicle_distance = min_dist if min_dist < 100.0 else None
    
    # Simple TTC calculation
    if front_vehicle_distance and nearest_rel_speed > 0.1:
        ttc = front_vehicle_distance / nearest_rel_speed
    else:
        ttc = None
    
    # Simple risk score (higher when TTC is low)
    risk_score = 0.0
    if ttc is not None and ttc < 5.0:
        risk_score = max(0.0, 1.0 - (ttc / 5.0))
    
    # Build metadata
    metadata = {
        'frame': snapshot.frame,
        'timestamp': float(snapshot.timestamp.elapsed_seconds),
        'ego': {
            'location': [ego_loc.x, ego_loc.y, ego_loc.z],
            'rotation': [transform.rotation.pitch, transform.rotation.yaw, transform.rotation.roll],
            'speed_m_s': speed
        },
        'front_vehicle_distance_m': front_vehicle_distance,
        'time_to_collision_s': ttc,
        'lane_offset_m': None,  # TODO: implement with waypoint API
        'traffic_light_state': None,  # TODO: implement with traffic light API
        'high_level_command': 'straight',
        'risk_score': risk_score
    }
    
    return metadata


def cleanup_actors(actors):
    """Clean up spawned actors"""
    for actor in actors:
        try:
            if hasattr(actor, 'is_listening') and actor.is_listening:
                actor.stop()
            if hasattr(actor, 'is_alive') and actor.is_alive:
                actor.destroy()
        except:
            pass


if __name__ == '__main__':
    collect_clean_dataset(duration_minutes=2, scenario_count=1)