"""
Fixed Data Collection Script for Training
Collects complete training data with proper JSON structure
"""
import carla
import random
import time
import numpy as np
import cv2
import os
import json
from datetime import datetime
import math


def collect_training_data():
    """Collect proper training data for the model"""
    
    print("=" * 60)
    print("COLLECTING TRAINING DATA FOR ACCIDENT-AWARE DRIVING")
    print("=" * 60)
    
    # Create directories
    scenario_dir = 'training_data/scenario_00'
    rgb_dir = os.path.join(scenario_dir, 'rgb')
    metadata_dir = os.path.join(scenario_dir, 'metadata')
    
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)
    
    # Connect to CARLA
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    
    # Set clear weather
    weather = carla.WeatherParameters.ClearNoon
    world.set_weather(weather)
    
    # Synchronous mode
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.1  # 10 FPS for data collection
    world.apply_settings(settings)
    
    blueprint_library = world.get_blueprint_library()
    
    try:
        # Spawn vehicle
        vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
        spawn_points = world.get_map().get_spawn_points()
        
        print(f"Available spawn points: {len(spawn_points)}")
        
        collected_samples = 0
        target_samples = 1000  # Collect 1000 good samples
        
        for spawn_idx in range(min(10, len(spawn_points))):  # Try different spawn points
            
            spawn_point = spawn_points[spawn_idx]
            print(f"\nSpawn point {spawn_idx + 1}: {spawn_point}")
            
            try:
                vehicle = world.spawn_actor(vehicle_bp, spawn_point)
                print(f"Vehicle spawned successfully")
            except Exception as e:
                print(f"Failed to spawn at point {spawn_idx}: {e}")
                continue
            
            # Setup camera
            camera_bp = blueprint_library.find('sensor.camera.rgb')
            camera_bp.set_attribute('image_size_x', '640')
            camera_bp.set_attribute('image_size_y', '480')
            camera_bp.set_attribute('fov', '90')
            
            camera_transform = carla.Transform(carla.Location(x=2.0, z=1.4))
            camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
            
            # Image capture
            image_data = {"frame": None}
            def process_image(image):
                array = np.frombuffer(image.raw_data, dtype=np.uint8)
                array = array.reshape((image.height, image.width, 4))
                array = array[:, :, :3]  # Remove alpha
                image_data["frame"] = array.copy()
            
            camera.listen(process_image)
            
            # Add some traffic for more realistic scenarios
            traffic_manager = client.get_trafficmanager(8000)
            traffic_manager.set_synchronous_mode(True)
            
            # Spawn some AI vehicles
            traffic_vehicles = []
            for i in range(10):  # Spawn 10 AI vehicles
                try:
                    ai_spawn = random.choice(spawn_points)
                    ai_vehicle = world.spawn_actor(
                        random.choice(blueprint_library.filter('vehicle.*')), 
                        ai_spawn
                    )
                    ai_vehicle.set_autopilot(True, 8000)
                    traffic_vehicles.append(ai_vehicle)
                except:
                    continue
            
            print(f"Spawned {len(traffic_vehicles)} AI vehicles")
            
            # Wait for camera to initialize
            for _ in range(10):
                world.tick()
                if image_data["frame"] is not None:
                    break
                time.sleep(0.1)
            
            if image_data["frame"] is None:
                print("Camera initialization failed")
                continue
            
            # Data collection loop for this spawn point
            frame_count = 0
            commands = ['straight', 'follow', 'left', 'right']
            
            print(f"Starting data collection at spawn point {spawn_idx + 1}...")
            
            while frame_count < 100 and collected_samples < target_samples:  # 100 frames per spawn point
                world.tick()
                frame_count += 1
                
                if image_data["frame"] is not None:
                    # Get vehicle state
                    vehicle_transform = vehicle.get_transform()
                    velocity = vehicle.get_velocity()
                    speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2) * 3.6  # km/h
                    
                    # Simple driving control to get vehicle moving
                    if speed < 30:  # Keep moving
                        throttle = 0.5
                        steer = random.uniform(-0.1, 0.1)  # Small random steering
                    else:
                        throttle = 0.3
                        steer = 0.0
                        
                    control = carla.VehicleControl()
                    control.throttle = throttle
                    control.steer = steer
                    control.brake = 0.0
                    vehicle.apply_control(control)
                    
                    # Simulate different scenarios and commands
                    current_command = random.choice(commands)
                    
                    # Calculate simple affordance values
                    # These are simulated values - in real scenario you'd use sensor data
                    front_distance = random.uniform(10.0, 50.0)  # 10-50 meters
                    lane_offset = random.uniform(-2.0, 2.0)  # -2 to +2 meters from lane center
                    
                    # Risk calculation based on distance and speed
                    if front_distance < 15:
                        risk_score = 0.8
                    elif front_distance < 25:
                        risk_score = 0.4
                    else:
                        risk_score = 0.1
                        
                    # Add some randomness
                    risk_score += random.uniform(-0.2, 0.2)
                    risk_score = np.clip(risk_score, 0.0, 1.0)
                    
                    # Save image
                    frame_id = f"frame_{collected_samples:06d}"
                    img_path = os.path.join(rgb_dir, f'{frame_id}.png')
                    cv2.imwrite(img_path, cv2.cvtColor(image_data["frame"], cv2.COLOR_RGB2BGR))
                    
                    # Save metadata with COMPLETE JSON
                    metadata = {
                        "frame": collected_samples,
                        "timestamp": time.time(),
                        "spawn_point": spawn_idx,
                        "vehicle_location": {
                            "x": vehicle_transform.location.x,
                            "y": vehicle_transform.location.y,
                            "z": vehicle_transform.location.z
                        },
                        "vehicle_rotation": {
                            "pitch": vehicle_transform.rotation.pitch,
                            "yaw": vehicle_transform.rotation.yaw,
                            "roll": vehicle_transform.rotation.roll
                        },
                        "velocity": {
                            "x": velocity.x,
                            "y": velocity.y,
                            "z": velocity.z
                        },
                        "speed_kmh": speed,
                        "control": {
                            "throttle": throttle,
                            "steer": steer,
                            "brake": 0.0
                        },
                        "command": current_command,
                        "front_distance": front_distance,
                        "lane_offset": lane_offset,
                        "risk_score": risk_score,
                        "weather": "ClearNoon",
                        "collection_time": datetime.now().isoformat()
                    }
                    
                    # Save metadata as complete JSON
                    meta_path = os.path.join(metadata_dir, f'{frame_id}.json')
                    with open(meta_path, 'w') as f:
                        json.dump(metadata, f, indent=2)
                    
                    collected_samples += 1
                    
                    if collected_samples % 50 == 0:
                        print(f"Collected {collected_samples}/{target_samples} samples...")
                
                # Break if we have enough samples
                if collected_samples >= target_samples:
                    break
            
            # Cleanup for this spawn point
            try:
                camera.destroy()
                vehicle.destroy()
                for ai_vehicle in traffic_vehicles:
                    ai_vehicle.destroy()
            except:
                pass
            
            print(f"Completed spawn point {spawn_idx + 1}: {collected_samples} total samples")
            
            if collected_samples >= target_samples:
                break
                
    finally:
        # Reset to async mode
        settings.synchronous_mode = False
        world.apply_settings(settings)
    
    print("\n" + "=" * 60)
    print("DATA COLLECTION COMPLETED")
    print("=" * 60)
    print(f"Total samples collected: {collected_samples}")
    print(f"Data saved in: {scenario_dir}")
    print("Ready for training!")
    
    return collected_samples


if __name__ == "__main__":
    samples = collect_training_data()
    print(f"\n? Successfully collected {samples} training samples!")
    print("You can now run the extended training script.")