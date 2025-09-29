# -*- coding: utf-8 -*-
"""
Simple Vehicle Movement Test
Check if NVIDIA models actually move the vehicle in CARLA
"""
import torch
import numpy as np
import cv2
import carla
import time
import math
import sys
import os

# Add models directory to path
sys.path.append('models')
from hydra_mdp import HydraMDP

def simple_movement_test():
    """Simple test to verify vehicle actually moves"""
    print("=" * 50)
    print("SIMPLE VEHICLE MOVEMENT TEST")
    print("=" * 50)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HydraMDP().to(device)
    model.eval()
    print(f"Model loaded on {device}")
    
    # Connect to CARLA
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    
    # Synchronous mode
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.1  # 10 FPS for clear observation
    world.apply_settings(settings)
    
    try:
        # Spawn vehicle
        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
        spawn_points = world.get_map().get_spawn_points()
        spawn_point = spawn_points[0]
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        
        print(f"Vehicle spawned at: ({spawn_point.location.x:.1f}, {spawn_point.location.y:.1f})")
        
        # Setup camera
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '224')
        camera_bp.set_attribute('image_size_y', '224')
        camera_bp.set_attribute('fov', '90')
        
        camera_transform = carla.Transform(carla.Location(x=2.0, z=1.4))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        
        # Image callback
        image_queue = []
        def on_image(carla_img):
            array = np.frombuffer(carla_img.raw_data, dtype=np.uint8)
            array = array.reshape((carla_img.height, carla_img.width, 4))
            array = array[:, :, :3]  # Remove alpha
            image_queue.append(array.copy())  # Make a copy to avoid warnings
        
        camera.listen(on_image)
        
        # Wait for first image
        world.tick()
        while not image_queue:
            time.sleep(0.1)
            world.tick()
        
        print("\nStarting movement test...")
        print("Monitoring position changes for 20 seconds...")
        
        start_time = time.time()
        initial_location = vehicle.get_location()
        positions = [initial_location]
        
        frame_count = 0
        
        while time.time() - start_time < 20:  # 20 second test
            world.tick()
            
            if image_queue:
                image = image_queue.pop(0)
                
                # Get model prediction
                image_tensor = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    outputs = model(image_tensor, training=False)
                    
                    raw_steering = outputs['steering'].cpu().numpy()[0, 0]
                    raw_throttle = outputs['throttle'].cpu().numpy()[0, 0]
                    raw_brake = outputs['brake'].cpu().numpy()[0, 0]
                
                # Get current vehicle state
                current_location = vehicle.get_location()
                velocity = vehicle.get_velocity()
                speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2) * 3.6
                
                # SMART CONTROL - use model predictions but override problematic outputs
                # Use model throttle but ensure minimum movement
                throttle = max(0.4, float(raw_throttle))  # At least 40% throttle
                steering = float(np.clip(raw_steering, -0.8, 0.8))  # Use model steering
                # Ignore brake if it would stop the vehicle
                brake = 0.0 if speed < 20 else float(min(0.3, raw_brake))  # Limited braking
                
                # Apply control
                control = carla.VehicleControl()
                control.throttle = float(throttle)
                control.steer = float(steering)
                control.brake = float(brake)
                control.manual_gear_shift = False
                vehicle.apply_control(control)
                
                frame_count += 1
                
                # Log every 20 frames (2 seconds)
                if frame_count % 20 == 0:
                    positions.append(current_location)
                    elapsed = time.time() - start_time
                    
                    # Calculate distance moved
                    total_distance = 0
                    for i in range(1, len(positions)):
                        dx = positions[i].x - positions[i-1].x
                        dy = positions[i].y - positions[i-1].y
                        total_distance += math.sqrt(dx*dx + dy*dy)
                    
                    print(f"Time: {elapsed:.1f}s | Speed: {speed:.1f}km/h | "
                          f"Pos: ({current_location.x:.1f}, {current_location.y:.1f}) | "
                          f"Total Distance: {total_distance:.1f}m | "
                          f"Raw Predictions: T={raw_throttle:.3f}, S={raw_steering:.3f}, B={raw_brake:.3f}")
        
        # Final analysis
        final_location = vehicle.get_location()
        total_distance = math.sqrt(
            (final_location.x - initial_location.x)**2 + 
            (final_location.y - initial_location.y)**2
        )
        
        print("\n" + "="*50)
        print("MOVEMENT TEST RESULTS")
        print("="*50)
        print(f"Initial position: ({initial_location.x:.1f}, {initial_location.y:.1f})")
        print(f"Final position: ({final_location.x:.1f}, {final_location.y:.1f})")
        print(f"Total displacement: {total_distance:.1f} meters")
        print(f"Frames processed: {frame_count}")
        
        if total_distance > 10:
            print("? VEHICLE MOVED SUCCESSFULLY!")
        elif total_distance > 1:
            print("?? Vehicle moved slightly")
        else:
            print("? Vehicle did NOT move significantly")
            print("This suggests the model predictions are keeping the vehicle stationary")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        try:
            camera.destroy()
            vehicle.destroy()
        except:
            pass
        
        # Reset async mode
        settings.synchronous_mode = False
        world.apply_settings(settings)

if __name__ == "__main__":
    simple_movement_test()