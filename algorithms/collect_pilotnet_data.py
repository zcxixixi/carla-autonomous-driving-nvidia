"""
Data Collection for NVIDIA PilotNet
Collects simple image-steering pairs by driving manually
"""
import carla
import numpy as np
import cv2
import os
import json
import time
import math
from datetime import datetime

def collect_pilotnet_data():
    """Collect driving data for PilotNet training"""
    
    print("=" * 60)
    print("COLLECTING DATA FOR NVIDIA PILOTNET")
    print("=" * 60)
    print("Drive manually using WASD keys")
    print("Data will be saved automatically")
    
    # Create directories
    data_dir = 'pilotnet_data'
    images_dir = os.path.join(data_dir, 'images')
    labels_dir = os.path.join(data_dir, 'labels')
    
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    # Connect to CARLA
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    world.set_weather(carla.WeatherParameters.ClearNoon)
    
    # Synchronous mode
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.1  # 10 FPS for data collection
    world.apply_settings(settings)
    
    blueprint_library = world.get_blueprint_library()
    
    try:
        # Spawn vehicle in a good location
        spawn_points = world.get_map().get_spawn_points()
        vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
        
        # Try different spawn points
        for i in [0, 1, 50, 100, 150]:
            try:
                spawn_point = spawn_points[i % len(spawn_points)]
                vehicle = world.spawn_actor(vehicle_bp, spawn_point)
                print(f"? Vehicle spawned at point {i}")
                break
            except Exception as e:
                print(f"? Failed to spawn at point {i}: {e}")
        
        if not vehicle:
            raise Exception("Could not spawn vehicle")
        
        # Setup camera (PilotNet uses specific resolution)
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '640')
        camera_bp.set_attribute('image_size_y', '480')
        camera_bp.set_attribute('fov', '90')
        
        # Camera mounted on the car like in PilotNet
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
        
        # Wait for camera initialization
        world.tick()
        while image_data["frame"] is None:
            time.sleep(0.01)
            world.tick()
        
        print("\n" + "="*50)
        print("MANUAL DRIVING DATA COLLECTION:")
        print("W = Forward")
        print("S = Brake/Reverse") 
        print("A = Steer Left")
        print("D = Steer Right")
        print("Q = Quit")
        print("Data saves automatically while driving!")
        print("="*50)
        
        # Control variables
        throttle = 0.0
        steer = 0.0
        brake = 0.0
        
        sample_count = 0
        target_samples = 2000  # Collect 2000 samples
        
        while sample_count < target_samples:
            world.tick()
            
            if image_data["frame"] is not None:
                frame = image_data["frame"]
                
                # Get vehicle state
                velocity = vehicle.get_velocity()
                speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2) * 3.6
                
                # Display current frame
                display_frame = frame.copy()
                
                # Add information overlay
                cv2.putText(display_frame, f"Samples: {sample_count}/{target_samples}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Speed: {speed:.1f} km/h", 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Steering: {steer:.2f}", 
                           (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                cv2.putText(display_frame, f"Throttle: {throttle:.2f}", 
                           (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                cv2.putText(display_frame, "WASD to drive, Q to quit", 
                           (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.imshow("PilotNet Data Collection", display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                # Reset controls
                throttle = 0.0
                steer = 0.0
                brake = 0.0
                
                if key == ord('w'):  # Forward
                    throttle = 0.6
                elif key == ord('s'):  # Brake
                    brake = 0.5
                elif key == ord('a'):  # Left
                    steer = -0.4
                    throttle = 0.3  # Keep moving while steering
                elif key == ord('d'):  # Right
                    steer = 0.4
                    throttle = 0.3  # Keep moving while steering
                elif key == ord('q'):  # Quit
                    break
                
                # Apply control to vehicle
                control = carla.VehicleControl()
                control.throttle = float(throttle)
                control.steer = float(steer)
                control.brake = float(brake)
                vehicle.apply_control(control)
                
                # Save data only when there's meaningful control input
                if abs(steer) > 0.01 or throttle > 0.01 or brake > 0.01:
                    # Save image (resize to PilotNet format)
                    pilotnet_image = cv2.resize(frame, (200, 66))  # Original PilotNet size
                    img_filename = f"image_{sample_count:06d}.jpg"
                    img_path = os.path.join(images_dir, img_filename)
                    
                    # Convert RGB to BGR for OpenCV
                    pilotnet_bgr = cv2.cvtColor(pilotnet_image, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(img_path, pilotnet_bgr)
                    
                    # Save label (steering angle and speed)
                    label_data = {
                        "steering_angle": float(steer),
                        "throttle": float(throttle),
                        "brake": float(brake),
                        "speed": float(speed),
                        "timestamp": time.time(),
                        "sample_id": sample_count
                    }
                    
                    label_filename = f"label_{sample_count:06d}.json"
                    label_path = os.path.join(labels_dir, label_filename)
                    
                    with open(label_path, 'w') as f:
                        json.dump(label_data, f, indent=2)
                    
                    sample_count += 1
                    
                    if sample_count % 100 == 0:
                        print(f"Collected {sample_count}/{target_samples} samples...")
        
        print(f"\n? Data collection completed!")
        print(f"Total samples: {sample_count}")
        print(f"Images saved in: {images_dir}")
        print(f"Labels saved in: {labels_dir}")
        
    finally:
        # Cleanup
        cv2.destroyAllWindows()
        try:
            camera.destroy()
            vehicle.destroy()
        except:
            pass
        
        # Reset to async mode
        settings.synchronous_mode = False
        world.apply_settings(settings)


if __name__ == "__main__":
    collect_pilotnet_data()