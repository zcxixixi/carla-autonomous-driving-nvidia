"""
Manual PilotNet Data Collection for CARLA
Drive manually with WASD keys to collect training data
Saves image-steering pairs in PilotNet format
"""
import carla
import cv2
import numpy as np
import os
import json
import time
import math
from datetime import datetime

def collect_pilotnet_data():
    """Collect manual driving data for PilotNet training"""
    
    print("=" * 60)
    print("PILOTNET DATA COLLECTION")
    print("=" * 60)
    print("Controls:")
    print("W/A/S/D - Drive manually")
    print("SPACE - Brake")
    print("Q - Quit")
    print("R - Start/Stop recording")
    print("=" * 60)
    
    # Create data directory
    data_dir = "pilotnet_data"
    os.makedirs(data_dir, exist_ok=True)
    
    # Create session directory
    session_dir = os.path.join(data_dir, f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(session_dir, exist_ok=True)
    
    images_dir = os.path.join(session_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    print(f"Data will be saved to: {session_dir}")
    
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
        # Spawn vehicle
        spawn_points = world.get_map().get_spawn_points()
        vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
        spawn_point = spawn_points[0]
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        print(f"? Vehicle spawned")
        
        # Setup camera (PilotNet format: 224x224)
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '224')
        camera_bp.set_attribute('image_size_y', '224')
        camera_bp.set_attribute('fov', '90')
        
        camera_transform = carla.Transform(carla.Location(x=2.0, z=1.4))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        
        # Camera callback
        image_data = {"frame": None, "timestamp": None}
        def process_image(image):
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = array.reshape((image.height, image.width, 4))
            array = array[:, :, :3]  # Remove alpha channel
            image_data["frame"] = array.copy()
            image_data["timestamp"] = image.timestamp
        
        camera.listen(process_image)
        
        # Wait for camera initialization
        world.tick()
        while image_data["frame"] is None:
            print("Waiting for camera...")
            time.sleep(0.1)
            world.tick()
        
        print("\n" + "="*50)
        print("? MANUAL DRIVING MODE ACTIVE")
        print("Drive with WASD keys, press R to start recording")
        print("="*50)
        
        # Data collection variables
        recording = False
        sample_count = 0
        labels = []
        
        # Control state
        throttle = 0.0
        steer = 0.0
        brake = 0.0
        
        # Key states for smooth control
        keys_pressed = set()
        
        while True:
            world.tick()
            
            if image_data["frame"] is not None:
                frame = image_data["frame"]
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                # Update key states
                if key == ord('w'):
                    keys_pressed.add('w')
                elif key == ord('s'):
                    keys_pressed.add('s')
                elif key == ord('a'):
                    keys_pressed.add('a')
                elif key == ord('d'):
                    keys_pressed.add('d')
                elif key == ord(' '):
                    keys_pressed.add('space')
                elif key == ord('q'):
                    break
                elif key == ord('r'):
                    recording = not recording
                    if recording:
                        print("? RECORDING STARTED")
                    else:
                        print("?? RECORDING STOPPED")
                
                # Calculate control inputs
                throttle = 0.0
                steer = 0.0
                brake = 0.0
                
                if 'w' in keys_pressed:
                    throttle = 0.6
                if 's' in keys_pressed:
                    throttle = -0.3  # Reverse
                if 'a' in keys_pressed:
                    steer = -0.7
                if 'd' in keys_pressed:
                    steer = 0.7
                if 'space' in keys_pressed:
                    brake = 1.0
                    throttle = 0.0
                
                # Apply smooth control
                control = carla.VehicleControl()
                control.throttle = max(0, throttle)
                control.steer = steer
                control.brake = max(0, brake if brake > 0 else -throttle)
                vehicle.apply_control(control)
                
                # Get vehicle state
                velocity = vehicle.get_velocity()
                speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2) * 3.6
                
                # Save data if recording
                if recording and abs(steer) > 0.05:  # Only save when steering
                    image_filename = f"image_{sample_count:06d}.png"
                    image_path = os.path.join(images_dir, image_filename)
                    
                    # Save image
                    cv2.imwrite(image_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    
                    # Save label
                    label = {
                        "image": image_filename,
                        "steering": float(steer),
                        "throttle": float(max(0, throttle)),
                        "brake": float(brake),
                        "speed": float(speed),
                        "timestamp": image_data["timestamp"]
                    }
                    labels.append(label)
                    
                    sample_count += 1
                    
                    if sample_count % 50 == 0:
                        print(f"? Collected {sample_count} samples")
                
                # Display frame with information
                display_frame = frame.copy()
                
                # Add information overlay
                y_offset = 25
                status_color = (0, 255, 0) if recording else (0, 0, 255)
                status_text = "RECORDING" if recording else "PAUSED"
                cv2.putText(display_frame, f"Status: {status_text}", (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                
                y_offset += 30
                cv2.putText(display_frame, f"Samples: {sample_count}", (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                y_offset += 30
                cv2.putText(display_frame, f"Speed: {speed:.1f} km/h", (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                y_offset += 35
                cv2.putText(display_frame, "Controls:", (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                y_offset += 25
                cv2.putText(display_frame, f"  Steering: {steer:.2f}", (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                y_offset += 20
                cv2.putText(display_frame, f"  Throttle: {throttle:.2f}", (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                y_offset += 20
                cv2.putText(display_frame, f"  Brake: {brake:.2f}", (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # Show steering direction
                if abs(steer) > 0.05:
                    center_x, center_y = 112, 180
                    if steer < 0:  # Left
                        cv2.arrowedLine(display_frame, (center_x, center_y), 
                                       (center_x - 40, center_y), (0, 255, 255), 2)
                    else:  # Right
                        cv2.arrowedLine(display_frame, (center_x, center_y), 
                                       (center_x + 40, center_y), (0, 255, 255), 2)
                
                cv2.imshow("PilotNet Data Collection", display_frame)
                
                # Clear key states (for single-frame inputs)
                keys_pressed.clear()
        
        # Save labels to JSON file
        labels_file = os.path.join(session_dir, "labels.json")
        with open(labels_file, 'w') as f:
            json.dump(labels, f, indent=2)
        
        print("\n" + "="*50)
        print("? DATA COLLECTION COMPLETED")
        print("="*50)
        print(f"Total samples collected: {sample_count}")
        print(f"Data saved to: {session_dir}")
        print(f"Images: {images_dir}")
        print(f"Labels: {labels_file}")
        
        if sample_count > 0:
            avg_steering = np.mean([abs(label['steering']) for label in labels])
            print(f"Average steering magnitude: {avg_steering:.3f}")
            
            steering_distribution = [label['steering'] for label in labels]
            print(f"Steering range: [{min(steering_distribution):.3f}, {max(steering_distribution):.3f}]")
        
    except Exception as e:
        print(f"? Error during data collection: {e}")
        import traceback
        traceback.print_exc()
    
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