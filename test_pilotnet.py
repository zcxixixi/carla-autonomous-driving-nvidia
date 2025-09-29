"""
NVIDIA PilotNet Real-time Testing in CARLA
Test the trained PilotNet model with live camera feed
"""
import torch
import numpy as np
import cv2
import carla
import random
import time
import math
import sys
import os

# Add models directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))
from nvidia_pilotnet_fixed import ImprovedPilotNet

def test_pilotnet_realtime():
    """Test PilotNet in real-time CARLA environment"""
    
    print("=" * 60)
    print("NVIDIA PILOTNET REAL-TIME TEST")
    print("=" * 60)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    model = ImprovedPilotNet().to(device)
    
    # Load trained weights
    checkpoint_path = 'checkpoints/pilotnet_best.pt'
    if not os.path.exists(checkpoint_path):
        print(f"? Model checkpoint not found: {checkpoint_path}")
        print("Please train the model first using train_pilotnet.py")
        return
    
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    best_loss = checkpoint.get('best_val_loss', 'unknown')
    print(f"? PilotNet loaded (best val loss: {best_loss})")
    
    # Connect to CARLA
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    world.set_weather(carla.WeatherParameters.ClearNoon)
    
    # Synchronous mode
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05  # 20 FPS
    world.apply_settings(settings)
    
    blueprint_library = world.get_blueprint_library()
    
    try:
        # Spawn vehicle in different locations
        spawn_points = world.get_map().get_spawn_points()
        vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
        
        # Try multiple spawn points to avoid obstacles
        for i in [0, 1, 50, 100, 150, 200]:
            try:
                spawn_point = spawn_points[i % len(spawn_points)]
                vehicle = world.spawn_actor(vehicle_bp, spawn_point)
                print(f"? Vehicle spawned at point {i}")
                print(f"   Location: {spawn_point.location}")
                break
            except Exception as e:
                print(f"? Failed to spawn at point {i}: {e}")
        
        if not vehicle:
            raise Exception("Could not spawn vehicle")
        
        # Setup camera (same as training)
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '640')
        camera_bp.set_attribute('image_size_y', '480')
        camera_bp.set_attribute('fov', '90')
        
        camera_transform = carla.Transform(carla.Location(x=2.0, z=1.4))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        
        image_data = {"frame": None}
        def process_image(image):
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = array.reshape((image.height, image.width, 4))
            array = array[:, :, :3]  # Remove alpha
            image_data["frame"] = array.copy()
        
        camera.listen(process_image)
        
        # Collision sensor
        collision_bp = blueprint_library.find('sensor.other.collision')
        collision_sensor = world.spawn_actor(collision_bp, carla.Transform(), attach_to=vehicle)
        
        collision_data = {"count": 0}
        def on_collision(event):
            collision_data["count"] += 1
            other_actor = event.other_actor.type_id if event.other_actor else "unknown"
            print(f"? Collision #{collision_data['count']} with {other_actor}")
        
        collision_sensor.listen(on_collision)
        
        # Wait for camera initialization
        world.tick()
        while image_data["frame"] is None:
            time.sleep(0.01)
            world.tick()
        
        print("\n" + "="*50)
        print("? PILOTNET AUTONOMOUS DRIVING ACTIVE")
        print("Press 'Q' in camera window to quit")
        print("Press 'R' to respawn at new location")
        print("="*50)
        
        frame_count = 0
        start_time = time.time()
        last_respawn = time.time()
        
        while True:
            world.tick()
            frame_count += 1
            
            if image_data["frame"] is not None:
                frame = image_data["frame"]
                
                # Preprocess image for PilotNet
                input_image = cv2.resize(frame, (224, 224))
                input_tensor = torch.from_numpy(input_image).float().permute(2, 0, 1).unsqueeze(0).to(device)
                
                # Get model predictions
                with torch.no_grad():
                    outputs = model(input_tensor)
                    
                    steering = outputs['steering'].cpu().numpy()[0, 0]
                    throttle = outputs['throttle'].cpu().numpy()[0, 0]
                    brake = outputs['brake'].cpu().numpy()[0, 0]
                
                # Get vehicle state
                velocity = vehicle.get_velocity()
                speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2) * 3.6
                
                # Safety checks and adjustments
                # If speed is too low, increase throttle
                if speed < 10 and throttle < 0.3:
                    throttle = 0.5
                
                # If brake is high, reduce throttle
                if brake > 0.3:
                    throttle = max(0, throttle - 0.2)
                
                # Limit steering for stability
                steering = np.clip(steering, -0.8, 0.8)
                
                # Apply control to vehicle
                control = carla.VehicleControl()
                control.throttle = float(np.clip(throttle, 0, 1))
                control.steer = float(steering)
                control.brake = float(np.clip(brake, 0, 1))
                vehicle.apply_control(control)
                
                # Display
                display_frame = frame.copy()
                
                # Add overlay information
                y_offset = 30
                # Speed with color coding
                speed_color = (0, 255, 0) if speed > 10 else (0, 165, 255) if speed > 5 else (0, 0, 255)
                cv2.putText(display_frame, f"Speed: {speed:.1f} km/h", (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, speed_color, 2)
                
                y_offset += 35
                cv2.putText(display_frame, "PilotNet Predictions:", (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                y_offset += 30
                # Steering with color based on magnitude
                steer_color = (0, 255, 255) if abs(steering) > 0.3 else (255, 255, 0)
                cv2.putText(display_frame, f"  Steering: {steering:.3f}", (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, steer_color, 2)
                
                y_offset += 25
                cv2.putText(display_frame, f"  Throttle: {throttle:.3f}", (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                y_offset += 25
                cv2.putText(display_frame, f"  Brake: {brake:.3f}", (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                
                y_offset += 35
                cv2.putText(display_frame, f"Collisions: {collision_data['count']}", (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
                y_offset += 35
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                cv2.putText(display_frame, f"FPS: {fps:.1f} | Time: {elapsed:.1f}s", (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Add steering visualization
                center_x, center_y = 320, 400
                steer_length = int(abs(steering) * 150)
                if steering < 0:  # Left
                    cv2.arrowedLine(display_frame, (center_x, center_y), 
                                   (center_x - steer_length, center_y), (0, 255, 255), 3)
                    cv2.putText(display_frame, "LEFT", (center_x - 80, center_y + 25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                elif steering > 0:  # Right
                    cv2.arrowedLine(display_frame, (center_x, center_y), 
                                   (center_x + steer_length, center_y), (0, 255, 255), 3)
                    cv2.putText(display_frame, "RIGHT", (center_x + 20, center_y + 25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                cv2.imshow("PilotNet Autonomous Driving", display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):  # Respawn
                    if time.time() - last_respawn > 5:  # Prevent rapid respawning
                        print("? Respawning vehicle...")
                        try:
                            new_spawn = random.choice(spawn_points)
                            vehicle.set_transform(new_spawn)
                            collision_data["count"] = 0
                            last_respawn = time.time()
                            print(f"? Respawned at {new_spawn.location}")
                        except Exception as e:
                            print(f"? Respawn failed: {e}")
                
                # Status updates
                if frame_count % 100 == 0:
                    print(f"Time: {elapsed:.1f}s | FPS: {fps:.1f} | Speed: {speed:.1f}km/h | Collisions: {collision_data['count']}")
                
                # Auto-quit after long test
                if elapsed > 120:  # 2 minutes
                    print("? 2-minute test completed!")
                    break
        
        # Final statistics
        elapsed = time.time() - start_time
        avg_fps = frame_count / elapsed if elapsed > 0 else 0
        
        print("\n" + "="*50)
        print("? PILOTNET TEST COMPLETED")
        print("="*50)
        print(f"Duration: {elapsed:.1f} seconds")
        print(f"Frames processed: {frame_count}")
        print(f"Average FPS: {avg_fps:.1f}")
        print(f"Total collisions: {collision_data['count']}")
        print(f"Collision rate: {collision_data['count']/elapsed*60:.1f} per minute")
        
        if collision_data['count'] == 0:
            print("? PERFECT! No collisions!")
        elif collision_data['count'] < 5:
            print("? GOOD performance!")
        else:
            print("?? Could use improvement")
        
    finally:
        # Cleanup
        cv2.destroyAllWindows()
        try:
            collision_sensor.destroy()
            camera.destroy()
            vehicle.destroy()
        except:
            pass
        
        # Reset to async mode
        settings.synchronous_mode = False
        world.apply_settings(settings)

if __name__ == "__main__":
    test_pilotnet_realtime()