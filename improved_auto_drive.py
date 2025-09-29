"""
Improved Auto-Driving with Better Control Logic
Based on model predictions, but with smarter steering/throttle control
"""
import torch
import numpy as np
from simple_working_model import ConditionalAffordanceNet
import cv2
import carla
import random
import time
import math

def improved_auto_drive():
    """Auto-driving with improved control logic based on model predictions"""
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    model = ConditionalAffordanceNet().to(device)
    checkpoint = torch.load('checkpoints/gpu_trained_model.pt', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    loss_val = checkpoint.get('loss', checkpoint.get('val_loss', 'unknown'))
    print(f"GPU model loaded (loss: {loss_val})")
    
    # Connect to CARLA
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    world.set_weather(carla.WeatherParameters.ClearNoon)
    
    # Synchronous mode
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)
    
    blueprint_library = world.get_blueprint_library()
    
    try:
        # Spawn vehicle at a better location (try multiple points)
        spawn_points = world.get_map().get_spawn_points()
        vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
        
        # Try different spawn points to avoid being stuck
        vehicle = None
        for i in [10, 50, 100, 150, 0, 1, 2]:
            try:
                spawn_point = spawn_points[i % len(spawn_points)]
                vehicle = world.spawn_actor(vehicle_bp, spawn_point)
                print(f"Vehicle spawned at point {i}: {spawn_point.location}")
                break
            except Exception as e:
                print(f"Failed to spawn at point {i}: {e}")
                if i == 2:  # Last attempt
                    raise Exception("Could not spawn vehicle at any location")
        
        # Setup camera
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
            array = array[:, :, :3]
            image_data["frame"] = array.copy()
        
        camera.listen(process_image)
        
        # Collision sensor
        collision_bp = blueprint_library.find('sensor.other.collision')
        collision_sensor = world.spawn_actor(collision_bp, carla.Transform(), attach_to=vehicle)
        
        collision_data = {"count": 0, "history": []}
        def on_collision(event):
            collision_data["count"] += 1
            collision_data["history"].append(time.time())
            other_actor = event.other_actor.type_id if event.other_actor else "unknown"
            print(f"Collision #{collision_data['count']} with {other_actor}")
        
        collision_sensor.listen(on_collision)
        
        # Wait for first frame
        world.tick()
        while image_data["frame"] is None:
            time.sleep(0.01)
            world.tick()
        
        print("Connected to CARLA (sync mode)")
        print("Vehicle spawned and ready")
        print("Starting improved auto-driving test")
        print("Press Q in camera window to quit")
        
        # Control variables
        command_mapping = {'straight': 0, 'follow': 1, 'left': 2, 'right': 3}
        current_command = 'straight'
        frame_count = 0
        start_time = time.time()
        
        # Control state
        last_steer = 0.0
        stuck_counter = 0
        last_velocity = 0.0
        
        while True:
            world.tick()
            frame_count += 1
            
            if image_data["frame"] is not None:
                frame = image_data["frame"]
                
                # Get vehicle state
                velocity = vehicle.get_velocity()
                speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2) * 3.6  # km/h
                
                # Preprocess image
                rgb_resized = cv2.resize(frame, (224, 224))
                rgb_normalized = rgb_resized.astype(np.float32) / 255.0
                rgb_tensor = torch.from_numpy(rgb_normalized).permute(2, 0, 1).unsqueeze(0).to(device)
                
                # Predict with current command
                cmd_tensor = torch.tensor([command_mapping[current_command]], dtype=torch.long).to(device)
                
                with torch.no_grad():
                    distance, lane_offset, risk = model(rgb_tensor, cmd_tensor)
                
                dist_val = distance.cpu().numpy()[0, 0]
                lane_val = lane_offset.cpu().numpy()[0, 0]
                risk_val = risk.cpu().numpy()[0, 0]
                
                # IMPROVED CONTROL LOGIC - Fixed to ensure movement
                
                # 1. Throttle based on distance and risk (more aggressive driving)
                if speed < 5.0:  # Always accelerate when too slow
                    throttle = 0.8
                    brake = 0.0
                elif dist_val < 5.0 and risk_val > 0.5:  # Only brake for very close/high risk
                    throttle = 0.0
                    brake = 0.4
                elif dist_val < 10.0:  # Slow down for moderate distance
                    throttle = 0.3
                    brake = 0.0
                elif speed < 30:  # Keep accelerating until good speed
                    throttle = 0.6
                    brake = 0.0
                else:  # Maintain speed
                    throttle = 0.4
                    brake = 0.0
                
                # 2. Steering based on lane offset and command
                target_steer = 0.0
                
                if current_command == 'left':
                    target_steer = -0.3
                elif current_command == 'right':
                    target_steer = 0.3
                elif current_command in ['straight', 'follow']:
                    # Lane centering with lane offset
                    target_steer = np.clip(lane_val * 2.0, -0.5, 0.5)  # Proportional control
                
                # 3. Smooth steering to avoid oscillation
                steer = last_steer * 0.7 + target_steer * 0.3
                steer = np.clip(steer, -1.0, 1.0)
                last_steer = steer
                
                # 4. Anti-stuck mechanism (more aggressive)
                if speed < 1.0:  # Vehicle might be stuck
                    stuck_counter += 1
                    if stuck_counter > 10:  # Stuck for 10 frames (0.5 second)
                        print(f"Vehicle stuck (speed: {speed:.1f}) - emergency maneuver...")
                        # Try aggressive forward movement first
                        throttle = 1.0
                        brake = 0.0
                        steer = random.choice([-0.6, 0.6])  # Sharp turn
                        
                        if stuck_counter > 30:  # If still stuck after aggressive forward
                            print("Trying reverse...")
                            control = carla.VehicleControl()
                            control.throttle = 0.0
                            control.steer = float(steer)
                            control.brake = 0.0
                            control.reverse = True
                            vehicle.apply_control(control)
                            
                            # Wait briefly
                            for _ in range(5):
                                world.tick()
                            
                            stuck_counter = 0
                            continue
                else:
                    stuck_counter = 0
                
                # 5. Command switching logic
                if frame_count % 60 == 0:  # Every 3 seconds, consider changing command
                    # Test all commands and pick the best one
                    best_command = current_command
                    best_score = -float('inf')
                    
                    for cmd_name in ['straight', 'follow', 'left', 'right']:
                        cmd_tensor = torch.tensor([command_mapping[cmd_name]], dtype=torch.long).to(device)
                        
                        with torch.no_grad():
                            d, l, r = model(rgb_tensor, cmd_tensor)
                        
                        d_val = d.cpu().numpy()[0, 0]
                        r_val = r.cpu().numpy()[0, 0]
                        
                        # Score: prioritize safety (low risk) and good distance
                        score = d_val * 0.5 - r_val * 10.0  # Distance good, risk bad
                        
                        if score > best_score:
                            best_score = score
                            best_command = cmd_name
                    
                    if best_command != current_command:
                        print(f"Switching command: {current_command} -> {best_command}")
                        current_command = best_command
                
                # Apply control
                control = carla.VehicleControl()
                control.throttle = float(throttle)
                control.steer = float(steer)
                control.brake = float(brake)
                control.reverse = False
                vehicle.apply_control(control)
                
                # Display
                display_frame = frame.copy()
                
                # Overlay information with more debugging details
                y_offset = 30
                # Speed with color coding
                speed_color = (0, 255, 0) if speed > 5 else (0, 165, 255) if speed > 1 else (0, 0, 255)
                cv2.putText(display_frame, f"Speed: {speed:.1f} km/h", (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, speed_color, 2)
                y_offset += 30
                cv2.putText(display_frame, f"Command: {current_command}", (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y_offset += 30
                cv2.putText(display_frame, f"AI: Dist={dist_val:.1f}m, Lane={lane_val:.2f}m, Risk={risk_val:.3f}", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                y_offset += 30
                # Control with color coding
                control_color = (0, 255, 0) if throttle > 0.1 else (255, 255, 0) if brake > 0.1 else (0, 0, 255)
                cv2.putText(display_frame, f"Control: T={throttle:.2f}, S={steer:.2f}, B={brake:.2f}", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, control_color, 2)
                y_offset += 30
                cv2.putText(display_frame, f"Collisions: {collision_data['count']}", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                y_offset += 30
                # Add stuck counter
                if stuck_counter > 0:
                    cv2.putText(display_frame, f"Stuck Counter: {stuck_counter}", 
                               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                
                cv2.imshow("Improved Auto-Driving", display_frame)
                
                # Status every 5 seconds
                if frame_count % 100 == 0:
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed
                    print(f"Time: {elapsed:.1f}s | FPS: {fps:.1f} | Speed: {speed:.1f}km/h | Collisions: {collision_data['count']}")
                
                # Check for quit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
            
            # Auto-quit after 60 seconds
            if time.time() - start_time > 60:
                print("60 second test completed!")
                break
        
        # Final statistics
        elapsed = time.time() - start_time
        collision_rate = collision_data['count'] / elapsed * 60  # collisions per minute
        
        print("\n" + "="*50)
        print("IMPROVED AUTO-DRIVING TEST RESULTS")
        print("="*50)
        print(f"Duration: {elapsed:.1f} seconds")
        print(f"Frames processed: {frame_count}")
        print(f"Average FPS: {frame_count/elapsed:.1f}")
        print(f"Total collisions: {collision_data['count']}")
        print(f"Collision rate: {collision_rate:.1f} collisions/minute")
        
        if collision_data['count'] == 0:
            print("? PERFECT! No collisions!")
        elif collision_rate < 5:
            print("? GOOD: Low collision rate")
        elif collision_rate < 15:
            print("??  OKAY: Moderate collision rate")
        else:
            print("? POOR: High collision rate - needs improvement")
        
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
    improved_auto_drive()