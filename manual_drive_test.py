"""
Simple Manual Driving Test
Test if the vehicle can move properly with manual control
"""
import torch
import numpy as np
from simple_working_model import ConditionalAffordanceNet
import cv2
import carla
import random
import time
import math

def manual_drive_test():
    """Test manual driving to verify vehicle movement"""
    
    print("Manual Driving Test - Use WASD keys to control")
    
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
        # Try different spawn points to find an open area
        spawn_points = world.get_map().get_spawn_points()
        vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
        
        print(f"Total spawn points available: {len(spawn_points)}")
        
        # Try spawn points that are likely to be in open areas
        good_spawn_indices = [0, 1, 50, 100, 150, 200, 250]
        
        vehicle = None
        for i in good_spawn_indices:
            try:
                if i < len(spawn_points):
                    spawn_point = spawn_points[i]
                    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
                    print(f"? Vehicle spawned at point {i}")
                    print(f"   Location: {spawn_point.location}")
                    print(f"   Rotation: {spawn_point.rotation}")
                    break
            except Exception as e:
                print(f"? Failed to spawn at point {i}: {e}")
                if vehicle:
                    vehicle.destroy()
                    vehicle = None
        
        if not vehicle:
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
        
        collision_count = 0
        def on_collision(event):
            nonlocal collision_count
            collision_count += 1
            other_actor = event.other_actor.type_id if event.other_actor else "unknown"
            print(f"Collision #{collision_count} with {other_actor}")
        
        collision_sensor.listen(on_collision)
        
        # Wait for first frame
        world.tick()
        while image_data["frame"] is None:
            time.sleep(0.01)
            world.tick()
        
        print("\n" + "="*50)
        print("MANUAL DRIVING CONTROLS:")
        print("W/S = Throttle/Brake")
        print("A/D = Steer Left/Right")
        print("Q = Quit")
        print("="*50)
        
        # Manual control variables
        throttle = 0.0
        steer = 0.0
        brake = 0.0
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            world.tick()
            frame_count += 1
            
            if image_data["frame"] is not None:
                frame = image_data["frame"]
                
                # Get vehicle state
                velocity = vehicle.get_velocity()
                speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2) * 3.6  # km/h
                location = vehicle.get_location()
                
                # Display
                display_frame = frame.copy()
                
                # Add control information
                y_offset = 30
                cv2.putText(display_frame, f"Speed: {speed:.1f} km/h", (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y_offset += 30
                cv2.putText(display_frame, f"Location: ({location.x:.1f}, {location.y:.1f})", (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y_offset += 30
                cv2.putText(display_frame, f"Control: T={throttle:.2f}, S={steer:.2f}, B={brake:.2f}", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                y_offset += 30
                cv2.putText(display_frame, f"Collisions: {collision_count}", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                y_offset += 30
                cv2.putText(display_frame, "W/S=Throttle/Brake, A/D=Steer, Q=Quit", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                cv2.imshow("Manual Driving Test", display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                # Reset controls
                throttle = 0.0
                steer = 0.0
                brake = 0.0
                
                if key == ord('w'):  # Forward
                    throttle = 0.8
                elif key == ord('s'):  # Brake/Reverse
                    brake = 0.5
                elif key == ord('a'):  # Steer left
                    steer = -0.5
                    throttle = 0.3  # Give some power while steering
                elif key == ord('d'):  # Steer right
                    steer = 0.5
                    throttle = 0.3  # Give some power while steering
                elif key == ord('q'):  # Quit
                    break
                
                # Apply control
                control = carla.VehicleControl()
                control.throttle = float(throttle)
                control.steer = float(steer)
                control.brake = float(brake)
                control.reverse = False
                vehicle.apply_control(control)
                
                # Status every 100 frames
                if frame_count % 100 == 0:
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed
                    print(f"Time: {elapsed:.1f}s | FPS: {fps:.1f} | Speed: {speed:.1f}km/h | Collisions: {collision_count}")
        
        # Final result
        elapsed = time.time() - start_time
        print(f"\nTest completed after {elapsed:.1f} seconds")
        print(f"Total collisions: {collision_count}")
        print(f"Max speed reached: {speed:.1f} km/h")
        
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
    manual_drive_test()