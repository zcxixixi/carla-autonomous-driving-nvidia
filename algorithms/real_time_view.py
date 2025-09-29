"""Real-time CARLA Control with Live Camera View"""
import os
import sys
import time
import numpy as np
import torch
import cv2
from PIL import Image
import torchvision.transforms as transforms
import json
import random
import math
import threading
import queue

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import carla
except ImportError:
    print("? CARLA module not found! Make sure CARLA is installed.")
    sys.exit(1)

from algorithms.cal_model import ConditionalAffordanceNet


class RealTimeControllerWithView:
    """Real-time accident-aware driving controller with live camera view"""
    
    def __init__(self, model_path='checkpoints/gpu_trained_model.pt'):
        self.client = None
        self.world = None
        self.vehicle = None
        self.camera = None
        self.collision_sensor = None
        
        # Model setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"? Using device: {self.device}")
        
        # Load GPU trained model
        self.model = ConditionalAffordanceNet(
            feature_dim=256,
            affordance_dims={
                'front_distance': 1,
                'lane_offset': 1,
                'risk_score': 1
            }
        )
        
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"? Model loaded from {model_path}")
            print(f"? Training loss: {checkpoint['val_loss']:.4f}")
        else:
            print(f"? Model not found: {model_path}")
            sys.exit(1)
            
        self.model.to(self.device)
        self.model.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Display and control
        self.current_image = None
        self.display_image = None
        self.latest_prediction = None
        self.collision_count = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        # Image queue for display
        self.image_queue = queue.Queue(maxsize=5)
        self.display_thread = None
        self.display_running = False
        
        # Command mapping
        self.cmd_map = {'left': 0, 'right': 1, 'straight': 2, 'follow': 3}
        self.current_command = 'straight'
        
        # Control history for smoothing
        self.throttle_history = []
        self.steer_history = []
        self.max_history = 5
        
        # Statistics
        self.last_status_time = 0
        
    def connect_to_carla(self, host='localhost', port=2000):
        """Connect to CARLA server"""
        try:
            self.client = carla.Client(host, port)
            self.client.set_timeout(10.0)
            self.world = self.client.get_world()
            print(f"? Connected to CARLA server at {host}:{port}")
            
            # Set synchronous mode for better control
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05  # 20 FPS
            self.world.apply_settings(settings)
            print("? Synchronous mode enabled (20 FPS)")
            
            return True
        except Exception as e:
            print(f"? Failed to connect to CARLA: {e}")
            return False
    
    def spawn_vehicle(self):
        """Spawn ego vehicle at a good road location"""
        try:
            # Get spawn points and filter for road locations
            spawn_points = self.world.get_map().get_spawn_points()
            
            # Choose a spawn point on a main road (avoid vegetation areas)
            good_spawns = []
            for spawn in spawn_points:
                # Check if spawn point is on a road
                waypoint = self.world.get_map().get_waypoint(spawn.location)
                if waypoint and waypoint.lane_type == carla.LaneType.Driving:
                    good_spawns.append(spawn)
            
            if not good_spawns:
                spawn_point = random.choice(spawn_points)
            else:
                spawn_point = random.choice(good_spawns)
            
            # Get vehicle blueprint
            blueprint_library = self.world.get_blueprint_library()
            vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
            
            # Spawn vehicle
            self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
            print(f"? Vehicle spawned at road location")
            
            # Wait a moment for vehicle to settle
            time.sleep(1)
            
            # Setup camera
            self.setup_camera()
            
            # Setup collision sensor
            self.setup_collision_sensor()
            
            return True
        except Exception as e:
            print(f"? Failed to spawn vehicle: {e}")
            return False
    
    def setup_camera(self):
        """Setup RGB camera with higher resolution for display"""
        try:
            camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
            camera_bp.set_attribute('image_size_x', '800')
            camera_bp.set_attribute('image_size_y', '600')
            camera_bp.set_attribute('fov', '90')
            
            # Camera transform (front of vehicle, slightly higher)
            camera_transform = carla.Transform(
                carla.Location(x=2.0, z=1.8),
                carla.Rotation(pitch=-15)
            )
            
            self.camera = self.world.spawn_actor(
                camera_bp, camera_transform, attach_to=self.vehicle
            )
            
            # Setup callback
            self.camera.listen(self.process_image)
            print("? Camera setup complete (800x600)")
            
        except Exception as e:
            print(f"? Camera setup failed: {e}")
    
    def setup_collision_sensor(self):
        """Setup collision sensor"""
        try:
            collision_bp = self.world.get_blueprint_library().find('sensor.other.collision')
            self.collision_sensor = self.world.spawn_actor(
                collision_bp, carla.Transform(), attach_to=self.vehicle
            )
            self.collision_sensor.listen(self.on_collision)
            print("? Collision sensor setup complete")
        except Exception as e:
            print(f"? Collision sensor setup failed: {e}")
    
    def process_image(self, image):
        """Process camera image for both display and AI model"""
        try:
            # Convert CARLA image to numpy array
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = array.reshape((image.height, image.width, 4))
            rgb_array = array[:, :, :3]  # Remove alpha channel
            
            # Store for display (BGR for OpenCV)
            display_image = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
            
            # Add overlays with prediction info
            if self.latest_prediction:
                self.add_info_overlay(display_image)
            
            # Store for display thread
            if not self.image_queue.full():
                self.image_queue.put(display_image.copy())
            
            # Convert to PIL for AI model
            pil_image = Image.fromarray(rgb_array)
            self.current_image = pil_image
            
        except Exception as e:
            print(f"?? Image processing error: {e}")
    
    def add_info_overlay(self, image):
        """Add prediction information overlay to image"""
        try:
            pred = self.latest_prediction
            h, w = image.shape[:2]
            
            # Create semi-transparent overlay
            overlay = image.copy()
            
            # Info panel background
            cv2.rectangle(overlay, (10, 10), (400, 180), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
            
            # Text info
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            
            # Command
            cv2.putText(image, f"Command: {pred['command']}", (20, 40), 
                       font, font_scale, (255, 255, 255), thickness)
            
            # Distance
            color = (0, 255, 0) if pred['front_distance'] > 10 else (0, 255, 255) if pred['front_distance'] > 5 else (0, 0, 255)
            cv2.putText(image, f"Front Dist: {pred['front_distance']:.1f}m", (20, 70), 
                       font, font_scale, color, thickness)
            
            # Lane offset
            offset_color = (0, 255, 0) if abs(pred['lane_offset']) < 0.5 else (0, 255, 255)
            cv2.putText(image, f"Lane Offset: {pred['lane_offset']:.2f}m", (20, 100), 
                       font, font_scale, offset_color, thickness)
            
            # Risk score
            risk = pred['risk_score']
            if risk > 0.7:
                risk_color = (0, 0, 255)  # Red
                risk_text = "HIGH RISK"
            elif risk > 0.3:
                risk_color = (0, 255, 255)  # Yellow
                risk_text = "MEDIUM RISK"
            else:
                risk_color = (0, 255, 0)  # Green
                risk_text = "LOW RISK"
            
            cv2.putText(image, f"Risk: {risk:.3f} ({risk_text})", (20, 130), 
                       font, font_scale, risk_color, thickness)
            
            # Statistics
            fps = self.frame_count / max(time.time() - self.start_time, 1)
            cv2.putText(image, f"FPS: {fps:.1f} | Collisions: {self.collision_count}", (20, 160), 
                       font, 0.5, (255, 255, 255), 1)
            
            # Risk indicator bar
            bar_x, bar_y = w - 60, 50
            bar_height = int(100 * min(risk, 1.0))
            cv2.rectangle(image, (bar_x, bar_y + 100 - bar_height), (bar_x + 30, bar_y + 100), 
                         risk_color, -1)
            cv2.rectangle(image, (bar_x, bar_y), (bar_x + 30, bar_y + 100), (255, 255, 255), 2)
            cv2.putText(image, "RISK", (bar_x - 10, bar_y + 120), font, 0.4, (255, 255, 255), 1)
            
        except Exception as e:
            print(f"?? Overlay error: {e}")
    
    def display_thread_func(self):
        """Display thread function"""
        cv2.namedWindow('CARLA Vehicle View', cv2.WINDOW_AUTOSIZE)
        
        while self.display_running:
            try:
                # Get latest image
                if not self.image_queue.empty():
                    image = self.image_queue.get()
                    cv2.imshow('CARLA Vehicle View', image)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\\n?? Display window closed")
                    self.display_running = False
                    break
                elif key == ord('w'):
                    self.current_command = 'straight'
                    print("Command: straight")
                elif key == ord('a'):
                    self.current_command = 'left'
                    print("Command: left")
                elif key == ord('d'):
                    self.current_command = 'right'
                    print("Command: right")
                elif key == ord('s'):
                    self.current_command = 'follow'
                    print("Command: follow")
                
            except Exception as e:
                print(f"?? Display error: {e}")
                time.sleep(0.1)
        
        cv2.destroyAllWindows()
    
    def start_display(self):
        """Start display thread"""
        self.display_running = True
        self.display_thread = threading.Thread(target=self.display_thread_func)
        self.display_thread.daemon = True
        self.display_thread.start()
        print("?? Live camera view started (Press Q to close, WASD to change commands)")
    
    def stop_display(self):
        """Stop display thread"""
        self.display_running = False
        if self.display_thread:
            self.display_thread.join(timeout=2)
    
    def on_collision(self, event):
        """Handle collision events"""
        self.collision_count += 1
        
        # Only print major collisions to avoid spam
        other_actor = event.other_actor
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        
        if intensity > 100:  # Major collision
            print(f"? COLLISION #{self.collision_count} - {other_actor.type_id} (intensity: {intensity:.1f})")
    
    def predict_affordances(self):
        """Use model to predict affordances"""
        if self.current_image is None:
            return None, None, None
        
        try:
            # Preprocess image
            image_tensor = self.transform(self.current_image).unsqueeze(0).to(self.device)
            
            # Get command
            command_idx = self.cmd_map.get(self.current_command, 2)
            command_tensor = torch.tensor([command_idx], dtype=torch.long).to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(image_tensor, command_tensor)
            
            # Extract predictions
            front_distance = outputs['front_distance'].item()
            lane_offset = outputs['lane_offset'].item()
            risk_score = outputs['risk_score'].item()
            
            self.latest_prediction = {
                'front_distance': front_distance,
                'lane_offset': lane_offset,
                'risk_score': risk_score,
                'command': self.current_command
            }
            
            return front_distance, lane_offset, risk_score
            
        except Exception as e:
            print(f"?? Prediction error: {e}")
            return None, None, None
    
    def compute_control(self, front_distance, lane_offset, risk_score):
        """Compute vehicle control based on affordances"""
        if front_distance is None:
            return carla.VehicleControl()
        
        control = carla.VehicleControl()
        
        # Improved control logic
        if front_distance < 3.0 or risk_score > 0.8:
            # Emergency stop
            control.throttle = 0.0
            control.brake = 1.0
        elif front_distance < 8.0 or risk_score > 0.5:
            # Slow down
            control.throttle = max(0.0, 0.3 - risk_score * 0.2)
            control.brake = 0.4
        elif front_distance < 15.0:
            # Moderate speed
            control.throttle = 0.5
            control.brake = 0.0
        else:
            # Normal speed
            control.throttle = 0.7
            control.brake = 0.0
        
        # Improved steering with command consideration
        base_steer = -lane_offset * 1.5  # Lane keeping
        
        # Add command-based steering
        if self.current_command == 'left':
            base_steer -= 0.3
        elif self.current_command == 'right':
            base_steer += 0.3
        
        # Clamp steering
        control.steer = max(-1.0, min(1.0, base_steer))
        
        # Smooth controls
        self.throttle_history.append(control.throttle)
        self.steer_history.append(control.steer)
        
        if len(self.throttle_history) > self.max_history:
            self.throttle_history.pop(0)
            self.steer_history.pop(0)
        
        # Apply smoothing
        control.throttle = sum(self.throttle_history) / len(self.throttle_history)
        control.steer = sum(self.steer_history) / len(self.steer_history)
        
        return control
    
    def run_episode(self, duration=60):
        """Run real-time control episode with display"""
        print(f"? Starting real-time control with live view ({duration}s)")
        print("? Camera window controls:")
        print("   Q - Quit")
        print("   W - Straight")
        print("   A - Left")
        print("   D - Right") 
        print("   S - Follow")
        
        # Start display
        self.start_display()
        
        episode_start = time.time()
        self.last_status_time = episode_start
        
        try:
            while time.time() - episode_start < duration and self.display_running:
                # Tick the world
                self.world.tick()
                self.frame_count += 1
                
                # Predict affordances
                front_dist, lane_offset, risk_score = self.predict_affordances()
                
                if front_dist is not None:
                    # Compute and apply control
                    control = self.compute_control(front_dist, lane_offset, risk_score)
                    self.vehicle.apply_control(control)
                    
                    # Print status every 5 seconds
                    if time.time() - self.last_status_time > 5.0:
                        elapsed = time.time() - episode_start
                        fps = self.frame_count / (time.time() - self.start_time)
                        
                        print(f"\\n??  Time: {elapsed:.1f}s | FPS: {fps:.1f}")
                        print(f"? Command: {self.current_command}")
                        print(f"? Distance: {front_dist:.1f}m | Offset: {lane_offset:.2f}m | Risk: {risk_score:.3f}")
                        print(f"? Throttle: {control.throttle:.2f} | Steer: {control.steer:.2f}")
                        print(f"? Collisions: {self.collision_count}")
                        
                        self.last_status_time = time.time()
                
                # Small delay
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            print("\\n?? Episode stopped by user")
        
        # Stop display
        self.stop_display()
        
        # Episode summary
        total_time = time.time() - episode_start
        avg_fps = self.frame_count / (time.time() - self.start_time)
        
        print(f"\\n? Episode Summary:")
        print(f"   Duration: {total_time:.1f}s")
        print(f"   Frames: {self.frame_count}")
        print(f"   Average FPS: {avg_fps:.1f}")
        print(f"   Collisions: {self.collision_count}")
        
        # Calculate success rate (less than 1 collision per 10 seconds is good)
        collision_rate = self.collision_count / total_time * 10
        if collision_rate < 1:
            print(f"   ? Success! Low collision rate: {collision_rate:.2f}/10s")
        else:
            print(f"   ?? High collision rate: {collision_rate:.2f}/10s")
    
    def cleanup(self):
        """Clean up resources"""
        try:
            self.stop_display()
            
            if self.camera:
                self.camera.destroy()
            if self.collision_sensor:
                self.collision_sensor.destroy()
            if self.vehicle:
                self.vehicle.destroy()
            
            # Restore async mode
            if self.world:
                settings = self.world.get_settings()
                settings.synchronous_mode = False
                self.world.apply_settings(settings)
            
            print("? Cleanup complete")
        except Exception as e:
            print(f"?? Cleanup error: {e}")


def main():
    """Main function"""
    print("? Real-time CARLA Control with Live Camera View")
    print("=" * 60)
    
    controller = RealTimeControllerWithView()
    
    try:
        # Connect to CARLA
        if not controller.connect_to_carla():
            return
        
        # Spawn vehicle
        if not controller.spawn_vehicle():
            return
        
        # Wait for camera to initialize
        print("? Waiting for camera initialization...")
        time.sleep(3)
        
        # Run episode
        controller.run_episode(duration=60)  # 60 second test
        
    except Exception as e:
        print(f"? Error: {e}")
    finally:
        controller.cleanup()


if __name__ == '__main__':
    main()