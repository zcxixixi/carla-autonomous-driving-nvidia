"""Smart Road-only CARLA Controller with Live View"""
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


class SmartRoadController:
    """Smart road-only controller with obstacle avoidance"""
    
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
        self.latest_prediction = None
        self.collision_count = 0
        self.major_collision_count = 0
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
        self.max_history = 3
        
        # Statistics
        self.last_status_time = 0
        self.last_teleport_time = 0
        
    def connect_to_carla(self, host='localhost', port=2000):
        """Connect to CARLA server"""
        try:
            self.client = carla.Client(host, port)
            self.client.set_timeout(10.0)
            self.world = self.client.get_world()
            print(f"? Connected to CARLA server at {host}:{port}")
            
            # Load a simpler map if needed
            current_map = self.world.get_map().name
            print(f"? Current map: {current_map}")
            
            # Try to load a simple map for better performance
            available_maps = self.client.get_available_maps()
            simple_maps = [m for m in available_maps if 'Town01' in m or 'Town02' in m]
            
            if simple_maps and 'Town10' in current_map:
                print(f"? Switching to simpler map: {simple_maps[0]}")
                self.client.load_world(simple_maps[0])
                self.world = self.client.get_world()
                time.sleep(2)  # Wait for map to load
            
            # Set synchronous mode
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05  # 20 FPS
            settings.no_rendering_mode = False  # Enable rendering
            self.world.apply_settings(settings)
            print("? Synchronous mode enabled (20 FPS)")
            
            return True
        except Exception as e:
            print(f"? Failed to connect to CARLA: {e}")
            return False
    
    def find_safe_spawn_point(self):
        """Find a safe spawn point on a main road"""
        spawn_points = self.world.get_map().get_spawn_points()
        map_obj = self.world.get_map()
        
        safe_spawns = []
        
        for spawn in spawn_points:
            try:
                # Get waypoint at spawn location
                waypoint = map_obj.get_waypoint(spawn.location)
                
                if waypoint and waypoint.lane_type == carla.LaneType.Driving:
                    # Check if it's a main road (wider lanes are usually main roads)
                    if waypoint.lane_width > 3.0:
                        # Check if there are no obstacles nearby
                        location = spawn.location
                        location.z += 2.0  # Raise a bit to check above ground
                        
                        # Simple obstacle check - no collision detection needed
                        safe_spawns.append((spawn, waypoint.lane_width))
                        
            except Exception:
                continue
        
        if safe_spawns:
            # Sort by lane width (prefer wider roads)
            safe_spawns.sort(key=lambda x: x[1], reverse=True)
            best_spawn = safe_spawns[0][0]
            print(f"?? Selected spawn on wide road (width: {safe_spawns[0][1]:.1f}m)")
            return best_spawn
        else:
            # Fallback to any spawn point
            return random.choice(spawn_points)
    
    def spawn_vehicle(self):
        """Spawn ego vehicle at a safe road location"""
        try:
            # Find safe spawn point
            spawn_point = self.find_safe_spawn_point()
            
            # Get vehicle blueprint
            blueprint_library = self.world.get_blueprint_library()
            vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
            
            # Spawn vehicle
            self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
            print(f"? Vehicle spawned on main road")
            
            # Wait for vehicle to settle
            time.sleep(1)
            self.world.tick()
            
            # Setup camera
            self.setup_camera()
            
            # Setup collision sensor
            self.setup_collision_sensor()
            
            return True
        except Exception as e:
            print(f"? Failed to spawn vehicle: {e}")
            return False
    
    def setup_camera(self):
        """Setup RGB camera"""
        try:
            camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
            camera_bp.set_attribute('image_size_x', '800')
            camera_bp.set_attribute('image_size_y', '600')
            camera_bp.set_attribute('fov', '90')
            
            # Camera transform
            camera_transform = carla.Transform(
                carla.Location(x=2.0, z=1.8),
                carla.Rotation(pitch=-15)
            )
            
            self.camera = self.world.spawn_actor(
                camera_bp, camera_transform, attach_to=self.vehicle
            )
            
            self.camera.listen(self.process_image)
            print("? Camera setup complete")
            
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
        """Process camera image"""
        try:
            # Convert CARLA image to numpy array
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = array.reshape((image.height, image.width, 4))
            rgb_array = array[:, :, :3]
            
            # Store for display (BGR for OpenCV)
            display_image = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
            
            # Add overlays
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
        """Add prediction information overlay"""
        try:
            pred = self.latest_prediction
            h, w = image.shape[:2]
            
            # Semi-transparent panel
            overlay = image.copy()
            cv2.rectangle(overlay, (10, 10), (400, 200), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.8, image, 0.2, 0, image)
            
            # Text info
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            # Status info
            cv2.putText(image, f"GPU Model Active", (20, 35), font, 0.6, (0, 255, 0), 2)
            cv2.putText(image, f"Command: {pred['command']}", (20, 65), font, 0.6, (255, 255, 255), 2)
            
            # Distance with color coding
            dist = pred['front_distance']
            if dist < 5:
                dist_color = (0, 0, 255)  # Red
            elif dist < 15:
                dist_color = (0, 255, 255)  # Yellow
            else:
                dist_color = (0, 255, 0)  # Green
            
            cv2.putText(image, f"Front: {dist:.1f}m", (20, 95), font, 0.6, dist_color, 2)
            
            # Lane offset
            offset = pred['lane_offset']
            offset_color = (0, 255, 0) if abs(offset) < 1.0 else (0, 255, 255)
            cv2.putText(image, f"Lane: {offset:.2f}m", (20, 125), font, 0.6, offset_color, 2)
            
            # Risk assessment
            risk = pred['risk_score']
            if risk > 0.7:
                risk_color = (0, 0, 255)
                risk_text = "HIGH"
            elif risk > 0.3:
                risk_color = (0, 255, 255)
                risk_text = "MED"
            else:
                risk_color = (0, 255, 0)
                risk_text = "LOW"
            
            cv2.putText(image, f"Risk: {risk:.3f} ({risk_text})", (20, 155), font, 0.6, risk_color, 2)
            
            # Statistics
            fps = self.frame_count / max(time.time() - self.start_time, 1)
            cv2.putText(image, f"FPS: {fps:.1f} | Collisions: {self.major_collision_count}", 
                       (20, 185), font, 0.5, (255, 255, 255), 1)
            
        except Exception as e:
            print(f"?? Overlay error: {e}")
    
    def display_thread_func(self):
        """Display thread function"""
        cv2.namedWindow('CARLA AI Driver View', cv2.WINDOW_AUTOSIZE)
        
        while self.display_running:
            try:
                if not self.image_queue.empty():
                    image = self.image_queue.get()
                    cv2.imshow('CARLA AI Driver View', image)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\\n?? Stopping via display window")
                    self.display_running = False
                    break
                elif key == ord('w'):
                    self.current_command = 'straight'
                elif key == ord('a'):
                    self.current_command = 'left'
                elif key == ord('d'):
                    self.current_command = 'right'
                elif key == ord('s'):
                    self.current_command = 'follow'
                elif key == ord('r'):
                    # Reset vehicle position if stuck
                    self.teleport_to_road()
                
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
        print("?? Live view started (Q=quit, WASD=commands, R=reset position)")
    
    def stop_display(self):
        """Stop display thread"""
        self.display_running = False
        if self.display_thread:
            self.display_thread.join(timeout=2)
    
    def teleport_to_road(self):
        """Teleport vehicle to a safe road location"""
        current_time = time.time()
        if current_time - self.last_teleport_time < 5.0:  # Cooldown
            return
        
        try:
            spawn_point = self.find_safe_spawn_point()
            self.vehicle.set_transform(spawn_point)
            print("? Vehicle teleported to safe location")
            self.last_teleport_time = current_time
        except Exception as e:
            print(f"?? Teleport failed: {e}")
    
    def on_collision(self, event):
        """Handle collision events"""
        self.collision_count += 1
        
        other_actor = event.other_actor
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        
        # Only count major collisions
        if intensity > 500:  # Significant collision
            self.major_collision_count += 1
            print(f"? Major collision #{self.major_collision_count} - {other_actor.type_id}")
            
            # Auto-reset if stuck
            if self.collision_count > 50:  # Too many small collisions
                self.teleport_to_road()
                self.collision_count = 0
    
    def predict_affordances(self):
        """Predict affordances using AI model"""
        if self.current_image is None:
            return None, None, None
        
        try:
            image_tensor = self.transform(self.current_image).unsqueeze(0).to(self.device)
            command_idx = self.cmd_map.get(self.current_command, 2)
            command_tensor = torch.tensor([command_idx], dtype=torch.long).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(image_tensor, command_tensor)
            
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
        """Compute smart vehicle control"""
        if front_distance is None:
            return carla.VehicleControl()
        
        control = carla.VehicleControl()
        
        # Emergency braking
        if front_distance < 2.0 or risk_score > 0.9:
            control.throttle = 0.0
            control.brake = 1.0
            control.steer = 0.0
        # Cautious driving
        elif front_distance < 8.0 or risk_score > 0.4:
            control.throttle = 0.2
            control.brake = 0.3
        # Normal driving
        elif front_distance > 20.0 and risk_score < 0.2:
            control.throttle = 0.6
            control.brake = 0.0
        # Moderate driving
        else:
            control.throttle = 0.4
            control.brake = 0.0
        
        # Smart steering
        base_steer = -lane_offset * 0.8  # Gentler lane keeping
        
        # Command-based steering
        if self.current_command == 'left':
            base_steer -= 0.2
        elif self.current_command == 'right':
            base_steer += 0.2
        
        control.steer = max(-0.8, min(0.8, base_steer))  # Limit steering
        
        # Smooth controls
        self.throttle_history.append(control.throttle)
        self.steer_history.append(control.steer)
        
        if len(self.throttle_history) > self.max_history:
            self.throttle_history.pop(0)
            self.steer_history.pop(0)
        
        control.throttle = sum(self.throttle_history) / len(self.throttle_history)
        control.steer = sum(self.steer_history) / len(self.steer_history)
        
        return control
    
    def run_episode(self, duration=120):
        """Run smart driving episode"""
        print(f"? Starting smart AI driving ({duration}s)")
        print("?? Watch the camera window for live view!")
        
        # Start display
        self.start_display()
        
        episode_start = time.time()
        self.last_status_time = episode_start
        
        try:
            while time.time() - episode_start < duration and self.display_running:
                # Tick world
                self.world.tick()
                self.frame_count += 1
                
                # AI prediction and control
                front_dist, lane_offset, risk_score = self.predict_affordances()
                
                if front_dist is not None:
                    control = self.compute_control(front_dist, lane_offset, risk_score)
                    self.vehicle.apply_control(control)
                    
                    # Status update every 5 seconds
                    if time.time() - self.last_status_time > 5.0:
                        elapsed = time.time() - episode_start
                        fps = self.frame_count / (time.time() - self.start_time)
                        
                        print(f"\\n??  {elapsed:.1f}s | FPS: {fps:.1f} | Command: {self.current_command}")
                        print(f"? Dist: {front_dist:.1f}m | Risk: {risk_score:.3f} | Major Collisions: {self.major_collision_count}")
                        
                        self.last_status_time = time.time()
                
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            print("\\n?? Stopped by user")
        
        self.stop_display()
        
        # Final summary
        total_time = time.time() - episode_start
        success_rate = max(0, 100 - (self.major_collision_count * 10))
        
        print(f"\\n? Episode Complete!")
        print(f"? Duration: {total_time:.1f}s")
        print(f"? Major Collisions: {self.major_collision_count}")
        print(f"? Success Rate: {success_rate:.1f}%")
        
        if self.major_collision_count < 3:
            print("? Excellent driving performance!")
        elif self.major_collision_count < 10:
            print("? Good performance with room for improvement")
        else:
            print("?? Performance needs improvement")
    
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
            
            if self.world:
                settings = self.world.get_settings()
                settings.synchronous_mode = False
                self.world.apply_settings(settings)
            
            print("? Cleanup complete")
        except Exception as e:
            print(f"?? Cleanup error: {e}")


def main():
    """Main function"""
    print("? Smart CARLA AI Driver with Live View")
    print("=" * 50)
    
    controller = SmartRoadController()
    
    try:
        if not controller.connect_to_carla():
            return
        
        if not controller.spawn_vehicle():
            return
        
        print("? Initializing camera...")
        time.sleep(3)
        
        controller.run_episode(duration=120)  # 2 minute test
        
    except Exception as e:
        print(f"? Error: {e}")
    finally:
        controller.cleanup()


if __name__ == '__main__':
    main()