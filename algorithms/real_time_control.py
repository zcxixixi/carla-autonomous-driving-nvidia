"""Real-time CARLA Control with GPU Trained Model"""
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

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import carla
except ImportError:
    print("? CARLA module not found! Make sure CARLA is installed.")
    sys.exit(1)

from algorithms.cal_model import ConditionalAffordanceNet


class RealTimeController:
    """Real-time accident-aware driving controller"""
    
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
        
        # Control parameters
        self.current_image = None
        self.latest_prediction = None
        self.collision_count = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        # Command mapping
        self.cmd_map = {'left': 0, 'right': 1, 'straight': 2, 'follow': 3}
        self.current_command = 'straight'
        
        # Control history for smoothing
        self.throttle_history = []
        self.steer_history = []
        self.max_history = 5
        
    def connect_to_carla(self, host='localhost', port=2000):
        """Connect to CARLA server"""
        try:
            self.client = carla.Client(host, port)
            self.client.set_timeout(10.0)
            self.world = self.client.get_world()
            print(f"? Connected to CARLA server at {host}:{port}")
            
            # Set synchronous mode
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
        """Spawn ego vehicle"""
        try:
            # Get spawn points
            spawn_points = self.world.get_map().get_spawn_points()
            spawn_point = random.choice(spawn_points)
            
            # Get vehicle blueprint
            blueprint_library = self.world.get_blueprint_library()
            vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
            
            # Spawn vehicle
            self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
            print(f"? Vehicle spawned at {spawn_point.location}")
            
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
            
            # Camera transform (front of vehicle)
            camera_transform = carla.Transform(
                carla.Location(x=2.0, z=1.4),
                carla.Rotation(pitch=-15)
            )
            
            self.camera = self.world.spawn_actor(
                camera_bp, camera_transform, attach_to=self.vehicle
            )
            
            # Setup callback
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
        """Process camera image and update current_image"""
        try:
            # Convert CARLA image to PIL
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = array.reshape((image.height, image.width, 4))
            array = array[:, :, :3]  # Remove alpha channel
            
            # Convert to PIL Image
            pil_image = Image.fromarray(array)
            self.current_image = pil_image
            
        except Exception as e:
            print(f"?? Image processing error: {e}")
    
    def on_collision(self, event):
        """Handle collision events"""
        self.collision_count += 1
        print(f"? COLLISION! Total: {self.collision_count}")
        
        # Get collision details
        other_actor = event.other_actor
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        
        print(f"   Collided with: {other_actor.type_id}")
        print(f"   Impact intensity: {intensity:.2f}")
    
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
        
        # Base control
        control = carla.VehicleControl()
        
        # Throttle control based on front distance and risk
        if front_distance < 5.0 or risk_score > 0.8:
            # Emergency braking
            control.throttle = 0.0
            control.brake = 1.0
        elif front_distance < 10.0 or risk_score > 0.5:
            # Slow down
            control.throttle = 0.2
            control.brake = 0.3
        elif front_distance < 20.0:
            # Moderate speed
            control.throttle = 0.4
            control.brake = 0.0
        else:
            # Normal speed
            control.throttle = 0.6
            control.brake = 0.0
        
        # Steering based on lane offset
        steer = -lane_offset * 2.0  # Negative because of coordinate system
        steer = max(-1.0, min(1.0, steer))  # Clamp to [-1, 1]
        control.steer = steer
        
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
        """Run real-time control episode"""
        print(f"? Starting real-time control episode ({duration}s)")
        print("Commands: WASD to change direction, Q to quit")
        
        episode_start = time.time()
        last_print = 0
        
        try:
            while time.time() - episode_start < duration:
                # Tick the world
                self.world.tick()
                self.frame_count += 1
                
                # Change command randomly (simulate navigation)
                if self.frame_count % 100 == 0:  # Every 5 seconds
                    commands = ['straight', 'left', 'right', 'follow']
                    self.current_command = random.choice(commands)
                
                # Predict affordances
                front_dist, lane_offset, risk_score = self.predict_affordances()
                
                if front_dist is not None:
                    # Compute control
                    control = self.compute_control(front_dist, lane_offset, risk_score)
                    
                    # Apply control
                    self.vehicle.apply_control(control)
                    
                    # Print status every 2 seconds
                    if time.time() - last_print > 2.0:
                        elapsed = time.time() - episode_start
                        fps = self.frame_count / (time.time() - self.start_time)
                        
                        # Risk level
                        if risk_score > 0.7:
                            risk_level = "? HIGH RISK"
                        elif risk_score > 0.3:
                            risk_level = "? MEDIUM RISK"
                        else:
                            risk_level = "? LOW RISK"
                        
                        print(f"\\n??  Time: {elapsed:.1f}s | FPS: {fps:.1f}")
                        print(f"? Command: {self.current_command}")
                        print(f"? Front Distance: {front_dist:.2f}m")
                        print(f"? Lane Offset: {lane_offset:.3f}m")
                        print(f"??  Risk Level: {risk_level} ({risk_score:.3f})")
                        print(f"? Control: Throttle={control.throttle:.2f}, Steer={control.steer:.2f}")
                        print(f"? Collisions: {self.collision_count}")
                        
                        last_print = time.time()
                
                # Small delay to prevent overload
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            print("\\n??  Episode stopped by user")
        
        # Episode summary
        total_time = time.time() - episode_start
        avg_fps = self.frame_count / (time.time() - self.start_time)
        
        print(f"\\n? Episode Summary:")
        print(f"   Duration: {total_time:.1f}s")
        print(f"   Frames: {self.frame_count}")
        print(f"   Average FPS: {avg_fps:.1f}")
        print(f"   Collisions: {self.collision_count}")
        print(f"   Success Rate: {((total_time - self.collision_count*5) / total_time * 100):.1f}%")
    
    def cleanup(self):
        """Clean up resources"""
        try:
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
    print("? Real-time CARLA Control with GPU Model")
    print("=" * 50)
    
    controller = RealTimeController()
    
    try:
        # Connect to CARLA
        if not controller.connect_to_carla():
            return
        
        # Spawn vehicle
        if not controller.spawn_vehicle():
            return
        
        # Wait for camera to initialize
        print("? Waiting for camera...")
        time.sleep(2)
        
        # Run episode
        controller.run_episode(duration=30)  # 30 second test
        
    except Exception as e:
        print(f"? Error: {e}")
    finally:
        controller.cleanup()


if __name__ == '__main__':
    main()