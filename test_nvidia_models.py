# -*- coding: utf-8 -*-
"""
NVIDIA Advanced Models Real-time Testing in CARLA
Test Hydra-MDP and OmniDrive models in live CARLA environment
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
import argparse

# Add models directory to path
sys.path.append('models')
from hydra_mdp import HydraMDP
from omnidrive import OmniDrive

class NVIDIAModelTester:
    """Real-time tester for NVIDIA advanced models"""
    
    def __init__(self, model_type='hydra', device='cuda'):
        self.model_type = model_type
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        print(f"? Initializing NVIDIA {model_type.upper()} Model")
        print(f"Device: {self.device}")
        
        # Load model
        if model_type == 'hydra':
            self.model = HydraMDP().to(self.device)
            print("? Hydra-MDP model loaded")
        elif model_type == 'omnidrive':
            self.model = OmniDrive().to(self.device)
            print("? OmniDrive model loaded")
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.model.eval()
        
        # Model statistics
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"? Model parameters: {total_params:,}")
        
        # Initialize CARLA
        self.client = None
        self.world = None
        self.vehicle = None
        self.camera = None
        
    def connect_carla(self):
        """Connect to CARLA simulator"""
        try:
            self.client = carla.Client('localhost', 2000)
            self.client.set_timeout(10.0)
            self.world = self.client.get_world()
            self.world.set_weather(carla.WeatherParameters.ClearNoon)
            
            # Synchronous mode
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05  # 20 FPS
            self.world.apply_settings(settings)
            
            print("? Connected to CARLA")
            return True
        except Exception as e:
            print(f"? Failed to connect to CARLA: {e}")
            return False
    
    def spawn_vehicle_and_camera(self):
        """Spawn vehicle and camera sensor"""
        try:
            blueprint_library = self.world.get_blueprint_library()
            spawn_points = self.world.get_map().get_spawn_points()
            
            # Spawn vehicle
            vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
            spawn_point = random.choice(spawn_points)
            self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
            
            # Setup camera
            camera_bp = blueprint_library.find('sensor.camera.rgb')
            camera_bp.set_attribute('image_size_x', '224')
            camera_bp.set_attribute('image_size_y', '224')
            camera_bp.set_attribute('fov', '90')
            
            camera_transform = carla.Transform(carla.Location(x=2.0, z=1.4))
            self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
            
            print("? Vehicle and camera spawned")
            return True
        except Exception as e:
            print(f"? Failed to spawn vehicle: {e}")
            return False
    
    def process_image(self, carla_image):
        """Convert CARLA image to model input tensor"""
        array = np.frombuffer(carla_image.raw_data, dtype=np.uint8)
        array = array.reshape((carla_image.height, carla_image.width, 4))
        array = array[:, :, :3]  # Remove alpha channel
        
        # Convert to tensor
        tensor = torch.from_numpy(array).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
        return array, tensor
    
    def predict_actions(self, image_tensor):
        """Get model predictions"""
        with torch.no_grad():
            if self.model_type == 'hydra':
                outputs = self.model(image_tensor, training=False)
                return {
                    'steering': outputs['steering'].cpu().numpy()[0, 0],
                    'throttle': outputs['throttle'].cpu().numpy()[0, 0],
                    'brake': outputs['brake'].cpu().numpy()[0, 0],
                    'speed': outputs['speed'].cpu().numpy()[0, 0],
                    'waypoints': outputs['waypoints'].cpu().numpy()[0],
                    'safety_score': outputs['safety']['safety_score'].cpu().numpy()[0, 0],
                    'collision_risk': outputs['safety']['collision_risk'].cpu().numpy()[0, 0]
                }
            elif self.model_type == 'omnidrive':
                # Create dummy scene tokens for OmniDrive
                scene_tokens = torch.randint(0, 100, (1, 5)).to(self.device)  # 5 scene tokens
                outputs = self.model(image_tensor, scene_tokens, return_reasoning=True)
                return {
                    'steering': outputs['steering'].cpu().numpy()[0, 0],
                    'throttle': outputs['throttle'].cpu().numpy()[0, 0],
                    'brake': outputs['brake'].cpu().numpy()[0, 0],
                    'speed': outputs['speed'].cpu().numpy()[0, 0],
                    'trajectory': outputs['trajectory'].cpu().numpy()[0],
                    'risk_assessment': outputs['risk_assessment'].cpu().numpy()[0].mean(),
                    'scene_reasoning': outputs['scene_reasoning'].cpu().numpy()[0][:5]  # First 5 values
                }
    
    def apply_control(self, predictions):
        """Apply model predictions to vehicle"""
        steering = float(np.clip(predictions['steering'], -1.0, 1.0))
        throttle = float(np.clip(predictions['throttle'], 0.0, 1.0))
        brake = float(np.clip(predictions['brake'], 0.0, 1.0))
        
        # Get vehicle state
        velocity = self.vehicle.get_velocity()
        speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2) * 3.6
        location = self.vehicle.get_location()
        
        # FORCE MOVEMENT - ensure vehicle moves!
        # If model prediction throttle is too small, force increase
        if throttle < 0.2:
            throttle = 0.5  # Force throttle
            print(f"WARNING: Force throttle: {throttle:.2f} (original: {predictions['throttle']:.3f})")
        
        # If speed is too low, further increase throttle
        if speed < 10:
            throttle = max(throttle, 0.6)
            brake = 0.0  # Cancel brake
            print(f"BOOST: Low speed acceleration: Speed={speed:.1f}km/h, Throttle={throttle:.2f}")
        
        # If vehicle is completely stationary, force start
        if speed < 0.5:
            throttle = 0.8
            brake = 0.0
            steering = steering * 0.5  # Reduce steering to avoid spinning
            print(f"FORCE START: Throttle={throttle:.2f}, Steering={steering:.2f}")
        
        # Safety-based brake override (但保持最小速度)
        if self.model_type == 'hydra' and predictions.get('collision_risk', 0) > 0.8:
            if speed > 15:  # 只在高速时刹车
                brake = max(brake, 0.3)
                throttle = max(0.3, throttle - 0.2)  # 保持最小油门
        elif self.model_type == 'omnidrive' and predictions.get('risk_assessment', 0) > 0.9:
            if speed > 15:  # 只在高速时刹车
                brake = max(brake, 0.3)
                throttle = max(0.3, throttle - 0.2)  # 保持最小油门
        
        control = carla.VehicleControl()
        control.throttle = throttle
        control.steer = steering
        control.brake = brake
        control.manual_gear_shift = False  # 确保自动档
        self.vehicle.apply_control(control)
        
        return {
            'steering': steering, 
            'throttle': throttle, 
            'brake': brake, 
            'speed': speed,
            'location': location,
            'velocity': velocity
        }
    
    def visualize_predictions(self, image, predictions, applied_control):
        """Create visualization overlay"""
        display_img = image.copy()
        
        # Title
        title = f"NVIDIA {self.model_type.upper()} - Real-time Test"
        cv2.putText(display_img, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        y_offset = 50
        
        # Vehicle status - color coding for speed
        speed = applied_control['speed']
        if speed < 5:
            speed_color = (0, 0, 255)  # Red - too slow
        elif speed < 15:
            speed_color = (0, 165, 255)  # Orange - slow
        else:
            speed_color = (0, 255, 0)  # Green - normal
        
        cv2.putText(display_img, f"Speed: {speed:.1f} km/h", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, speed_color, 2)
        y_offset += 25
        
        # Show position change
        location = applied_control['location']
        cv2.putText(display_img, f"Pos: ({location.x:.1f}, {location.y:.1f})", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 20
        
        # Show velocity vector
        velocity = applied_control['velocity']
        vel_mag = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        cv2.putText(display_img, f"Velocity: {vel_mag:.2f} m/s", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 25
        
        # Control outputs
        cv2.putText(display_img, "Applied Controls:", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        y_offset += 20
        
        cv2.putText(display_img, f"  Steering: {applied_control['steering']:.3f}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 18
        
        cv2.putText(display_img, f"  Throttle: {applied_control['throttle']:.3f}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        y_offset += 18
        
        cv2.putText(display_img, f"  Brake: {applied_control['brake']:.3f}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        y_offset += 25
        
        # Model-specific predictions
        if self.model_type == 'hydra':
            cv2.putText(display_img, "Hydra-MDP Predictions:", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            y_offset += 20
            
            cv2.putText(display_img, f"  Safety Score: {predictions['safety_score']:.3f}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            y_offset += 18
            
            cv2.putText(display_img, f"  Collision Risk: {predictions['collision_risk']:.3f}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            y_offset += 18
            
            cv2.putText(display_img, f"  Predicted Speed: {predictions['speed']:.1f} km/h", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
        elif self.model_type == 'omnidrive':
            cv2.putText(display_img, "OmniDrive Predictions:", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            y_offset += 20
            
            cv2.putText(display_img, f"  Risk Assessment: {predictions['risk_assessment']:.3f}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)
            y_offset += 18
            
            cv2.putText(display_img, f"  Predicted Speed: {predictions['speed']:.1f} km/h", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 18
            
            cv2.putText(display_img, f"  Scene Reasoning: {predictions['scene_reasoning'][:3]}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 255, 150), 1)
        
        # Steering visualization
        center_x, center_y = 112, 180
        steer_magnitude = abs(applied_control['steering'])
        if steer_magnitude > 0.05:
            arrow_length = int(steer_magnitude * 60)
            if applied_control['steering'] < 0:  # Left
                cv2.arrowedLine(display_img, (center_x, center_y), 
                               (center_x - arrow_length, center_y), (0, 255, 255), 3)
                cv2.putText(display_img, "LEFT", (center_x - 50, center_y + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            else:  # Right
                cv2.arrowedLine(display_img, (center_x, center_y), 
                               (center_x + arrow_length, center_y), (0, 255, 255), 3)
                cv2.putText(display_img, "RIGHT", (center_x + 20, center_y + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        return display_img
    
    def run_test(self, duration=60):
        """Run real-time test"""
        if not self.connect_carla():
            return
        
        if not self.spawn_vehicle_and_camera():
            return
        
        # Setup image callback
        image_queue = []
        def on_image(carla_img):
            image_queue.append(carla_img)
        
        self.camera.listen(on_image)
        
        # Setup collision detection
        blueprint_library = self.world.get_blueprint_library()
        collision_bp = blueprint_library.find('sensor.other.collision')
        collision_sensor = self.world.spawn_actor(collision_bp, carla.Transform(), attach_to=self.vehicle)
        
        collision_count = 0
        def on_collision(event):
            nonlocal collision_count
            collision_count += 1
        
        collision_sensor.listen(on_collision)
        
        print(f"\nSTARTING {duration}s test with {self.model_type.upper()}")
        print("Press 'Q' to quit early, 'R' to respawn")
        
        start_time = time.time()
        frame_count = 0
        
        try:
            while time.time() - start_time < duration:
                self.world.tick()
                
                if image_queue:
                    carla_image = image_queue.pop(0)
                    image_array, image_tensor = self.process_image(carla_image)
                    
                    # Get predictions
                    predictions = self.predict_actions(image_tensor)
                    
                    # Apply control
                    applied_control = self.apply_control(predictions)
                    
                    # Visualize
                    display_img = self.visualize_predictions(image_array, predictions, applied_control)
                    cv2.imshow(f"NVIDIA {self.model_type.upper()} Test", display_img)
                    
                    frame_count += 1
                    
                    # Handle keyboard input
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('r'):
                        # Simple respawn
                        spawn_points = self.world.get_map().get_spawn_points()
                        new_spawn = random.choice(spawn_points)
                        self.vehicle.set_transform(new_spawn)
                        collision_count = 0
                
                # Status update every 5 seconds - 显示更多信息
                elapsed = time.time() - start_time
                if frame_count % 100 == 0 and frame_count > 0:
                    fps = frame_count / elapsed
                    location = self.vehicle.get_location()
                    velocity = self.vehicle.get_velocity()
                    speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2) * 3.6
                    print(f"Time: {elapsed:.1f}s | FPS: {fps:.1f} | Speed: {speed:.1f}km/h | "
                          f"Pos: ({location.x:.1f},{location.y:.1f}) | Collisions: {collision_count}")
        
        except KeyboardInterrupt:
            print("\nTest interrupted by user")
        
        finally:
            # Results
            elapsed = time.time() - start_time
            avg_fps = frame_count / elapsed if elapsed > 0 else 0
            
            print(f"\nTest Results for {self.model_type.upper()}:")
            print(f"Duration: {elapsed:.1f}s")
            print(f"Frames: {frame_count}")
            print(f"Average FPS: {avg_fps:.1f}")
            print(f"Collisions: {collision_count}")
            print(f"Collision rate: {collision_count/elapsed*60:.1f} per minute")
            
            # Cleanup
            cv2.destroyAllWindows()
            try:
                collision_sensor.destroy()
                self.camera.destroy()
                self.vehicle.destroy()
            except:
                pass
            
            # Reset synchronous mode
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)

def main():
    parser = argparse.ArgumentParser(description='Test NVIDIA Advanced Models in CARLA')
    parser.add_argument('--model', choices=['hydra', 'omnidrive'], default='hydra',
                       help='Model to test (default: hydra)')
    parser.add_argument('--duration', type=int, default=60,
                       help='Test duration in seconds (default: 60)')
    parser.add_argument('--device', default='cuda',
                       help='Device to use (default: cuda)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("NVIDIA ADVANCED MODELS CARLA TEST")
    print("=" * 60)
    
    tester = NVIDIAModelTester(args.model, args.device)
    tester.run_test(args.duration)

if __name__ == "__main__":
    main()