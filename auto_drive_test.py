"""Auto-driving GPU Model Test with Movement and English Comments"""
import os
import sys
import time
import numpy as np
import torch
import cv2
from PIL import Image
import torchvision.transforms as transforms
import threading
import queue
import random

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import carla
except ImportError:
    print("CARLA module not found!")
    sys.exit(1)

from algorithms.cal_model import ConditionalAffordanceNet


class AutoDrivingGPUTest:
    """Auto-driving GPU model test with actual movement"""
    
    def __init__(self):
        # Setup device and model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {self.device}")
        
        # Load model
        self.model = ConditionalAffordanceNet(
            feature_dim=256,
            affordance_dims={'front_distance': 1, 'lane_offset': 1, 'risk_score': 1}
        )
        
        checkpoint = torch.load('checkpoints/gpu_trained_model.pt', 
                               map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        print(f"GPU model loaded (loss: {checkpoint['val_loss']:.4f})")
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # CARLA connection
        self.client = None
        self.world = None
        self.vehicle = None
        self.camera = None
        self.collision_sensor = None
        
        # Display and control
        self.current_image = None
        self.latest_prediction = None
        self.image_queue = queue.Queue(maxsize=3)
        self.display_running = False
        
        # Stats and control
        self.frame_count = 0
        self.collision_count = 0
        self.start_time = time.time()
        self.current_command = 'straight'
        
        # Control smoothing
        self.throttle_history = []
        self.steer_history = []
        
    def connect_and_spawn(self):
        """Connect to CARLA and spawn with proper setup"""
        try:
            # Connect
            self.client = carla.Client('localhost', 2000)
            self.client.set_timeout(15.0)
            self.world = self.client.get_world()
            
            # Enable synchronous mode for better control
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05
            self.world.apply_settings(settings)
            
            print("Connected to CARLA (sync mode)")
            
            # Spawn vehicle
            blueprint_library = self.world.get_blueprint_library()
            vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
            
            # Find a good spawn point
            spawn_points = self.world.get_map().get_spawn_points()
            spawn_point = random.choice(spawn_points)
            
            self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
            print("Vehicle spawned")
            
            # Add some AI traffic for interaction
            self.spawn_traffic()
            
            # Setup camera
            camera_bp = blueprint_library.find('sensor.camera.rgb')
            camera_bp.set_attribute('image_size_x', '800')
            camera_bp.set_attribute('image_size_y', '600')
            camera_bp.set_attribute('fov', '90')
            
            camera_transform = carla.Transform(
                carla.Location(x=2.0, z=1.4),
                carla.Rotation(pitch=-15)
            )
            self.camera = self.world.spawn_actor(camera_bp, camera_transform, 
                                               attach_to=self.vehicle)
            self.camera.listen(self.process_image)
            print("Camera ready")
            
            # Setup collision sensor
            collision_bp = blueprint_library.find('sensor.other.collision')
            self.collision_sensor = self.world.spawn_actor(
                collision_bp, carla.Transform(), attach_to=self.vehicle
            )
            self.collision_sensor.listen(self.on_collision)
            print("Collision sensor ready")
            
            return True
            
        except Exception as e:
            print(f"Setup failed: {e}")
            return False
    
    def spawn_traffic(self):
        """Spawn some AI traffic for interaction"""
        try:
            # Get traffic manager
            traffic_manager = self.client.get_trafficmanager(8000)
            traffic_manager.set_global_distance_to_leading_vehicle(2.0)
            
            # Spawn some vehicles
            blueprint_library = self.world.get_blueprint_library()
            vehicle_bps = blueprint_library.filter('vehicle.*')
            vehicle_bps = [bp for bp in vehicle_bps if bp.id.endswith('.*')][:10]
            
            spawn_points = self.world.get_map().get_spawn_points()
            random.shuffle(spawn_points)
            
            spawned = 0
            for i, spawn_point in enumerate(spawn_points[:20]):
                if spawned >= 5:  # Limit to 5 traffic vehicles
                    break
                    
                try:
                    vehicle_bp = random.choice(vehicle_bps)
                    vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
                    vehicle.set_autopilot(True, 8000)
                    spawned += 1
                except:
                    continue
            
            print(f"Spawned {spawned} traffic vehicles")
            
        except Exception as e:
            print(f"Traffic spawn failed: {e}")
    
    def process_image(self, image):
        """Process camera image"""
        try:
            # Convert to numpy
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = array.reshape((image.height, image.width, 4))
            rgb_array = array[:, :, :3]
            
            # For display
            display_img = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
            
            # Add AI prediction overlay
            if self.latest_prediction:
                self.add_overlay(display_img)
            
            # Queue for display
            if not self.image_queue.full():
                self.image_queue.put(display_img.copy())
            
            # For AI model
            self.current_image = Image.fromarray(rgb_array)
            
        except Exception as e:
            print(f"Image processing error: {e}")
    
    def add_overlay(self, image):
        """Add comprehensive prediction overlay"""
        try:
            pred = self.latest_prediction
            h, w = image.shape[:2]
            
            # Semi-transparent background
            overlay = image.copy()
            cv2.rectangle(overlay, (10, 10), (450, 200), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.8, image, 0.2, 0, image)
            
            # Text info
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            # Title
            cv2.putText(image, "GPU AI Driver - Live Predictions", (20, 35), 
                       font, 0.7, (0, 255, 0), 2)
            
            # Command
            cv2.putText(image, f"Command: {self.current_command}", (20, 65), 
                       font, 0.6, (255, 255, 255), 2)
            
            # Distance with color coding
            dist = pred['front_distance']
            if dist < 5:
                dist_color = (0, 0, 255)  # Red - danger
                dist_status = "DANGER"
            elif dist < 15:
                dist_color = (0, 255, 255)  # Yellow - caution
                dist_status = "CAUTION"
            else:
                dist_color = (0, 255, 0)  # Green - safe
                dist_status = "SAFE"
            
            cv2.putText(image, f"Front Distance: {dist:.1f}m ({dist_status})", 
                       (20, 95), font, 0.6, dist_color, 2)
            
            # Lane offset
            offset = pred['lane_offset']
            offset_color = (0, 255, 0) if abs(offset) < 0.5 else (0, 255, 255)
            cv2.putText(image, f"Lane Offset: {offset:.2f}m", 
                       (20, 125), font, 0.6, offset_color, 2)
            
            # Risk assessment
            risk = pred['risk_score']
            if risk > 0.7:
                risk_color = (0, 0, 255)
                risk_text = "HIGH RISK"
            elif risk > 0.3:
                risk_color = (0, 255, 255)
                risk_text = "MEDIUM RISK"
            else:
                risk_color = (0, 255, 0)
                risk_text = "LOW RISK"
            
            cv2.putText(image, f"Risk: {risk:.3f} ({risk_text})", 
                       (20, 155), font, 0.6, risk_color, 2)
            
            # Stats
            fps = self.frame_count / max(time.time() - self.start_time, 1)
            cv2.putText(image, f"FPS: {fps:.1f} | Collisions: {self.collision_count}", 
                       (20, 185), font, 0.5, (255, 255, 255), 1)
            
            # Risk meter on the right
            meter_x, meter_y = w - 80, 50
            meter_height = int(120 * min(risk, 1.0))
            
            # Risk bar
            cv2.rectangle(image, (meter_x, meter_y + 120 - meter_height), 
                         (meter_x + 40, meter_y + 120), risk_color, -1)
            cv2.rectangle(image, (meter_x, meter_y), (meter_x + 40, meter_y + 120), 
                         (255, 255, 255), 2)
            cv2.putText(image, "RISK", (meter_x - 5, meter_y + 140), 
                       font, 0.4, (255, 255, 255), 1)
            
        except Exception as e:
            print(f"Overlay error: {e}")
    
    def on_collision(self, event):
        """Handle collisions"""
        self.collision_count += 1
        other_actor = event.other_actor
        print(f"Collision #{self.collision_count} with {other_actor.type_id}")
    
    def predict_and_control(self):
        """Run AI prediction and compute control"""
        if self.current_image is None:
            return carla.VehicleControl()
        
        try:
            # AI Prediction
            image_tensor = self.transform(self.current_image).unsqueeze(0).to(self.device)
            
            # Rotate commands for variety
            commands = ['straight', 'left', 'right', 'follow']
            if self.frame_count % 200 == 0:  # Change every 10 seconds
                self.current_command = random.choice(commands)
            
            cmd_map = {'left': 0, 'right': 1, 'straight': 2, 'follow': 3}
            command_idx = cmd_map.get(self.current_command, 2)
            command_tensor = torch.tensor([command_idx], dtype=torch.long).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(image_tensor, command_tensor)
            
            front_distance = outputs['front_distance'].item()
            lane_offset = outputs['lane_offset'].item()
            risk_score = outputs['risk_score'].item()
            
            # Store for display
            self.latest_prediction = {
                'front_distance': front_distance,
                'lane_offset': lane_offset,
                'risk_score': risk_score
            }
            
            # Compute control based on AI predictions
            control = carla.VehicleControl()
            
            # Throttle based on distance and risk
            if front_distance < 3.0 or risk_score > 0.8:
                control.throttle = 0.0
                control.brake = 1.0
            elif front_distance < 10.0 or risk_score > 0.4:
                control.throttle = 0.3
                control.brake = 0.2
            elif front_distance > 25.0 and risk_score < 0.2:
                control.throttle = 0.7
                control.brake = 0.0
            else:
                control.throttle = 0.5
                control.brake = 0.0
            
            # Steering based on lane offset and command
            base_steer = -lane_offset * 1.0  # Lane keeping
            
            # Command influence
            if self.current_command == 'left':
                base_steer -= 0.3
            elif self.current_command == 'right':
                base_steer += 0.3
            
            control.steer = max(-1.0, min(1.0, base_steer))
            
            # Smooth controls
            self.throttle_history.append(control.throttle)
            self.steer_history.append(control.steer)
            
            if len(self.throttle_history) > 5:
                self.throttle_history.pop(0)
                self.steer_history.pop(0)
            
            control.throttle = sum(self.throttle_history) / len(self.throttle_history)
            control.steer = sum(self.steer_history) / len(self.steer_history)
            
            return control
            
        except Exception as e:
            print(f"Control error: {e}")
            return carla.VehicleControl()
    
    def display_loop(self):
        """Display loop"""
        cv2.namedWindow('GPU Auto-Driver Live View', cv2.WINDOW_AUTOSIZE)
        
        while self.display_running:
            try:
                if not self.image_queue.empty():
                    image = self.image_queue.get()
                    cv2.imshow('GPU Auto-Driver Live View', image)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Stopping via display window")
                    self.display_running = False
                    break
                elif key == ord('w'):
                    self.current_command = 'straight'
                    print("Manual command: straight")
                elif key == ord('a'):
                    self.current_command = 'left'
                    print("Manual command: left")
                elif key == ord('d'):
                    self.current_command = 'right'
                    print("Manual command: right")
                elif key == ord('s'):
                    self.current_command = 'follow'
                    print("Manual command: follow")
                    
            except:
                pass
        
        cv2.destroyAllWindows()
    
    def run_test(self, duration=90):
        """Run the auto-driving test"""
        print(f"Starting {duration}s GPU auto-driving test")
        print("Camera controls: Q=quit, WASD=manual commands")
        print("AI will automatically drive and avoid obstacles!")
        
        # Start display
        self.display_running = True
        display_thread = threading.Thread(target=self.display_loop)
        display_thread.daemon = True
        display_thread.start()
        
        start_time = time.time()
        last_print = start_time
        
        try:
            while time.time() - start_time < duration and self.display_running:
                # Tick world
                self.world.tick()
                self.frame_count += 1
                
                # Get AI control and apply
                control = self.predict_and_control()
                self.vehicle.apply_control(control)
                
                # Print status every 5 seconds
                now = time.time()
                if now - last_print > 5.0:
                    elapsed = now - start_time
                    fps = self.frame_count / (now - self.start_time)
                    
                    if self.latest_prediction:
                        pred = self.latest_prediction
                        print(f"\nTime: {elapsed:.1f}s | FPS: {fps:.1f} | Command: {self.current_command}")
                        print(f"AI: Dist={pred['front_distance']:.1f}m, "
                              f"Lane={pred['lane_offset']:.2f}m, Risk={pred['risk_score']:.3f}")
                        print(f"Control: Throttle={control.throttle:.2f}, Steer={control.steer:.2f}")
                        print(f"Collisions: {self.collision_count}")
                    
                    last_print = now
                
                time.sleep(0.01)  # Small delay
                
        except KeyboardInterrupt:
            print("\nTest stopped by user")
        
        self.display_running = False
        display_thread.join(timeout=2)
        
        # Final summary
        total_time = time.time() - start_time
        success_rate = max(0, 100 - (self.collision_count * 20))
        
        print(f"\nAuto-driving test completed!")
        print(f"Duration: {total_time:.1f}s")
        print(f"Frames processed: {self.frame_count}")
        print(f"Total collisions: {self.collision_count}")
        print(f"Success rate: {success_rate:.1f}%")
        
        if self.collision_count == 0:
            print("Perfect driving! No collisions!")
        elif self.collision_count < 3:
            print("Good performance with minimal collisions")
        else:
            print("Multiple collisions - room for improvement")
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            self.display_running = False
            
            if self.camera:
                self.camera.destroy()
            if self.collision_sensor:
                self.collision_sensor.destroy()
            if self.vehicle:
                self.vehicle.destroy()
            
            # Clean up traffic
            for actor in self.world.get_actors().filter('vehicle.*'):
                if actor.id != self.vehicle.id if self.vehicle else -1:
                    actor.destroy()
            
            # Restore async mode
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)
            
            print("Cleanup complete")
        except Exception as e:
            print(f"Cleanup error: {e}")


def main():
    """Main test function"""
    print("GPU Auto-Driving Test with Live View")
    print("=" * 50)
    
    tester = AutoDrivingGPUTest()
    
    try:
        if tester.connect_and_spawn():
            time.sleep(3)  # Let everything initialize
            tester.run_test(duration=90)  # 90 second test
    except Exception as e:
        print(f"Error: {e}")
    finally:
        tester.cleanup()


if __name__ == '__main__':
    main()