"""Simple GPU Model Test with Camera View"""
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

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import carla
except ImportError:
    print("? CARLA module not found!")
    sys.exit(1)

from algorithms.cal_model import ConditionalAffordanceNet


class SimpleGPUTester:
    """Simple GPU model tester with live camera view"""
    
    def __init__(self):
        # Setup device and model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"? Device: {self.device}")
        
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
        print(f"? GPU model loaded (loss: {checkpoint['val_loss']:.4f})")
        
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
        
        # Display
        self.current_image = None
        self.latest_prediction = None
        self.image_queue = queue.Queue(maxsize=3)
        self.display_running = False
        
        # Stats
        self.frame_count = 0
        self.start_time = time.time()
        
    def connect_and_spawn(self):
        """Connect to CARLA and spawn simple setup"""
        try:
            # Connect
            self.client = carla.Client('localhost', 2000)
            self.client.set_timeout(10.0)
            self.world = self.client.get_world()
            print("? Connected to CARLA")
            
            # Simple setup - just spawn anywhere
            blueprint_library = self.world.get_blueprint_library()
            vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
            
            # Try multiple spawn points until one works
            spawn_points = self.world.get_map().get_spawn_points()
            
            for i, spawn_point in enumerate(spawn_points[:10]):  # Try first 10
                try:
                    self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
                    print(f"? Vehicle spawned at point {i}")
                    break
                except:
                    continue
            
            if not self.vehicle:
                print("? Could not spawn vehicle")
                return False
            
            # Simple camera
            camera_bp = blueprint_library.find('sensor.camera.rgb')
            camera_bp.set_attribute('image_size_x', '640')
            camera_bp.set_attribute('image_size_y', '480')
            
            camera_transform = carla.Transform(carla.Location(x=2.0, z=1.4))
            self.camera = self.world.spawn_actor(camera_bp, camera_transform, 
                                               attach_to=self.vehicle)
            self.camera.listen(self.process_image)
            print("? Camera ready")
            
            return True
            
        except Exception as e:
            print(f"? Setup failed: {e}")
            return False
    
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
            print(f"?? Image error: {e}")
    
    def add_overlay(self, image):
        """Add prediction overlay"""
        try:
            pred = self.latest_prediction
            h, w = image.shape[:2]
            
            # Background
            cv2.rectangle(image, (10, 10), (350, 140), (0, 0, 0), -1)
            
            # Text
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image, f"GPU Model Predictions:", (20, 35), font, 0.6, (0, 255, 0), 2)
            cv2.putText(image, f"Front Distance: {pred['front_distance']:.1f}m", 
                       (20, 65), font, 0.5, (255, 255, 255), 1)
            cv2.putText(image, f"Lane Offset: {pred['lane_offset']:.2f}m", 
                       (20, 85), font, 0.5, (255, 255, 255), 1)
            
            # Risk with color
            risk = pred['risk_score']
            risk_color = (0, 255, 0) if risk < 0.3 else (0, 255, 255) if risk < 0.7 else (0, 0, 255)
            cv2.putText(image, f"Risk Score: {risk:.3f}", 
                       (20, 105), font, 0.5, risk_color, 1)
            
            # FPS
            fps = self.frame_count / max(time.time() - self.start_time, 1)
            cv2.putText(image, f"FPS: {fps:.1f}", (20, 125), font, 0.4, (255, 255, 255), 1)
            
        except Exception as e:
            print(f"?? Overlay error: {e}")
    
    def predict(self):
        """Run AI prediction"""
        if self.current_image is None:
            return
        
        try:
            # Preprocess
            image_tensor = self.transform(self.current_image).unsqueeze(0).to(self.device)
            command_tensor = torch.tensor([2], dtype=torch.long).to(self.device)  # straight
            
            # Predict
            with torch.no_grad():
                outputs = self.model(image_tensor, command_tensor)
            
            # Store results
            self.latest_prediction = {
                'front_distance': outputs['front_distance'].item(),
                'lane_offset': outputs['lane_offset'].item(),
                'risk_score': outputs['risk_score'].item()
            }
            
        except Exception as e:
            print(f"?? Prediction error: {e}")
    
    def display_loop(self):
        """Display loop"""
        cv2.namedWindow('GPU Model Live Test', cv2.WINDOW_AUTOSIZE)
        
        while self.display_running:
            try:
                if not self.image_queue.empty():
                    image = self.image_queue.get()
                    cv2.imshow('GPU Model Live Test', image)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.display_running = False
                    break
                    
            except:
                pass
        
        cv2.destroyAllWindows()
    
    def run_test(self, duration=30):
        """Run the test"""
        print(f"? Starting {duration}s GPU model test")
        print("? Press Q to quit the camera window")
        
        # Start display
        self.display_running = True
        display_thread = threading.Thread(target=self.display_loop)
        display_thread.daemon = True
        display_thread.start()
        
        start_time = time.time()
        last_print = start_time
        
        try:
            while time.time() - start_time < duration and self.display_running:
                # Tick world and run prediction
                self.world.tick()
                self.frame_count += 1
                self.predict()
                
                # Print status every 5 seconds
                now = time.time()
                if now - last_print > 5.0:
                    elapsed = now - start_time
                    fps = self.frame_count / (now - self.start_time)
                    
                    if self.latest_prediction:
                        pred = self.latest_prediction
                        print(f"\\n??  {elapsed:.1f}s | FPS: {fps:.1f}")
                        print(f"? AI: Dist={pred['front_distance']:.1f}m, "
                              f"Lane={pred['lane_offset']:.2f}m, Risk={pred['risk_score']:.3f}")
                    
                    last_print = now
                
                time.sleep(0.02)  # ~50 FPS
                
        except KeyboardInterrupt:
            print("\\n?? Test stopped")
        
        self.display_running = False
        display_thread.join(timeout=2)
        
        print(f"\\n? Test completed! Processed {self.frame_count} frames")
    
    def cleanup(self):
        """Cleanup"""
        try:
            if self.camera:
                self.camera.destroy()
            if self.vehicle:
                self.vehicle.destroy()
            print("? Cleanup done")
        except:
            pass


def main():
    """Main test function"""
    print("? Simple GPU Model Live Test")
    print("=" * 40)
    
    tester = SimpleGPUTester()
    
    try:
        if tester.connect_and_spawn():
            time.sleep(2)  # Let camera initialize
            tester.run_test(duration=60)  # 1 minute test
    except Exception as e:
        print(f"? Error: {e}")
    finally:
        tester.cleanup()


if __name__ == '__main__':
    main()