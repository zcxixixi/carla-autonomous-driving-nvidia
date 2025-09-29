import torch
import numpy as np
from simple_working_model import ConditionalAffordanceNet
import cv2
import carla
import random
import time

# Initialize CARLA and model
def debug_model_predictions():
    """Debug what the model is actually predicting vs what it should predict"""
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    model = ConditionalAffordanceNet().to(device)
    checkpoint = torch.load('checkpoints/gpu_trained_model.pt')
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
        # Spawn vehicle
        vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
        spawn_points = world.get_map().get_spawn_points()
        spawn_point = random.choice(spawn_points)
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        
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
            array = array[:, :, :3]  # Remove alpha channel
            image_data["frame"] = array.copy()
        
        camera.listen(process_image)
        
        # Wait for first frame
        world.tick()
        while image_data["frame"] is None:
            time.sleep(0.01)
            world.tick()
        
        print("Starting debug analysis...")
        
        # Test different commands and analyze predictions
        commands = ['straight', 'follow', 'left', 'right']
        command_mapping = {'straight': 0, 'follow': 1, 'left': 2, 'right': 3}
        
        for i in range(20):  # Test 20 frames
            world.tick()
            time.sleep(0.1)
            
            if image_data["frame"] is not None:
                frame = image_data["frame"]
                
                # Test all commands for this frame
                print(f"\n--- Frame {i+1} ---")
                
                for cmd_name in commands:
                    # Preprocess
                    rgb_resized = cv2.resize(frame, (224, 224))
                    rgb_normalized = rgb_resized.astype(np.float32) / 255.0
                    rgb_tensor = torch.from_numpy(rgb_normalized).permute(2, 0, 1).unsqueeze(0).to(device)
                    
                    # Command
                    cmd_tensor = torch.tensor([command_mapping[cmd_name]], dtype=torch.long).to(device)
                    
                    # Predict
                    with torch.no_grad():
                        distance, lane_offset, risk = model(rgb_tensor, cmd_tensor)
                        
                    dist_val = distance.cpu().numpy()[0, 0]
                    lane_val = lane_offset.cpu().numpy()[0, 0]
                    risk_val = risk.cpu().numpy()[0, 0]
                    
                    print(f"  {cmd_name:8}: Dist={dist_val:6.1f}m, Lane={lane_val:6.2f}m, Risk={risk_val:6.3f}")
                
                # Show the actual frame
                display_frame = frame.copy()
                cv2.putText(display_frame, f"Frame {i+1}/20", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Debug Analysis", display_frame)
                
                key = cv2.waitKey(100) & 0xFF
                if key == ord('q'):
                    break
        
        print("\nDebug analysis complete!")
        print("Check if:")
        print("1. Distance predictions make sense (should be 5-50m typically)")
        print("2. Lane offset is reasonable (-2 to +2m typically)")
        print("3. Risk values are in expected range (0-1 typically)")
        print("4. Different commands give different predictions")
        
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
    debug_model_predictions()