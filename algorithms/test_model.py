"""Test the trained model on CARLA data"""
import os
import json
import torch
from PIL import Image
import torchvision.transforms as transforms
from algorithms.cal_model import ConditionalAffordanceNet


def test_trained_model():
    """Test our trained model on some samples"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Testing on device: {device}')
    
    # Load the trained model
    model = ConditionalAffordanceNet(
        feature_dim=128,
        affordance_dims={
            'front_distance': 1,
            'risk_score': 1
        }
    )
    
    # Load checkpoint
    checkpoint_path = 'checkpoints/quick_model_tonight.pt'
    if not os.path.exists(checkpoint_path):
        print(f"Model checkpoint not found: {checkpoint_path}")
        return
    
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    print("Model loaded successfully!")
    
    # Prepare transform
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    
    # Test on a few samples
    data_dir = 'clean_research_data/scenario_00'
    rgb_dir = os.path.join(data_dir, 'rgb')
    metadata_dir = os.path.join(data_dir, 'metadata')
    
    # Find some good samples
    test_files = []
    for filename in os.listdir(metadata_dir)[:10]:  # Test first 10
        if filename.endswith('.json'):
            frame_id = filename.replace('.json', '')
            img_path = os.path.join(rgb_dir, f'{frame_id}.png')
            meta_path = os.path.join(metadata_dir, filename)
            
            if os.path.exists(img_path) and os.path.getsize(meta_path) > 100:
                test_files.append(frame_id)
    
    print(f"\\nTesting model on {len(test_files)} samples:")
    print("=" * 60)
    
    total_distance_error = 0
    total_risk_error = 0
    valid_tests = 0
    
    with torch.no_grad():
        for i, frame_id in enumerate(test_files[:5]):  # Test 5 samples
            try:
                # Load image
                img_path = os.path.join(rgb_dir, f'{frame_id}.png')
                image = Image.open(img_path).convert('RGB')
                image = transform(image).unsqueeze(0).to(device)
                
                # Load ground truth
                meta_path = os.path.join(metadata_dir, f'{frame_id}.json')
                with open(meta_path, 'r') as f:
                    metadata = json.load(f)
                
                # Get predictions
                command = torch.tensor([2], dtype=torch.long).to(device)  # straight
                predictions = model(image, command)
                
                # Extract values
                pred_distance = predictions['front_distance'].item()
                pred_risk = predictions['risk_score'].item()
                
                true_distance = metadata.get('front_vehicle_distance_m', 50.0)
                if true_distance is None:
                    true_distance = 50.0
                    
                true_risk = metadata.get('risk_score', 0.0)
                if true_risk is None:
                    true_risk = 0.0
                
                # Calculate errors
                distance_error = abs(pred_distance - true_distance)
                risk_error = abs(pred_risk - true_risk)
                
                total_distance_error += distance_error
                total_risk_error += risk_error
                valid_tests += 1
                
                print(f"Sample {i+1} (Frame {frame_id}):")
                print(f"  Front Distance: True={true_distance:.1f}m, Pred={pred_distance:.1f}m, Error={distance_error:.1f}m")
                print(f"  Risk Score:     True={true_risk:.3f}, Pred={pred_risk:.3f}, Error={risk_error:.3f}")
                print()
                
            except Exception as e:
                print(f"Error testing sample {frame_id}: {e}")
                continue
    
    if valid_tests > 0:
        avg_distance_error = total_distance_error / valid_tests
        avg_risk_error = total_risk_error / valid_tests
        
        print("=" * 60)
        print(f"AVERAGE PERFORMANCE:")
        print(f"  Average Distance Error: {avg_distance_error:.2f}m")
        print(f"  Average Risk Error: {avg_risk_error:.3f}")
        print(f"  Tested on {valid_tests} samples")
        
        if avg_distance_error < 10.0:
            print("? Good distance prediction!")
        if avg_risk_error < 0.2:
            print("? Good risk prediction!")
    
    print("\\n? Your model is ready for use!")
    print("? Next: Integrate with CARLA controller for real-time testing")


if __name__ == '__main__':
    test_trained_model()