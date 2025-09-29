"""Real Data Training Script for Conditional Affordance Learning.

This script loads the collected CARLA data (RGB images + JSON metadata) and
trains the ConditionalAffordanceNet model on real driving scenarios.
"""
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from algorithms.cal_model import ConditionalAffordanceNet


class CarlaAffordanceDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.rgb_dir = os.path.join(data_dir, 'rgb')
        self.metadata_dir = os.path.join(data_dir, 'metadata')
        
        # Get all frame files that have both RGB and metadata
        rgb_files = set(f.replace('.png', '') for f in os.listdir(self.rgb_dir) if f.endswith('.png'))
        metadata_files = set(f.replace('.json', '') for f in os.listdir(self.metadata_dir) if f.endswith('.json'))
        self.frame_ids = sorted(rgb_files.intersection(metadata_files))
        
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"Found {len(self.frame_ids)} valid frame pairs")
    
    def __len__(self):
        return len(self.frame_ids)
    
    def __getitem__(self, idx):
        frame_id = self.frame_ids[idx]
        
        # Load RGB image
        img_path = os.path.join(self.rgb_dir, f'{frame_id}.png')
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        # Load metadata with error handling
        meta_path = os.path.join(self.metadata_dir, f'{frame_id}.json')
        try:
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            # Return default values for corrupted files
            metadata = {
                'front_vehicle_distance_m': 50.0,
                'lane_offset_m': 0.0,
                'risk_score': 0.0,
                'high_level_command': 'straight'
            }
        
        # Extract targets
        front_distance = metadata.get('front_vehicle_distance_m', 50.0)  # default 50m if no vehicle
        if front_distance is None:
            front_distance = 50.0
        
        lane_offset = metadata.get('lane_offset_m', 0.0)  # default center
        if lane_offset is None:
            lane_offset = 0.0
            
        risk_score = metadata.get('risk_score', 0.0)
        if risk_score is None:
            risk_score = 0.0
        
        # High level command (default to 'straight' = 2)
        cmd_map = {'left': 0, 'right': 1, 'straight': 2, 'follow': 3}
        cmd_str = metadata.get('high_level_command', 'straight')
        command = cmd_map.get(cmd_str, 2)
        
        targets = {
            'front_distance': torch.tensor([front_distance], dtype=torch.float32),
            'lane_offset': torch.tensor([lane_offset], dtype=torch.float32),
            'risk_score': torch.tensor([risk_score], dtype=torch.float32)
        }
        
        return image, command, targets


def train_real_data():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load dataset
    dataset = CarlaAffordanceDataset('clean_research_data/scenario_00')
    if len(dataset) == 0:
        print("No data found! Make sure data collection completed successfully.")
        return
    
    # Split train/val (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    print(f'Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}')
    
    # Model
    model = ConditionalAffordanceNet()
    model.to(device)
    
    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    mse_loss = nn.MSELoss()
    
    # Training loop
    epochs = 10
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        for batch_idx, (images, commands, targets) in enumerate(train_loader):
            images = images.to(device)
            commands = torch.tensor(commands, dtype=torch.long).to(device)
            
            optimizer.zero_grad()
            outputs = model(images, commands)
            
            loss = 0.0
            if 'front_distance' in outputs:
                loss += mse_loss(outputs['front_distance'].squeeze(), targets['front_distance'].squeeze().to(device))
            if 'lane_offset' in outputs:
                loss += mse_loss(outputs['lane_offset'].squeeze(), targets['lane_offset'].squeeze().to(device))
            if 'risk_score' in outputs:
                loss += mse_loss(outputs['risk_score'].squeeze(), targets['risk_score'].squeeze().to(device))
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.6f}')
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, commands, targets in val_loader:
                images = images.to(device)
                commands = torch.tensor(commands, dtype=torch.long).to(device)
                
                outputs = model(images, commands)
                
                loss = 0.0
                if 'front_distance' in outputs:
                    loss += mse_loss(outputs['front_distance'].squeeze(), targets['front_distance'].squeeze().to(device))
                if 'lane_offset' in outputs:
                    loss += mse_loss(outputs['lane_offset'].squeeze(), targets['lane_offset'].squeeze().to(device))
                if 'risk_score' in outputs:
                    loss += mse_loss(outputs['risk_score'].squeeze(), targets['risk_score'].squeeze().to(device))
                
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f'Epoch {epoch+1}/{epochs}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}')
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(model.state_dict(), 'checkpoints/cal_real_data.pt')
            print(f'Saved new best model with val loss: {avg_val_loss:.6f}')
    
    print('Training completed!')
    print(f'Best validation loss: {best_val_loss:.6f}')
    print('Model saved to: checkpoints/cal_real_data.pt')


if __name__ == '__main__':
    train_real_data()