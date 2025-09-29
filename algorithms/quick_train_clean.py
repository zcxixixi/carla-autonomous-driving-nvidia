"""Quick Training Script - Simple version that will finish tonight"""
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from algorithms.cal_model import ConditionalAffordanceNet
import random


class SimpleCarlaDataset(Dataset):
    def __init__(self, data_dir, max_samples=200):
        self.data_dir = data_dir
        self.rgb_dir = os.path.join(data_dir, 'rgb')
        self.metadata_dir = os.path.join(data_dir, 'metadata')
        
        # Only take good files, limit quantity to speed up training
        self.valid_frames = []
        
        print("Scanning valid data files...")
        for filename in os.listdir(self.metadata_dir):
            if not filename.endswith('.json'):
                continue
                
            frame_id = filename.replace('.json', '')
            img_path = os.path.join(self.rgb_dir, f'{frame_id}.png')
            meta_path = os.path.join(self.metadata_dir, filename)
            
            # Check file size and integrity
            try:
                if os.path.getsize(meta_path) > 100 and os.path.exists(img_path) and os.path.getsize(img_path) > 1000:
                    with open(meta_path, 'r') as f:
                        metadata = json.load(f)  # Test if JSON is complete
                    
                    # Quick verify image can be opened
                    with Image.open(img_path) as img:
                        img.verify()
                    
                    self.valid_frames.append(frame_id)
                    
                    if len(self.valid_frames) >= max_samples:
                        break
            except:
                continue
        
        print(f"Found {len(self.valid_frames)} valid data samples")
        
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),  # Smaller input for faster training
            transforms.ToTensor(),
        ])
    
    def __len__(self):
        return len(self.valid_frames)
    
    def __getitem__(self, idx):
        frame_id = self.valid_frames[idx]
        
        # Load image
        img_path = os.path.join(self.rgb_dir, f'{frame_id}.png')
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        # Load labels
        meta_path = os.path.join(self.metadata_dir, f'{frame_id}.json')
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
        
        # Extract key labels
        front_distance = metadata.get('front_vehicle_distance_m', 50.0)
        if front_distance is None:
            front_distance = 50.0
        
        risk_score = metadata.get('risk_score', 0.0)
        if risk_score is None:
            risk_score = 0.0
        
        command = 2  # Simplified: all set to straight
        
        targets = {
            'front_distance': torch.tensor([front_distance], dtype=torch.float32),
            'risk_score': torch.tensor([risk_score], dtype=torch.float32)
        }
        
        return image, command, targets


def quick_train():
    """Quick training function - suitable for tonight's results"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load dataset (limit to 200 samples)
    dataset = SimpleCarlaDataset('clean_research_data/scenario_00', max_samples=200)
    
    if len(dataset) < 10:
        print("Too few data samples, cannot train")
        return None
    
    # Train and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)  # Small batch
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    print(f'Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}')
    
    # Simplified model - only predict two key affordances
    model = ConditionalAffordanceNet(
        feature_dim=128,
        affordance_dims={
            'front_distance': 1,
            'risk_score': 1
        }
    )
    model.to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # Higher learning rate
    mse_loss = nn.MSELoss()
    
    print("Starting quick training (5 epochs)...")
    
    # Only train 5 epochs for quick results
    epochs = 5
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        batch_count = 0
        
        for images, commands, targets in train_loader:
            images = images.to(device)
            commands = torch.tensor([commands] if isinstance(commands, int) else commands, dtype=torch.long).to(device)
            
            optimizer.zero_grad()
            outputs = model(images, commands)
            
            # Only calculate two losses
            loss = mse_loss(outputs['front_distance'].squeeze(), targets['front_distance'].squeeze().to(device))
            loss += mse_loss(outputs['risk_score'].squeeze(), targets['risk_score'].squeeze().to(device))
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            batch_count += 1
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for images, commands, targets in val_loader:
                images = images.to(device)
                commands = torch.tensor([commands] if isinstance(commands, int) else commands, dtype=torch.long).to(device)
                
                outputs = model(images, commands)
                
                loss = mse_loss(outputs['front_distance'].squeeze(), targets['front_distance'].squeeze().to(device))
                loss += mse_loss(outputs['risk_score'].squeeze(), targets['risk_score'].squeeze().to(device))
                
                val_loss += loss.item()
                val_batches += 1
        
        avg_train_loss = train_loss / max(batch_count, 1)
        avg_val_loss = val_loss / max(val_batches, 1)
        
        print(f'Epoch {epoch+1}/{epochs}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}')
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(model.state_dict(), 'checkpoints/quick_model_tonight.pt')
            print(f'Saved new best model, val loss: {avg_val_loss:.4f}')
    
    print(f'\nQuick training completed!')
    print(f'Best validation loss: {best_val_loss:.4f}')
    print(f'Model saved to: checkpoints/quick_model_tonight.pt')
    
    # Test one sample
    model.eval()
    test_sample = dataset[0]
    with torch.no_grad():
        img, cmd, targets = test_sample
        img = img.unsqueeze(0).to(device)
        cmd = torch.tensor([cmd], dtype=torch.long).to(device)
        pred = model(img, cmd)
        
        print(f'\nModel test:')
        print(f'Actual front distance: {targets["front_distance"].item():.2f}m')
        print(f'Predicted front distance: {pred["front_distance"].item():.2f}m')
        print(f'Actual risk score: {targets["risk_score"].item():.3f}')
        print(f'Predicted risk score: {pred["risk_score"].item():.3f}')
    
    return model


if __name__ == '__main__':
    quick_train()