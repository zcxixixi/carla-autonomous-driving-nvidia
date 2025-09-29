"""
NVIDIA PilotNet Training Script for CARLA
Train the classic end-to-end driving model
"""
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys

# Add models directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))
from nvidia_pilotnet import NVIDIAPilotNet, ImprovedPilotNet


class PilotNetDataset(Dataset):
    """Dataset for PilotNet training"""
    
    def __init__(self, data_dir, split='train', train_ratio=0.8):
        self.data_dir = data_dir
        self.images_dir = os.path.join(data_dir, 'images')
        self.labels_dir = os.path.join(data_dir, 'labels')
        
        # Find all samples
        self.samples = []
        for filename in os.listdir(self.labels_dir):
            if filename.endswith('.json'):
                sample_id = filename.replace('label_', '').replace('.json', '')
                img_path = os.path.join(self.images_dir, f'image_{sample_id}.jpg')
                label_path = os.path.join(self.labels_dir, filename)
                
                if os.path.exists(img_path) and os.path.exists(label_path):
                    self.samples.append((img_path, label_path))
        
        # Split data
        total_samples = len(self.samples)
        train_count = int(total_samples * train_ratio)
        
        if split == 'train':
            self.samples = self.samples[:train_count]
        else:  # validation
            self.samples = self.samples[train_count:]
        
        print(f"{split.capitalize()} dataset: {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label_path = self.samples[idx]
        
        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to 224x224 for our model (instead of original 200x66)
        image = cv2.resize(image, (224, 224))
        
        # Convert to tensor and normalize to [0, 255] for the model to handle
        image = torch.from_numpy(image).float().permute(2, 0, 1)  # CHW format
        
        # Load label
        with open(label_path, 'r') as f:
            label_data = json.load(f)
        
        steering = torch.tensor(label_data['steering_angle'], dtype=torch.float32)
        throttle = torch.tensor(label_data['throttle'], dtype=torch.float32)
        brake = torch.tensor(label_data['brake'], dtype=torch.float32)
        
        return image, {
            'steering': steering.unsqueeze(0),  # Add batch dimension
            'throttle': throttle.unsqueeze(0),
            'brake': brake.unsqueeze(0)
        }


def train_pilotnet():
    """Train NVIDIA PilotNet on CARLA data"""
    
    print("=" * 60)
    print("TRAINING NVIDIA PILOTNET FOR CARLA")
    print("=" * 60)
    
    # Check if data exists
    data_dir = 'pilotnet_data'
    if not os.path.exists(data_dir):
        print(f"? Data directory {data_dir} not found!")
        print("Please run collect_pilotnet_data.py first to collect training data.")
        return
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
    
    # Create datasets
    train_dataset = PilotNetDataset(data_dir, split='train')
    val_dataset = PilotNetDataset(data_dir, split='val')
    
    if len(train_dataset) == 0:
        print("? No training data found!")
        return
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    # Create model (use improved version for better control)
    model = ImprovedPilotNet().to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    # Training parameters
    num_epochs = 50
    best_val_loss = float('inf')
    
    # Training history
    train_losses = []
    val_losses = []
    
    print(f"\nTraining for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for images, labels in train_loader:
            images = images.to(device)
            steering_true = labels['steering'].to(device)
            throttle_true = labels['throttle'].to(device)
            brake_true = labels['brake'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            
            # Calculate individual losses
            steering_loss = criterion(outputs['steering'], steering_true)
            throttle_loss = criterion(outputs['throttle'], throttle_true)
            brake_loss = criterion(outputs['brake'], brake_true)
            
            # Combined loss (weighted)
            total_loss = steering_loss * 2.0 + throttle_loss * 1.0 + brake_loss * 1.0
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += total_loss.item()
            train_batches += 1
        
        avg_train_loss = train_loss / train_batches
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                steering_true = labels['steering'].to(device)
                throttle_true = labels['throttle'].to(device)
                brake_true = labels['brake'].to(device)
                
                outputs = model(images)
                
                steering_loss = criterion(outputs['steering'], steering_true)
                throttle_loss = criterion(outputs['throttle'], throttle_true)
                brake_loss = criterion(outputs['brake'], brake_true)
                
                total_loss = steering_loss * 2.0 + throttle_loss * 1.0 + brake_loss * 1.0
                
                val_loss += total_loss.item()
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches
        
        # Update learning rate
        scheduler.step()
        
        # Save history
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # Print progress
        print(f"Epoch [{epoch+1:2d}/{num_epochs}] | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'best_val_loss': best_val_loss
            }
            
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(checkpoint, 'checkpoints/pilotnet_best.pt')
            
            if (epoch + 1) % 10 == 0:
                print(f"? New best model saved! Val Loss: {best_val_loss:.4f}")
    
    print(f"\n? Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved as: checkpoints/pilotnet_best.pt")
    
    # Plot training curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('PilotNet Training Progress')
    plt.legend()
    plt.grid(True)
    plt.savefig('checkpoints/pilotnet_training_curves.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return model, best_val_loss


if __name__ == "__main__":
    # Create checkpoints directory
    os.makedirs('checkpoints', exist_ok=True)
    
    # Train the model
    model, best_loss = train_pilotnet()
    
    if model:
        print(f"\n? PilotNet training complete! Best loss: {best_loss:.4f}")
        print("Ready for real-time testing!")