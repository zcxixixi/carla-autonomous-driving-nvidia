"""
PilotNet Training Script for CARLA
Train NVIDIA PilotNet on collected manual driving data
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import json
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time

# Add models to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))
from nvidia_pilotnet_fixed import ImprovedPilotNet

class PilotNetDataset(Dataset):
    """Dataset for PilotNet training"""
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        
        # Load labels
        labels_file = os.path.join(data_dir, 'labels.json')
        if not os.path.exists(labels_file):
            raise FileNotFoundError(f"Labels file not found: {labels_file}")
        
        with open(labels_file, 'r') as f:
            self.labels = json.load(f)
        
        # Images directory
        self.images_dir = os.path.join(data_dir, 'images')
        
        # Filter valid samples
        self.valid_samples = []
        for label in self.labels:
            image_path = os.path.join(self.images_dir, label['image'])
            if os.path.exists(image_path):
                self.valid_samples.append(label)
        
        print(f"Loaded {len(self.valid_samples)} valid samples from {data_dir}")
    
    def __len__(self):
        return len(self.valid_samples)
    
    def __getitem__(self, idx):
        sample = self.valid_samples[idx]
        
        # Load image
        image_path = os.path.join(self.images_dir, sample['image'])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to tensor and normalize
        image = torch.from_numpy(image).float().permute(2, 0, 1)  # HWC -> CHW
        
        # Labels
        steering = torch.tensor([sample['steering']], dtype=torch.float32)
        throttle = torch.tensor([sample['throttle']], dtype=torch.float32)
        brake = torch.tensor([sample['brake']], dtype=torch.float32)
        
        return {
            'image': image,
            'steering': steering,
            'throttle': throttle,
            'brake': brake
        }

def find_data_directories(base_dir="pilotnet_data"):
    """Find all data collection sessions"""
    if not os.path.exists(base_dir):
        return []
    
    data_dirs = []
    for item in os.listdir(base_dir):
        session_dir = os.path.join(base_dir, item)
        if os.path.isdir(session_dir) and os.path.exists(os.path.join(session_dir, 'labels.json')):
            data_dirs.append(session_dir)
    
    return sorted(data_dirs)

def create_combined_dataset(data_dirs):
    """Combine multiple data collection sessions"""
    all_samples = []
    
    for data_dir in data_dirs:
        labels_file = os.path.join(data_dir, 'labels.json')
        images_dir = os.path.join(data_dir, 'images')
        
        with open(labels_file, 'r') as f:
            labels = json.load(f)
        
        # Add full paths and filter valid samples
        for label in labels:
            image_path = os.path.join(images_dir, label['image'])
            if os.path.exists(image_path):
                label['full_image_path'] = image_path
                all_samples.append(label)
    
    return all_samples

class CombinedPilotNetDataset(Dataset):
    """Dataset combining multiple sessions"""
    def __init__(self, samples):
        self.samples = samples
        print(f"Combined dataset: {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image = cv2.imread(sample['full_image_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to tensor
        image = torch.from_numpy(image).float().permute(2, 0, 1)
        
        # Labels
        steering = torch.tensor([sample['steering']], dtype=torch.float32)
        throttle = torch.tensor([sample['throttle']], dtype=torch.float32)
        brake = torch.tensor([sample['brake']], dtype=torch.float32)
        
        return {
            'image': image,
            'steering': steering,
            'throttle': throttle,
            'brake': brake
        }

def train_pilotnet():
    """Train PilotNet model"""
    print("=" * 60)
    print("PILOTNET TRAINING")
    print("=" * 60)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Find data directories
    data_dirs = find_data_directories()
    if not data_dirs:
        print("? No data directories found!")
        print("Please run collect_pilotnet_data.py first to collect training data")
        return
    
    print(f"Found {len(data_dirs)} data sessions:")
    total_samples = 0
    for i, data_dir in enumerate(data_dirs, 1):
        labels_file = os.path.join(data_dir, 'labels.json')
        with open(labels_file, 'r') as f:
            sample_count = len(json.load(f))
        total_samples += sample_count
        print(f"  {i}. {os.path.basename(data_dir)}: {sample_count} samples")
    
    print(f"Total samples available: {total_samples}")
    
    if total_samples < 50:
        print("?? Warning: Very few samples! Consider collecting more data")
        print("Minimum recommended: 500+ samples")
    
    # Create combined dataset
    print("\nCombining datasets...")
    all_samples = create_combined_dataset(data_dirs)
    
    # Train/validation split
    np.random.shuffle(all_samples)
    split_idx = int(0.8 * len(all_samples))
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]
    
    print(f"Training samples: {len(train_samples)}")
    print(f"Validation samples: {len(val_samples)}")
    
    # Create datasets
    train_dataset = CombinedPilotNetDataset(train_samples)
    val_dataset = CombinedPilotNetDataset(val_samples)
    
    # Data loaders
    batch_size = 16 if len(train_samples) > 100 else 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Batch size: {batch_size}")
    
    # Model
    model = ImprovedPilotNet().to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
    
    # Training tracking
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 20
    
    epochs = 100 if len(train_samples) > 100 else 50
    print(f"Training for {epochs} epochs with early stopping")
    
    print("\n" + "="*50)
    print("TRAINING STARTED")
    print("="*50)
    
    start_time = time.time()
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_steering_loss = 0.0
        train_throttle_loss = 0.0
        train_brake_loss = 0.0
        
        for batch in train_loader:
            images = batch['image'].to(device)
            targets = {
                'steering': batch['steering'].to(device),
                'throttle': batch['throttle'].to(device),
                'brake': batch['brake'].to(device)
            }
            
            optimizer.zero_grad()
            outputs = model(images)
            
            # Multi-task loss
            loss_steering = criterion(outputs['steering'], targets['steering'])
            loss_throttle = criterion(outputs['throttle'], targets['throttle'])
            loss_brake = criterion(outputs['brake'], targets['brake'])
            
            # Weighted loss (steering is most important for PilotNet)
            total_loss = 2.0 * loss_steering + 1.0 * loss_throttle + 1.0 * loss_brake
            
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += total_loss.item()
            train_steering_loss += loss_steering.item()
            train_throttle_loss += loss_throttle.item()
            train_brake_loss += loss_brake.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_steering_loss = 0.0
        val_throttle_loss = 0.0
        val_brake_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                targets = {
                    'steering': batch['steering'].to(device),
                    'throttle': batch['throttle'].to(device),
                    'brake': batch['brake'].to(device)
                }
                
                outputs = model(images)
                
                loss_steering = criterion(outputs['steering'], targets['steering'])
                loss_throttle = criterion(outputs['throttle'], targets['throttle'])
                loss_brake = criterion(outputs['brake'], targets['brake'])
                
                total_loss = 2.0 * loss_steering + 1.0 * loss_throttle + 1.0 * loss_brake
                
                val_loss += total_loss.item()
                val_steering_loss += loss_steering.item()
                val_throttle_loss += loss_throttle.item()
                val_brake_loss += loss_brake.item()
        
        # Calculate averages
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_steering_loss /= len(train_loader)
        val_steering_loss /= len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print progress
        if epoch % 5 == 0 or epoch < 10:
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                  f"Steer: {val_steering_loss:.4f} | LR: {current_lr:.2e}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save checkpoint
            os.makedirs('checkpoints', exist_ok=True)
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_val_loss': best_val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses
            }
            torch.save(checkpoint, 'checkpoints/pilotnet_best.pt')
            
            if epoch > 10:  # Don't print for every early epoch
                print(f"? New best model saved! Val loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= max_patience:
            print(f"Early stopping after {epoch+1} epochs (patience: {max_patience})")
            break
        
        # Stop if learning rate gets too small
        if current_lr < 1e-6:
            print("Learning rate too small, stopping")
            break
    
    # Training completed
    elapsed = time.time() - start_time
    print("\n" + "="*50)
    print("? TRAINING COMPLETED")
    print("="*50)
    print(f"Training time: {elapsed:.1f} seconds")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final epoch: {epoch+1}")
    print(f"Model saved: checkpoints/pilotnet_best.pt")
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(val_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('checkpoints/pilotnet_training_history.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("? Training plots saved to checkpoints/pilotnet_training_history.png")
    
    return model, best_val_loss

if __name__ == "__main__":
    train_pilotnet()