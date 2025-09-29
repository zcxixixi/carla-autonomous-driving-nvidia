"""GPU-Optimized Training Script for Accident-Aware Driving Model"""
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from algorithms.cal_model import ConditionalAffordanceNet
import time


class GPUCarlaDataset(Dataset):
    """Optimized dataset for GPU training"""
    def __init__(self, data_dir, max_samples=None):
        self.data_dir = data_dir
        self.rgb_dir = os.path.join(data_dir, 'rgb')
        self.metadata_dir = os.path.join(data_dir, 'metadata')
        
        # Find all valid samples
        self.valid_frames = []
        
        print("Scanning for valid data files...")
        for filename in os.listdir(self.metadata_dir):
            if not filename.endswith('.json'):
                continue
                
            frame_id = filename.replace('.json', '')
            img_path = os.path.join(self.rgb_dir, f'{frame_id}.png')
            meta_path = os.path.join(self.metadata_dir, filename)
            
            # Check file validity
            try:
                if (os.path.getsize(meta_path) > 100 and 
                    os.path.exists(img_path) and 
                    os.path.getsize(img_path) > 1000):
                    
                    # Quick JSON validation
                    with open(meta_path, 'r') as f:
                        json.load(f)
                    
                    # Quick image validation
                    try:
                        with Image.open(img_path) as img:
                            img.verify()  # Verify image integrity
                    except:
                        continue  # Skip corrupted images
                    
                    self.valid_frames.append(frame_id)
                    
                    if max_samples and len(self.valid_frames) >= max_samples:
                        break
            except:
                continue
        
        print(f"Found {len(self.valid_frames)} valid samples")
        
        # GPU-optimized transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Standard size for good performance
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])  # ImageNet normalization
        ])
    
    def __len__(self):
        return len(self.valid_frames)
    
    def __getitem__(self, idx):
        frame_id = self.valid_frames[idx]
        
        # Load and transform image with error handling
        img_path = os.path.join(self.rgb_dir, f'{frame_id}.png')
        try:
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
        except Exception as e:
            # Return a default black image if loading fails
            print(f"Warning: Failed to load image {frame_id}, using black image")
            image = torch.zeros(3, 224, 224)
        
        # Load metadata
        meta_path = os.path.join(self.metadata_dir, f'{frame_id}.json')
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
        
        # Extract targets
        front_distance = metadata.get('front_vehicle_distance_m', 50.0)
        if front_distance is None:
            front_distance = 50.0
        
        lane_offset = metadata.get('lane_offset_m', 0.0)
        if lane_offset is None:
            lane_offset = 0.0
            
        risk_score = metadata.get('risk_score', 0.0)
        if risk_score is None:
            risk_score = 0.0
        
        # High level command
        cmd_map = {'left': 0, 'right': 1, 'straight': 2, 'follow': 3}
        cmd_str = metadata.get('high_level_command', 'straight')
        command = cmd_map.get(cmd_str, 2)
        
        targets = {
            'front_distance': torch.tensor([front_distance], dtype=torch.float32),
            'lane_offset': torch.tensor([lane_offset], dtype=torch.float32),
            'risk_score': torch.tensor([risk_score], dtype=torch.float32)
        }
        
        return image, command, targets


def gpu_train():
    """GPU-accelerated training function"""
    # Check GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training device: {device}')
    
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')
        print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    else:
        print('Warning: CUDA not available. Install GPU PyTorch for faster training.')
        print('Command: conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia')
    
    # Load dataset
    dataset = GPUCarlaDataset('clean_research_data/scenario_00')
    
    if len(dataset) < 20:
        print("Need at least 20 samples for training")
        return None
    
    # Train/val split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # GPU-optimized data loaders (single worker to avoid image corruption issues)
    batch_size = 16 if torch.cuda.is_available() else 8
    num_workers = 0  # Single process to avoid corrupted image issues
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    print(f'Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}')
    print(f'Batch size: {batch_size}, Workers: {num_workers}')
    
    # Model with full affordances
    model = ConditionalAffordanceNet(
        feature_dim=256,
        affordance_dims={
            'front_distance': 1,
            'lane_offset': 1,
            'risk_score': 1
        }
    )
    model.to(device)
    
    # GPU-optimized training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    mse_loss = nn.MSELoss()
    
    # Mixed precision training for GPU
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    
    print("Starting GPU-optimized training...")
    
    epochs = 20  # More epochs for GPU training
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 7
    
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # Training
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for batch_idx, (images, commands, targets) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            commands = torch.tensor(commands, dtype=torch.long).to(device, non_blocking=True)
            
            # Move targets to GPU
            targets_gpu = {
                k: v.to(device, non_blocking=True) for k, v in targets.items()
            }
            
            optimizer.zero_grad()
            
            # Mixed precision forward pass
            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(images, commands)
                    loss = (mse_loss(outputs['front_distance'].squeeze(), targets_gpu['front_distance'].squeeze()) +
                           mse_loss(outputs['lane_offset'].squeeze(), targets_gpu['lane_offset'].squeeze()) +
                           mse_loss(outputs['risk_score'].squeeze(), targets_gpu['risk_score'].squeeze()))
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images, commands)
                loss = (mse_loss(outputs['front_distance'].squeeze(), targets_gpu['front_distance'].squeeze()) +
                       mse_loss(outputs['lane_offset'].squeeze(), targets_gpu['lane_offset'].squeeze()) +
                       mse_loss(outputs['risk_score'].squeeze(), targets_gpu['risk_score'].squeeze()))
                
                loss.backward()
                optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
            
            # Progress update
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for images, commands, targets in val_loader:
                images = images.to(device, non_blocking=True)
                commands = torch.tensor(commands, dtype=torch.long).to(device, non_blocking=True)
                
                targets_gpu = {
                    k: v.to(device, non_blocking=True) for k, v in targets.items()
                }
                
                if scaler:
                    with torch.cuda.amp.autocast():
                        outputs = model(images, commands)
                else:
                    outputs = model(images, commands)
                
                loss = (mse_loss(outputs['front_distance'].squeeze(), targets_gpu['front_distance'].squeeze()) +
                       mse_loss(outputs['lane_offset'].squeeze(), targets_gpu['lane_offset'].squeeze()) +
                       mse_loss(outputs['risk_score'].squeeze(), targets_gpu['risk_score'].squeeze()))
                
                val_loss += loss.item()
                val_batches += 1
        
        # Calculate averages
        avg_train_loss = train_loss / max(train_batches, 1)
        avg_val_loss = val_loss / max(val_batches, 1)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        epoch_time = time.time() - epoch_start
        
        print(f'Epoch {epoch+1}/{epochs}: Train={avg_train_loss:.4f}, Val={avg_val_loss:.4f}, Time={epoch_time:.1f}s')
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            os.makedirs('checkpoints', exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': avg_val_loss,
            }, 'checkpoints/gpu_trained_model.pt')
            
            print(f'? New best model saved! Val loss: {avg_val_loss:.4f}')
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= max_patience:
            print(f'Early stopping after {epoch+1} epochs (patience: {max_patience})')
            break
    
    total_time = time.time() - start_time
    
    print(f'\\n? GPU Training completed!')
    print(f'Total time: {total_time/60:.1f} minutes')
    print(f'Best validation loss: {best_val_loss:.4f}')
    print(f'Model saved: checkpoints/gpu_trained_model.pt')
    
    return model


if __name__ == '__main__':
    gpu_train()
