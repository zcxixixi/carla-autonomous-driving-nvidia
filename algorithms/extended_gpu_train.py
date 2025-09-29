"""
Extended GPU Training Script with Real-time Visualization
This version trains for much longer with better strategies and shows live training progress
"""
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import sys
import numpy as np
import time
import cv2
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import threading
from collections import deque

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simple_working_model import ConditionalAffordanceNet


class EnhancedCarlaDataset(Dataset):
    """Enhanced dataset with data augmentation and better preprocessing"""
    def __init__(self, data_dir, max_samples=None, augment=True):
        self.data_dir = data_dir
        self.rgb_dir = os.path.join(data_dir, 'rgb')
        self.metadata_dir = os.path.join(data_dir, 'metadata')
        self.augment = augment
        
        # Find all valid samples
        self.valid_frames = []
        
        print("Scanning for valid training data...")
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
                        data = json.load(f)
                        # Check required fields
                        if all(k in data for k in ['front_distance', 'lane_offset', 'risk_score', 'command']):
                            self.valid_frames.append(frame_id)
                            
            except Exception as e:
                continue
        
        if max_samples and len(self.valid_frames) > max_samples:
            self.valid_frames = self.valid_frames[:max_samples]
        
        print(f"Found {len(self.valid_frames)} valid training samples")
        
        # Data augmentation transforms
        if self.augment:
            self.transform = transforms.Compose([
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomHorizontalFlip(p=0.3),  # Careful with this for driving
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
            ])
        else:
            self.transform = None

    def __len__(self):
        return len(self.valid_frames)

    def __getitem__(self, idx):
        frame_id = self.valid_frames[idx]
        
        # Load image
        img_path = os.path.join(self.rgb_dir, f'{frame_id}.png')
        image = Image.open(img_path).convert('RGB')
        
        # Apply augmentation
        if self.transform and np.random.random() > 0.5:
            image = self.transform(image)
        
        # Resize and normalize
        image = image.resize((224, 224))
        image_array = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
        
        # Load metadata
        meta_path = os.path.join(self.metadata_dir, f'{frame_id}.json')
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
        
        # Extract targets with better normalization
        distance = float(metadata['front_distance'])
        lane_offset = float(metadata['lane_offset'])  
        risk_score = float(metadata['risk_score'])
        
        # Command mapping
        command_map = {'straight': 0, 'follow': 1, 'left': 2, 'right': 3}
        command = command_map.get(metadata['command'], 0)
        
        # Better target normalization
        distance = np.clip(distance / 50.0, 0.1, 2.0)  # Normalize to reasonable range
        lane_offset = np.clip(lane_offset / 3.0, -1.0, 1.0)  # Normalize lane offset
        risk_score = np.clip(risk_score, -1.0, 1.0)  # Keep risk in reasonable range
        
        return (image_tensor, 
                torch.tensor(command, dtype=torch.long),
                torch.tensor([distance], dtype=torch.float32),
                torch.tensor([lane_offset], dtype=torch.float32),
                torch.tensor([risk_score], dtype=torch.float32))


class TrainingVisualizer:
    """Handles real-time training visualization"""
    def __init__(self):
        self.train_losses = deque(maxlen=100)
        self.val_losses = deque(maxlen=100)
        self.current_sample = None
        self.current_predictions = None
        self.current_targets = None
        self.epoch = 0
        self.running = True
        
    def update_losses(self, train_loss, val_loss):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        
    def update_sample(self, image, predictions, targets, epoch):
        self.current_sample = image
        self.current_predictions = predictions
        self.current_targets = targets
        self.epoch = epoch
        
    def show_training_progress(self):
        """Display training progress in real-time"""
        while self.running:
            try:
                # Create display image
                if self.current_sample is not None:
                    # Convert tensor to displayable image
                    img = self.current_sample.cpu().numpy()
                    if len(img.shape) == 4:
                        img = img[0]  # Take first batch item
                    img = np.transpose(img, (1, 2, 0))
                    img = (img * 255).astype(np.uint8)
                    img = cv2.resize(img, (640, 480))
                    
                    # Create info overlay
                    info_img = np.zeros((480, 400, 3), dtype=np.uint8)
                    
                    # Display training info
                    y_offset = 30
                    cv2.putText(info_img, f"TRAINING PROGRESS", (10, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    y_offset += 40
                    
                    cv2.putText(info_img, f"Epoch: {self.epoch}", (10, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    y_offset += 30
                    
                    if len(self.train_losses) > 0:
                        cv2.putText(info_img, f"Train Loss: {self.train_losses[-1]:.4f}", (10, y_offset), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        y_offset += 30
                        
                    if len(self.val_losses) > 0:
                        cv2.putText(info_img, f"Val Loss: {self.val_losses[-1]:.4f}", (10, y_offset), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        y_offset += 40
                    
                    # Display current predictions vs targets
                    if self.current_predictions is not None and self.current_targets is not None:
                        pred_dist, pred_lane, pred_risk = self.current_predictions
                        targ_dist, targ_lane, targ_risk = self.current_targets
                        
                        cv2.putText(info_img, "PREDICTIONS vs TARGETS:", (10, y_offset), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                        y_offset += 30
                        
                        cv2.putText(info_img, f"Distance: {pred_dist:.2f} / {targ_dist:.2f}", (10, y_offset), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        y_offset += 25
                        
                        cv2.putText(info_img, f"Lane: {pred_lane:.3f} / {targ_lane:.3f}", (10, y_offset), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        y_offset += 25
                        
                        cv2.putText(info_img, f"Risk: {pred_risk:.3f} / {targ_risk:.3f}", (10, y_offset), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        y_offset += 40
                    
                    # Loss plot
                    if len(self.train_losses) > 10:
                        plot_height = 150
                        plot_y_start = 300
                        
                        # Draw loss plot background
                        cv2.rectangle(info_img, (10, plot_y_start), (390, plot_y_start + plot_height), (50, 50, 50), -1)
                        cv2.putText(info_img, "Loss History", (15, plot_y_start - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        
                        # Draw loss curves
                        train_losses_list = list(self.train_losses)
                        val_losses_list = list(self.val_losses)
                        
                        if len(train_losses_list) > 1:
                            max_loss = max(max(train_losses_list), max(val_losses_list) if val_losses_list else 0)
                            min_loss = min(min(train_losses_list), min(val_losses_list) if val_losses_list else 0)
                            loss_range = max_loss - min_loss if max_loss > min_loss else 1
                            
                            # Train loss (blue)
                            for i in range(1, len(train_losses_list)):
                                x1 = int(15 + (i-1) * 360 / len(train_losses_list))
                                y1 = int(plot_y_start + plot_height - ((train_losses_list[i-1] - min_loss) / loss_range) * plot_height)
                                x2 = int(15 + i * 360 / len(train_losses_list))
                                y2 = int(plot_y_start + plot_height - ((train_losses_list[i] - min_loss) / loss_range) * plot_height)
                                cv2.line(info_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                            
                            # Val loss (red)
                            if len(val_losses_list) > 1:
                                for i in range(1, len(val_losses_list)):
                                    x1 = int(15 + (i-1) * 360 / len(val_losses_list))
                                    y1 = int(plot_y_start + plot_height - ((val_losses_list[i-1] - min_loss) / loss_range) * plot_height)
                                    x2 = int(15 + i * 360 / len(val_losses_list))
                                    y2 = int(plot_y_start + plot_height - ((val_losses_list[i] - min_loss) / loss_range) * plot_height)
                                    cv2.line(info_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    
                    # Combine images
                    combined = np.hstack([img, info_img])
                    cv2.imshow("Training Progress", combined)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        self.running = False
                        break
                
                time.sleep(0.1)  # Update at 10 FPS
                
            except Exception as e:
                print(f"Visualization error: {e}")
                time.sleep(1)
        
        cv2.destroyAllWindows()
    
    def stop(self):
        self.running = False


def train_extended_model():
    """Train the model for extended time with better strategies and real-time visualization"""
    
    print("="*60)
    print("EXTENDED GPU TRAINING WITH REAL-TIME VISUALIZATION")
    print("="*60)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load data
    data_dir = 'training_data/scenario_00'
    
    # Create train/validation split
    full_dataset = EnhancedCarlaDataset(data_dir, augment=True)
    
    if len(full_dataset) < 100:
        print(f"WARNING: Only {len(full_dataset)} samples found. Need more data for good training!")
    
    # Split into train/val
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    # Create validation dataset without augmentation
    val_dataset_clean = EnhancedCarlaDataset(data_dir, augment=False)
    val_indices = list(range(train_size, len(full_dataset)))
    val_dataset = torch.utils.data.Subset(val_dataset_clean, val_indices)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)
    
    # Create model
    model = ConditionalAffordanceNet().to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Better loss function - weighted MSE
    class WeightedMSELoss(nn.Module):
        def __init__(self):
            super().__init__()
            
        def forward(self, distance_pred, distance_true, lane_pred, lane_true, risk_pred, risk_true):
            # Weight losses based on importance for driving
            distance_loss = nn.MSELoss()(distance_pred, distance_true) * 2.0  # Distance is very important
            lane_loss = nn.MSELoss()(lane_pred, lane_true) * 1.5  # Lane keeping is important
            risk_loss = nn.MSELoss()(risk_pred, risk_true) * 1.0  # Risk awareness
            
            return distance_loss + lane_loss + risk_loss
    
    criterion = WeightedMSELoss()
    
    # Better optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
    
    # Training parameters
    num_epochs = 100  # Much longer training
    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0
    
    # Training history
    train_losses = []
    val_losses = []
    
    # Initialize visualizer
    visualizer = TrainingVisualizer()
    
    # Start visualization thread
    vis_thread = threading.Thread(target=visualizer.show_training_progress)
    vis_thread.daemon = True
    vis_thread.start()
    
    print(f"\nStarting extended training for {num_epochs} epochs...")
    print(f"Early stopping patience: {patience}")
    print("Real-time visualization window opened - press 'q' in window to stop training")
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for batch_idx, (images, commands, distances, lane_offsets, risks) in enumerate(train_loader):
            images = images.to(device)
            commands = commands.to(device)
            distances = distances.to(device)
            lane_offsets = lane_offsets.to(device)
            risks = risks.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            pred_distance, pred_lane, pred_risk = model(images, commands)
            
            # Calculate loss
            loss = criterion(pred_distance, distances, pred_lane, lane_offsets, pred_risk, risks)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
        
        avg_train_loss = train_loss / train_batches
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch_idx, (images, commands, distances, lane_offsets, risks) in enumerate(val_loader):
                images = images.to(device)
                commands = commands.to(device)
                distances = distances.to(device)
                lane_offsets = lane_offsets.to(device)
                risks = risks.to(device)
                
                # Forward pass
                pred_distance, pred_lane, pred_risk = model(images, commands)
                
                # Calculate loss
                loss = criterion(pred_distance, distances, pred_lane, lane_offsets, pred_risk, risks)
                
                val_loss += loss.item()
                val_batches += 1
                
                # Update visualizer with sample from validation (first batch only)
                if batch_idx == 0:
                    visualizer.update_sample(
                        images, 
                        (pred_distance[0].item(), pred_lane[0].item(), pred_risk[0].item()),
                        (distances[0].item(), lane_offsets[0].item(), risks[0].item()),
                        epoch + 1
                    )
        
        avg_val_loss = val_loss / val_batches
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        # Save training history
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # Update visualizer
        visualizer.update_losses(avg_train_loss, avg_val_loss)
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            elapsed = time.time() - start_time
            print(f"Epoch [{epoch+1:3d}/{num_epochs}] | "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f} | "
                  f"Time: {elapsed:.1f}s")
        
        # Early stopping and model saving
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            # Save best model
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'best_val_loss': best_val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses
            }
            
            torch.save(checkpoint, 'checkpoints/extended_model_best.pt')
            
            if (epoch + 1) % 10 == 0:
                print(f"? New best model saved! Val Loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= patience:
            print(f"\n? Early stopping triggered after {epoch + 1} epochs")
            print(f"Best validation loss: {best_val_loss:.4f}")
            break
            
        # Check if visualization window was closed
        if not visualizer.running:
            print(f"\n? Training stopped by user after {epoch + 1} epochs")
            break
    
    # Final training summary
    total_time = time.time() - start_time
    
    print("\n" + "="*60)
    print("EXTENDED TRAINING COMPLETED")
    print("="*60)
    print(f"Total epochs: {epoch + 1}")
    print(f"Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final train loss: {avg_train_loss:.4f}")
    print(f"Training samples used: {len(train_dataset)}")
    print(f"Model saved as: checkpoints/extended_model_best.pt")
    
    # Save final model too
    final_checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': avg_train_loss,
        'val_loss': avg_val_loss,
        'best_val_loss': best_val_loss,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'total_time': total_time
    }
    
    torch.save(final_checkpoint, 'checkpoints/extended_model_final.pt')
    print("Final model also saved as: checkpoints/extended_model_final.pt")
    
    # Stop visualizer
    visualizer.stop()
    
    return model, best_val_loss


if __name__ == "__main__":
    # Create checkpoints directory
    os.makedirs('checkpoints', exist_ok=True)
    
    # Train the model
    model, best_loss = train_extended_model()
    
    print(f"\n? Extended training complete! Best validation loss: {best_loss:.4f}")
    print("Ready for testing with the improved model!")