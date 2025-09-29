# -*- coding: utf-8 -*-
"""
Accident-Aware End-to-End Learning
Based on NVIDIA architecture with accident avoidance learning.

Key innovation: Learn to avoid accidents by training on negative samples 
after standard end-to-end learning.
Author: Research Project
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader

def rgb_to_yuv(rgb_image):
    """
    Convert RGB image to YUV color space (as in NVIDIA paper).
    Paper: "The input image is split into YUV planes"
    """
    yuv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2YUV)
    return yuv_image

def preprocess_nvidia_image(image, target_size=(200, 66)):
    """
    Preprocess image as described in NVIDIA paper:
    1. Crop (remove sky and car hood)
    2. Resize to 66x200
    3. Convert to YUV color space
    """
    # 1. Crop image (remove top sky and bottom car hood)
    height, width = image.shape[:2]
    crop_top = int(0.35 * height)
    crop_bottom = int(0.85 * height)
    cropped = image[crop_top:crop_bottom, :, :]
    
    # 2. Resize (NVIDIA uses 66x200)
    resized = cv2.resize(cropped, target_size)
    
    # 3. Convert to YUV (key step in paper)
    yuv_image = rgb_to_yuv(resized)
    
    return yuv_image.astype(np.float32)

class AccidentAwareNVIDIANet(nn.Module):
    """
    Enhanced NVIDIA network supporting accident avoidance learning.
    Follows original paper: 9 layers (1 normalization + 5 conv + 3 fc).
    """
    
    def __init__(self):
        super(AccidentAwareNVIDIANet, self).__init__()
        
        # Layer 1: Image normalization (hard-coded as in NVIDIA paper)
        # Implemented in forward: (pixel_value / 127.5) - 1.0
        
        # Base NVIDIA conv layers (as in paper)
        # Input: YUV 3-channel image (66x200)
        self.feature_extractor = nn.Sequential(
            # First 3 conv layers: 5x5 kernel, 2x2 stride (feature extraction)
            nn.Conv2d(3, 24, kernel_size=5, stride=2),    # Paper Layer 2
            nn.ReLU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2),   # Paper Layer 3  
            nn.ReLU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2),   # Paper Layer 4
            nn.ReLU(),
            # Last 2 conv layers: 3x3 kernel, no stride (fine features)
            nn.Conv2d(48, 64, kernel_size=3, stride=1),   # Paper Layer 5
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),   # Paper Layer 6
            nn.ReLU(),
        )
        
        # 3 fully connected layers (Paper Layer 7-9)
        # Paper designed as controller, outputs inverse turning radius
        self.fc1 = nn.Linear(1152, 1164)          # Paper Layer 7
        self.fc2 = nn.Linear(1164, 100)           # Paper Layer 8  
        self.fc3 = nn.Linear(100, 50)             # Paper Layer 9 first part
        self.fc4 = nn.Linear(50, 10)              # Paper Layer 9 second part
        
        # Output layers - our innovation extension
        self.steering_head = nn.Linear(10, 1)      # Inverse turning radius (paper original output)
        self.safety_head = nn.Linear(10, 1)       # Safety score (our innovation)
        
        # Dropout layers (mentioned in paper for overfitting prevention)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # Layer 1: Image normalization (hard-coded as in NVIDIA paper)
        # Convert pixel values from [0,255] to [-1,1]
        x = x / 127.5 - 1.0
        
        # Layer 2-6: Convolutional feature extraction
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)  # Flatten
        
        # Layer 7-9: Fully connected controller layers
        x = torch.relu(self.fc1(features))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))  
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        
        # Output layers
        # Paper output: inverse turning radius (we use steering angle for interpretability)
        steering = self.steering_head(x)
        
        # Our innovation: safety score
        safety_score = torch.sigmoid(self.safety_head(x))  # [0,1]
        
        return steering, safety_score

class AccidentAwareDataset(Dataset):
    """
    Dataset supporting both positive and negative samples.
    """
    
    def __init__(self, data_dir, include_accidents=False):
        self.data_dir = data_dir
        self.include_accidents = include_accidents
        
        # Normal driving data
        self.normal_samples = self.load_normal_samples()
        
        # Accident data (if included)
        self.accident_samples = []
        if include_accidents:
            self.accident_samples = self.load_accident_samples()
    
    def load_normal_samples(self):
        """
        Load normal driving samples.
        Implements data augmentation as described in the NVIDIA paper.
        """
        samples = []
        
        # Generate some simulated normal driving data for testing
        for i in range(200):  # More normal samples
            # Create simulated normal driving image
            fake_image = np.random.randint(0, 255, (66, 200, 3), dtype=np.uint8)
            processed_image = preprocess_nvidia_image(fake_image)
            
            # Simulate normal driving steering angles (smaller range)
            normal_steering = np.random.uniform(-0.3, 0.3)
            
            samples.append((processed_image, normal_steering))
            
            # Paper data augmentation: left/right camera simulation
            # Left camera (add right turn correction)
            left_correction = 0.2
            samples.append((processed_image, normal_steering + left_correction))
            
            # Right camera (add left turn correction)  
            right_correction = -0.2
            samples.append((processed_image, normal_steering + right_correction))
        
        print(f"Generated {len(samples)} simulated normal driving samples")
        return samples
    
    def load_accident_samples(self):
        """Load accident samples (negative examples)"""  
        samples = []
        # Generate some simulated accident data for testing
        for i in range(50):  # Fewer accident samples
            # Create simulated dangerous scenario image
            fake_image = np.random.randint(0, 255, (66, 200, 3), dtype=np.uint8)
            processed_image = preprocess_nvidia_image(fake_image)
            
            # Simulate dangerous situation steering angles (larger range)
            dangerous_steering = np.random.uniform(-0.8, 0.8)
            
            samples.append((processed_image, dangerous_steering))
        
        print(f"Generated {len(samples)} simulated accident samples")
        return samples
    
    def __len__(self):
        return len(self.normal_samples) + len(self.accident_samples)
    
    def __getitem__(self, idx):
        if idx < len(self.normal_samples):
            # Normal sample
            image, steering = self.normal_samples[idx] 
            safety_label = 1.0  # Safe
        else:
            # Accident sample
            accident_idx = idx - len(self.normal_samples)
            image, steering = self.accident_samples[accident_idx]
            safety_label = 0.0  # Dangerous
        
        # Convert to PyTorch tensors
        image_tensor = torch.FloatTensor(image).permute(2, 0, 1)  # (H,W,C) -> (C,H,W)
        steering_tensor = torch.FloatTensor([steering])
        safety_tensor = torch.FloatTensor([safety_label])
        
        return image_tensor, steering_tensor, safety_tensor

class AccidentAwareLoss(nn.Module):
    """
    Combined loss for steering prediction and safety classification.
    """
    
    def __init__(self, alpha=1.0, beta=1.0):
        super(AccidentAwareLoss, self).__init__()
        self.alpha = alpha  # Steering loss weight
        self.beta = beta    # Safety loss weight
        
        self.steering_loss = nn.MSELoss()
        self.safety_loss = nn.BCELoss()
    
    def forward(self, pred_steering, pred_safety, true_steering, true_safety):
        # Steering prediction loss
        L_steering = self.steering_loss(pred_steering, true_steering)
        
        # Safety prediction loss - fix dimension mismatch
        L_safety = self.safety_loss(pred_safety.squeeze(), true_safety.squeeze())
        
        # Combined loss
        total_loss = self.alpha * L_steering + self.beta * L_safety
        
        return total_loss, L_steering, L_safety

class AccidentAwareTrainer:
    """
    Two-stage trainer: baseline end-to-end, then accident avoidance fine-tuning.
    """
    
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=0.0001)
        # Use MSE as main steering loss (as in paper)
        self.criterion = AccidentAwareLoss()
        
    def stage1_baseline_training(self, normal_dataloader, epochs=20):
        """
        Stage 1: Baseline NVIDIA end-to-end learning.
        Only normal driving data is used.
        """
        print("=== Stage 1: Baseline end-to-end learning ===")
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_idx, (images, steering, safety) in enumerate(normal_dataloader):
                images = images.to(self.device)
                steering = steering.to(self.device)
                safety = safety.to(self.device)
                
                self.optimizer.zero_grad()
                
                # Network outputs: steering + safety score
                pred_steering, pred_safety = self.model(images)
                
                # In stage 1, focus on steering prediction (MSE loss)
                # Paper outputs inverse turning radius; we use steering angle for interpretability
                loss, steering_loss, safety_loss = self.criterion(
                    pred_steering, pred_safety, steering, safety
                )
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                
                if batch_idx % 10 == 0:
                    print(f'Epoch {epoch}, Batch {batch_idx}: '
                          f'Total Loss = {loss.item():.4f}, '
                          f'Steering Loss = {steering_loss.item():.4f}')
            
            print(f'Epoch {epoch} completed. Average Loss: {total_loss/len(normal_dataloader):.4f}')
        
        print("Stage 1 training complete - baseline end-to-end model ready!")
        
    def stage2_accident_learning(self, mixed_dataloader, epochs=10):
        """
        Stage 2: Accident avoidance learning.
        Use mixed normal and accident data, focus on safety prediction.
        """
        print("\n=== Stage 2: Accident avoidance learning ===")
        
        # Adjust learning rate for fine-tuning
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = 0.00001  # Lower learning rate
        
        # Adjust loss weights, emphasize safety more
        self.criterion.alpha = 0.5  # Lower steering weight
        self.criterion.beta = 2.0   # Higher safety weight
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            safety_correct = 0
            total_samples = 0
            
            for batch_idx, (images, steering, safety) in enumerate(mixed_dataloader):
                images = images.to(self.device)
                steering = steering.to(self.device)
                safety = safety.to(self.device)
                
                self.optimizer.zero_grad()
                
                pred_steering, pred_safety = self.model(images)
                
                loss, steering_loss, safety_loss = self.criterion(
                    pred_steering, pred_safety, steering, safety
                )
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                
                # Calculate safety prediction accuracy
                safety_pred_binary = (pred_safety.squeeze() > 0.5).float()
                safety_correct += (safety_pred_binary == safety).sum().item()
                total_samples += safety.size(0)
                
                if batch_idx % 10 == 0:
                    print(f'Epoch {epoch}, Batch {batch_idx}: '
                          f'Total Loss = {loss.item():.4f}, '
                          f'Safety Loss = {safety_loss.item():.4f}, '
                          f'Safety Acc = {safety_correct/total_samples:.3f}')
            
            print(f'Epoch {epoch} completed. '
                  f'Average Loss: {total_loss/len(mixed_dataloader):.4f}, '
                  f'Safety Accuracy: {safety_correct/total_samples:.3f}')
        
        print("Stage 2 training complete - accident avoidance model ready!")

# CARLAAccidentGenerator removed for standalone testing
# Will be implemented when CARLA integration is needed

def main():
    """
    Main training workflow - quick test version.
    All comments and docstrings are in English only.
    """
    print("=== Accident-Aware NVIDIA System - Quick Validation ===")
    
    try:
        # 1. Create model and check device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Device: {device}")
        
        model = AccidentAwareNVIDIANet()
        trainer = AccidentAwareTrainer(model, device=device)
        
        print(f"Total model parameters: {sum(p.numel() for p in model.parameters())}")
        
        # 2. Prepare test data
        print("\n=== Data Preparation ===")
        
        # Stage 1: Only normal driving data
        normal_dataset = AccidentAwareDataset('data/normal_driving', include_accidents=False)
        normal_loader = DataLoader(normal_dataset, batch_size=16, shuffle=True)
        print(f"Normal driving dataset size: {len(normal_dataset)}")
        
        # Stage 2: Mixed normal + accident data  
        mixed_dataset = AccidentAwareDataset('data/mixed_training', include_accidents=True)
        mixed_loader = DataLoader(mixed_dataset, batch_size=16, shuffle=True)
        print(f"Mixed dataset size: {len(mixed_dataset)}")
        
        # 3. Quick two-stage training (few epochs for smoke test)
        print("\n=== Training ===")
        trainer.stage1_baseline_training(normal_loader, epochs=2)  # Only 2 epochs for test
        trainer.stage2_accident_learning(mixed_loader, epochs=1)   # Only 1 epoch for test
        
        # 4. Save model
        import os
        os.makedirs('models', exist_ok=True)
        torch.save(model.state_dict(), 'models/accident_aware_nvidia_test.pth')
        print("\n=== Test Complete! ===")
        print("Model saved to: models/accident_aware_nvidia_test.pth")
        
        # 5. Simple inference test
        print("\n=== Inference Test ===")
        model.eval()
        with torch.no_grad():
            # Create a test sample
            test_image = torch.randn(1, 3, 66, 200).to(device)
            steering, safety = model(test_image)
            print(f"Test output - Steering: {steering.item():.4f}, Safety: {safety.item():.4f}")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()