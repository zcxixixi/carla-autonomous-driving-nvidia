"""
NVIDIA PilotNet Implementation for CARLA
Classic end-to-end driving model from NVIDIA's 2016 paper
"End to End Learning for Self-Driving Cars"
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class NVIDIAPilotNet(nn.Module):
    """
    NVIDIA PilotNet architecture
    Input: 200x66 RGB image (or 224x224 resized)
    Output: Single steering angle
    """
    def __init__(self, input_channels=3):
        super(NVIDIAPilotNet, self).__init__()
        
        # Convolutional layers (as per NVIDIA paper)
        self.conv1 = nn.Conv2d(input_channels, 24, kernel_size=5, stride=2)  # 24@31x98
        self.conv2 = nn.Conv2d(24, 36, kernel_size=5, stride=2)              # 36@14x47  
        self.conv3 = nn.Conv2d(36, 48, kernel_size=5, stride=2)              # 48@5x22
        self.conv4 = nn.Conv2d(48, 64, kernel_size=3, stride=1)              # 64@3x20
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1)              # 64@1x18
        
        # Fully connected layers
        self.dropout = nn.Dropout(0.5)
        
        # Calculate the size after conv layers for 224x224 input
        # After conv layers: 64 * 26 * 26 = 43264 (for 224x224 input)
        self.fc1 = nn.Linear(64 * 26 * 26, 100)  # Adjusted for 224x224
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc4 = nn.Linear(10, 1)  # Single steering output
        
    def forward(self, x):
        # Normalize input to [-1, 1] as in original paper
        x = x / 255.0 * 2.0 - 1.0
        
        # Convolutional layers with ELU activation
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        x = F.elu(self.conv5(x))
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers with dropout
        x = self.dropout(F.elu(self.fc1(x)))
        x = self.dropout(F.elu(self.fc2(x)))
        x = self.dropout(F.elu(self.fc3(x)))
        x = self.fc4(x)  # No activation for final layer
        
        return x


class ImprovedPilotNet(nn.Module):
    """
    Enhanced version of PilotNet with additional outputs
    Outputs steering angle and speed control
    """
    def __init__(self, input_channels=3):
        super(ImprovedPilotNet, self).__init__()
        
        # Shared convolutional backbone
        self.conv1 = nn.Conv2d(input_channels, 24, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(24, 36, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(36, 48, kernel_size=5, stride=2)
        self.conv4 = nn.Conv2d(48, 64, kernel_size=3, stride=1)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        self.dropout = nn.Dropout(0.5)
        
        # Shared feature extraction
        self.shared_fc = nn.Linear(64 * 26 * 26, 512)
        
        # Steering head
        self.steering_fc1 = nn.Linear(512, 100)
        self.steering_fc2 = nn.Linear(100, 50)
        self.steering_output = nn.Linear(50, 1)
        
        # Speed/throttle head  
        self.speed_fc1 = nn.Linear(512, 100)
        self.speed_fc2 = nn.Linear(100, 50)
        self.speed_output = nn.Linear(50, 1)
        
        # Brake head
        self.brake_fc1 = nn.Linear(512, 50)
        self.brake_output = nn.Linear(50, 1)
        
    def forward(self, x):
        # Normalize input
        x = x / 255.0 * 2.0 - 1.0
        
        # Shared convolutional layers
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        x = F.elu(self.conv5(x))
        
        # Flatten and shared FC
        x = x.view(x.size(0), -1)
        shared_features = self.dropout(F.elu(self.shared_fc(x)))
        
        # Steering prediction
        steering = self.dropout(F.elu(self.steering_fc1(shared_features)))
        steering = self.dropout(F.elu(self.steering_fc2(steering)))
        steering = torch.tanh(self.steering_output(steering))  # [-1, 1]
        
        # Speed prediction
        speed = self.dropout(F.elu(self.speed_fc1(shared_features)))
        speed = self.dropout(F.elu(self.speed_fc2(speed)))
        speed = torch.sigmoid(self.speed_output(speed))  # [0, 1]
        
        # Brake prediction
        brake = self.dropout(F.elu(self.brake_fc1(shared_features)))
        brake = torch.sigmoid(self.brake_output(brake))  # [0, 1]
        
        return {
            'steering': steering,
            'throttle': speed,
            'brake': brake
        }


def test_model_architectures():
    """Test the model architectures"""
    print("Testing NVIDIA PilotNet architectures...")
    
    # Test input (batch_size=2, channels=3, height=224, width=224)
    test_input = torch.randn(2, 3, 224, 224)
    
    # Test original PilotNet
    print("\n1. Original PilotNet:")
    pilotnet = NVIDIAPilotNet()
    print(f"   Parameters: {sum(p.numel() for p in pilotnet.parameters()):,}")
    
    steering = pilotnet(test_input)
    print(f"   Input shape: {test_input.shape}")
    print(f"   Output shape: {steering.shape}")
    print(f"   Sample steering: {steering[0].item():.4f}")
    
    # Test improved version
    print("\n2. Improved PilotNet:")
    improved_net = ImprovedPilotNet()
    print(f"   Parameters: {sum(p.numel() for p in improved_net.parameters()):,}")
    
    outputs = improved_net(test_input)
    print(f"   Input shape: {test_input.shape}")
    print(f"   Steering shape: {outputs['steering'].shape}")
    print(f"   Throttle shape: {outputs['throttle'].shape}")
    print(f"   Brake shape: {outputs['brake'].shape}")
    print(f"   Sample outputs:")
    print(f"     Steering: {outputs['steering'][0].item():.4f}")
    print(f"     Throttle: {outputs['throttle'][0].item():.4f}")
    print(f"     Brake: {outputs['brake'][0].item():.4f}")
    
    print("\n? Both models working correctly!")


if __name__ == "__main__":
    test_model_architectures()