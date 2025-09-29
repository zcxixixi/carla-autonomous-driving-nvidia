"""
NVIDIA PilotNet Implementation for CARLA
Based on the 2016 NVIDIA End to End Learning paper
Fixed architecture for proper dimension handling
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class NVIDIAPilotNet(nn.Module):
    """
    Original NVIDIA PilotNet architecture (2016)
    Input: 224x224x3 RGB image
    Output: Single steering angle
    """
    def __init__(self):
        super(NVIDIAPilotNet, self).__init__()
        
        # Convolutional layers (as in NVIDIA paper)
        self.conv1 = nn.Conv2d(3, 24, kernel_size=5, stride=2)  
        self.conv2 = nn.Conv2d(24, 36, kernel_size=5, stride=2) 
        self.conv3 = nn.Conv2d(36, 48, kernel_size=5, stride=2) 
        self.conv4 = nn.Conv2d(48, 64, kernel_size=3, stride=1) 
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1) 
        
        # Calculate flattened size dynamically
        self._get_conv_output_size()
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.conv_output_size, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc4 = nn.Linear(10, 1)      # Single steering output
        
        self.dropout = nn.Dropout(0.5)
    
    def _get_conv_output_size(self):
        """Calculate the output size of convolutional layers"""
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 224, 224)
            x = F.elu(self.conv1(dummy_input))
            x = F.elu(self.conv2(x))
            x = F.elu(self.conv3(x))
            x = F.elu(self.conv4(x))
            x = F.elu(self.conv5(x))
            self.conv_output_size = x.view(1, -1).size(1)
            print(f"Conv output size: {self.conv_output_size}")
    
    def forward(self, x):
        # Normalize input
        x = x / 255.0
        
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
    Enhanced PilotNet with multiple outputs
    Outputs: steering, throttle, brake
    """
    def __init__(self):
        super(ImprovedPilotNet, self).__init__()
        
        # Shared convolutional backbone
        self.conv1 = nn.Conv2d(3, 24, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(24, 36, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(36, 48, kernel_size=5, stride=2)
        self.conv4 = nn.Conv2d(48, 64, kernel_size=3, stride=1)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate flattened size
        self._get_conv_output_size()
        
        # Shared fully connected layers
        self.fc1 = nn.Linear(self.conv_output_size, 200)
        self.fc2 = nn.Linear(200, 100)
        self.dropout = nn.Dropout(0.5)
        
        # Task-specific heads
        self.steering_head = nn.Sequential(
            nn.Linear(100, 50),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(50, 1)
        )
        
        self.throttle_head = nn.Sequential(
            nn.Linear(100, 50),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(50, 1),
            nn.Sigmoid()  # Throttle should be [0, 1]
        )
        
        self.brake_head = nn.Sequential(
            nn.Linear(100, 50),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(50, 1),
            nn.Sigmoid()  # Brake should be [0, 1]
        )
    
    def _get_conv_output_size(self):
        """Calculate the output size of convolutional layers"""
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 224, 224)
            x = F.elu(self.conv1(dummy_input))
            x = F.elu(self.conv2(x))
            x = F.elu(self.conv3(x))
            x = F.elu(self.conv4(x))
            x = F.elu(self.conv5(x))
            self.conv_output_size = x.view(1, -1).size(1)
    
    def forward(self, x):
        # Normalize input
        x = x / 255.0
        
        # Shared convolutional layers
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        x = F.elu(self.conv5(x))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Shared fully connected layers
        x = self.dropout(F.elu(self.fc1(x)))
        x = self.dropout(F.elu(self.fc2(x)))
        
        # Task-specific outputs
        steering = self.steering_head(x)
        throttle = self.throttle_head(x)
        brake = self.brake_head(x)
        
        return {
            'steering': steering,
            'throttle': throttle,
            'brake': brake
        }


def count_parameters(model):
    """Count total trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_model_architectures():
    """Test both model architectures"""
    print("Testing NVIDIA PilotNet architectures...")
    
    # Test input (batch_size=2, channels=3, height=224, width=224)
    test_input = torch.randn(2, 3, 224, 224)
    
    # Test original PilotNet
    print("\n1. Original PilotNet:")
    pilotnet = NVIDIAPilotNet()
    print(f"   Parameters: {count_parameters(pilotnet):,}")
    
    with torch.no_grad():
        steering = pilotnet(test_input)
        print(f"   Input shape: {test_input.shape}")
        print(f"   Steering output shape: {steering.shape}")
        print(f"   Steering range: [{steering.min():.3f}, {steering.max():.3f}]")
    
    # Test improved PilotNet
    print("\n2. Improved PilotNet:")
    improved = ImprovedPilotNet()
    print(f"   Parameters: {count_parameters(improved):,}")
    
    with torch.no_grad():
        outputs = improved(test_input)
        print(f"   Input shape: {test_input.shape}")
        for key, value in outputs.items():
            print(f"   {key.capitalize()} shape: {value.shape}, range: [{value.min():.3f}, {value.max():.3f}]")
    
    print("\n? Both architectures working correctly!")
    
    # Test with different input sizes
    print("\n3. Testing different input sizes:")
    test_sizes = [(1, 3, 224, 224), (4, 3, 224, 224), (8, 3, 224, 224)]
    
    for size in test_sizes:
        test_batch = torch.randn(*size)
        with torch.no_grad():
            try:
                out1 = pilotnet(test_batch)
                out2 = improved(test_batch)
                print(f"   ? Batch size {size[0]}: OK")
            except Exception as e:
                print(f"   ? Batch size {size[0]}: {e}")


if __name__ == "__main__":
    test_model_architectures()