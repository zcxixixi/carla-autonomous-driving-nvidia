"""Simple working model that matches the trained checkpoint"""
import torch
import torch.nn as nn

class SimpleBackbone(nn.Module):
    def __init__(self, feature_dim=256):
        super().__init__()
        # Match the exact structure from checkpoint analysis
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2),  # Layer 0: 5x5 kernel
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), # Layer 2: 3x3 kernel  
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling to get 64 features
        )
        
        self.fc = nn.Linear(64, feature_dim)  # 64 input features to match checkpoint

    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x

class ConditionalAffordanceNet(nn.Module):
    def __init__(self, feature_dim=256):
        super().__init__()
        self.backbone = SimpleBackbone(feature_dim=feature_dim)
        
        # Command embedding for high-level commands
        self.command_emb = nn.Embedding(4, 32)  # left/right/straight/follow
        
        # Only the 3 heads that were actually trained
        self.heads = nn.ModuleDict({
            'front_distance': nn.Sequential(
                nn.Linear(feature_dim + 32, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            ),
            'lane_offset': nn.Sequential(
                nn.Linear(feature_dim + 32, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            ),
            'risk_score': nn.Sequential(
                nn.Linear(feature_dim + 32, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )
        })

    def forward(self, image, command_index):
        """Forward pass returning the 3 basic affordances"""
        feats = self.backbone(image)  # [B, feature_dim]
        cmd = self.command_emb(command_index)  # [B, 32]
        fused = torch.cat([feats, cmd], dim=1)
        
        # Get outputs from each head
        distance = self.heads['front_distance'](fused)
        lane_offset = self.heads['lane_offset'](fused)
        risk = self.heads['risk_score'](fused)
        
        return distance, lane_offset, risk

if __name__ == '__main__':
    # Test the model
    model = ConditionalAffordanceNet()
    x = torch.randn(2, 3, 224, 224)
    cmd = torch.tensor([0, 1], dtype=torch.long)
    
    distance, lane_offset, risk = model(x, cmd)
    print(f"Distance: {distance.shape}")
    print(f"Lane offset: {lane_offset.shape}")
    print(f"Risk: {risk.shape}")