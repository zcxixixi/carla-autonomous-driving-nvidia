"""Minimal PyTorch skeleton for Conditional Affordance Learning (CAL).

This file provides a lightweight model skeleton with a backbone, conditional
injection of high-level command, multiple affordance heads and an additional
risk head for accident-aware learning. It is intentionally minimal and
designed to be extended.
"""
from typing import Dict
import torch
import torch.nn as nn


class SimpleBackbone(nn.Module):
    def __init__(self, in_channels=3, feature_dim=256):
        super().__init__()
        # Very small conv stack as a placeholder
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
        )
        self.fc = nn.Linear(64, feature_dim)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ConditionalAffordanceNet(nn.Module):
    def __init__(self, feature_dim=256, affordance_dims: Dict[str, int]=None):
        super().__init__()
        if affordance_dims is None:
            affordance_dims = {
                'front_distance': 1,
                'lane_offset': 1,
                'traffic_light': 3,   # categorical
                'speed_limit': 1,
                'risk_score': 1
            }

        self.backbone = SimpleBackbone(feature_dim=feature_dim)

        # A small embedding for high-level command
        self.command_emb = nn.Embedding(4, 32)  # left/right/straight/follow

        # Heads
        self.heads = nn.ModuleDict()
        for name, dim in affordance_dims.items():
            self.heads[name] = nn.Sequential(
                nn.Linear(feature_dim + 32, 128),
                nn.ReLU(),
                nn.Linear(128, dim)
            )

    def forward(self, image, command_index: torch.LongTensor):
        """Forward pass.

        image: tensor [B, C, H, W]
        command_index: tensor [B] with values 0..3
        returns dict of affordance outputs
        """
        feats = self.backbone(image)  # [B, feature_dim]
        cmd = self.command_emb(command_index)  # [B, 32]
        fused = torch.cat([feats, cmd], dim=1)

        outputs = {}
        for name, head in self.heads.items():
            out = head(fused)
            outputs[name] = out

        return outputs


if __name__ == '__main__':
    # Quick smoke test
    net = ConditionalAffordanceNet()
    x = torch.randn(2, 3, 224, 224)
    cmd = torch.tensor([0,1], dtype=torch.long)
    y = net(x, cmd)
    for k,v in y.items():
        print(k, v.shape)
