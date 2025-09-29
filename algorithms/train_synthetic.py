"""Minimal synthetic training loop for Conditional Affordance Net.

This script creates random synthetic data to validate the forward/backward
pass and a short training run. It is intended to be run locally in your
development environment where PyTorch is installed.
"""
import os
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from algorithms.cal_model import ConditionalAffordanceNet


class SyntheticAffordanceDataset(Dataset):
    def __init__(self, size=256, img_shape=(3,224,224)):
        self.size = size
        self.img_shape = img_shape

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Random image
        img = torch.randn(*self.img_shape)
        # Random command (0..3)
        cmd = torch.randint(0, 4, (1,)).item()
        # Targets
        front_distance = torch.tensor([random.uniform(0.5, 30.0)], dtype=torch.float32)
        lane_offset = torch.tensor([random.uniform(-2.0, 2.0)], dtype=torch.float32)
        traffic_light = torch.tensor(random.randint(0,2), dtype=torch.long)
        speed_limit = torch.tensor([random.choice([30,50,70])], dtype=torch.float32)
        risk_score = torch.tensor([random.random()], dtype=torch.float32)

        target = {
            'front_distance': front_distance,
            'lane_offset': lane_offset,
            'traffic_light': traffic_light,
            'speed_limit': speed_limit,
            'risk_score': risk_score
        }

        return img, cmd, target


def collate_fn(batch):
    imgs = torch.stack([b[0] for b in batch])
    cmds = torch.tensor([b[1] for b in batch], dtype=torch.long)
    targets = {k: torch.stack([b[2][k] if b[2][k].dim()>0 else b[2][k].unsqueeze(0) for b in batch])
               for k in batch[0][2].keys()}
    return imgs, cmds, targets


def train_one_epoch(model, loader, optim, device):
    model.train()
    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss()
    total_loss = 0.0
    for imgs, cmds, targets in loader:
        imgs = imgs.to(device)
        cmds = cmds.to(device)
        # Move targets
        t_front = targets['front_distance'].to(device).float()
        t_lane = targets['lane_offset'].to(device).float()
        t_tl = targets['traffic_light'].to(device).long().view(-1)
        t_risk = targets['risk_score'].to(device).float()

        optim.zero_grad()
        outputs = model(imgs, cmds)

        loss = 0.0
        # front_distance head
        if 'front_distance' in outputs:
            loss = loss + mse(outputs['front_distance'].view(-1,1), t_front)
        # lane_offset
        if 'lane_offset' in outputs:
            loss = loss + mse(outputs['lane_offset'].view(-1,1), t_lane)
        # traffic_light (assume logits)
        if 'traffic_light' in outputs:
            loss = loss + ce(outputs['traffic_light'].view(-1,3), t_tl)
        # risk_score
        if 'risk_score' in outputs:
            loss = loss + mse(outputs['risk_score'].view(-1,1), t_risk)

        loss.backward()
        optim.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ConditionalAffordanceNet()
    model.to(device)

    ds = SyntheticAffordanceDataset(size=256)
    loader = DataLoader(ds, batch_size=8, shuffle=True, collate_fn=collate_fn)

    optim = torch.optim.Adam(model.parameters(), lr=1e-4)

    epochs = 5
    for epoch in range(epochs):
        avg_loss = train_one_epoch(model, loader, optim, device)
        print(f'Epoch {epoch+1}/{epochs} avg_loss={avg_loss:.6f}')

    # Save checkpoint
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), 'checkpoints/cal_synthetic.pt')
    print('Saved checkpoint to checkpoints/cal_synthetic.pt')


if __name__ == '__main__':
    main()
