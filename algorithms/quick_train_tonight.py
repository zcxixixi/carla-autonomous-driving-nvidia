"""快速训练脚本 - 今晚就能出结果的简单版本"""
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from algorithms.cal_model import ConditionalAffordanceNet
import random


class SimpleCarlaDataset(Dataset):
    def __init__(self, data_dir, max_samples=200):
        self.data_dir = data_dir
        self.rgb_dir = os.path.join(data_dir, 'rgb')
        self.metadata_dir = os.path.join(data_dir, 'metadata')
        
        # 只取完好的文件，限制数量加速训练
        self.valid_frames = []
        
        print("扫描有效数据文件...")
        for filename in os.listdir(self.metadata_dir):
            if not filename.endswith('.json'):
                continue
                
            frame_id = filename.replace('.json', '')
            img_path = os.path.join(self.rgb_dir, f'{frame_id}.png')
            meta_path = os.path.join(self.metadata_dir, filename)
            
            # 检查文件大小和完整性
            try:
                if os.path.getsize(meta_path) > 100 and os.path.exists(img_path) and os.path.getsize(img_path) > 1000:
                    with open(meta_path, 'r') as f:
                        metadata = json.load(f)  # 测试JSON是否完整
                    
                    # 快速验证图像可以打开
                    with Image.open(img_path) as img:
                        img.verify()
                    
                    self.valid_frames.append(frame_id)
                    
                    if len(self.valid_frames) >= max_samples:
                        break
            except:
                continue
        
        print(f"找到 {len(self.valid_frames)} 个有效数据样本")
        
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),  # 更小的输入，训练更快
            transforms.ToTensor(),
        ])
    
    def __len__(self):
        return len(self.valid_frames)
    
    def __getitem__(self, idx):
        frame_id = self.valid_frames[idx]
        
        # 加载图像
        img_path = os.path.join(self.rgb_dir, f'{frame_id}.png')
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        # 加载标签
        meta_path = os.path.join(self.metadata_dir, f'{frame_id}.json')
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
        
        # 提取关键标签
        front_distance = metadata.get('front_vehicle_distance_m', 50.0)
        if front_distance is None:
            front_distance = 50.0
        
        risk_score = metadata.get('risk_score', 0.0)
        if risk_score is None:
            risk_score = 0.0
        
        command = 2  # 简化：都设为直行
        
        targets = {
            'front_distance': torch.tensor([front_distance], dtype=torch.float32),
            'risk_score': torch.tensor([risk_score], dtype=torch.float32)
        }
        
        return image, command, targets


def quick_train():
    """快速训练函数 - 适合今晚出结果"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 加载数据集（限制200个样本）
    dataset = SimpleCarlaDataset('clean_research_data/scenario_00', max_samples=200)
    
    if len(dataset) < 10:
        print("数据样本太少，无法训练")
        return None
    
    # 训练集和验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)  # 小批次
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    print(f'训练样本: {len(train_dataset)}, 验证样本: {len(val_dataset)}')
    
    # 简化模型 - 只预测两个关键affordances
    model = ConditionalAffordanceNet(
        feature_dim=128,
        affordance_dims={
            'front_distance': 1,
            'risk_score': 1
        }
    )
    model.to(device)
    
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # 更高学习率
    mse_loss = nn.MSELoss()
    
    print("开始快速训练（5轮）...")
    
    # 只训练5轮，快速出结果
    epochs = 5
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # 训练
        model.train()
        train_loss = 0.0
        batch_count = 0
        
        for images, commands, targets in train_loader:
            images = images.to(device)
            commands = torch.tensor([commands] if isinstance(commands, int) else commands, dtype=torch.long).to(device)
            
            optimizer.zero_grad()
            outputs = model(images, commands)
            
            # 只计算两个损失
            loss = mse_loss(outputs['front_distance'].squeeze(), targets['front_distance'].squeeze().to(device))
            loss += mse_loss(outputs['risk_score'].squeeze(), targets['risk_score'].squeeze().to(device))
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            batch_count += 1
        
        # 验证
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for images, commands, targets in val_loader:
                images = images.to(device)
                commands = torch.tensor([commands] if isinstance(commands, int) else commands, dtype=torch.long).to(device)
                
                outputs = model(images, commands)
                
                loss = mse_loss(outputs['front_distance'].squeeze(), targets['front_distance'].squeeze().to(device))
                loss += mse_loss(outputs['risk_score'].squeeze(), targets['risk_score'].squeeze().to(device))
                
                val_loss += loss.item()
                val_batches += 1
        
        avg_train_loss = train_loss / max(batch_count, 1)
        avg_val_loss = val_loss / max(val_batches, 1)
        
        print(f'轮次 {epoch+1}/{epochs}: 训练损失 = {avg_train_loss:.4f}, 验证损失 = {avg_val_loss:.4f}')
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(model.state_dict(), 'checkpoints/quick_model_tonight.pt')
            print(f'? 保存新的最佳模型，验证损失: {avg_val_loss:.4f}')
    
    print(f'\n? 快速训练完成！')
    print(f'最佳验证损失: {best_val_loss:.4f}')
    print(f'模型保存位置: checkpoints/quick_model_tonight.pt')
    
    # 测试一个样本
    model.eval()
    test_sample = dataset[0]
    with torch.no_grad():
        img, cmd, targets = test_sample
        img = img.unsqueeze(0).to(device)
        cmd = torch.tensor([cmd], dtype=torch.long).to(device)
        pred = model(img, cmd)
        
        print(f'\n? 模型测试:')
        print(f'真实前车距离: {targets["front_distance"].item():.2f}m')
        print(f'预测前车距离: {pred["front_distance"].item():.2f}m')
        print(f'真实风险评分: {targets["risk_score"].item():.3f}')
        print(f'预测风险评分: {pred["risk_score"].item():.3f}')
    
    return model


if __name__ == '__main__':
    quick_train()