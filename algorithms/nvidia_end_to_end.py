"""
CARLA End-to-End Deep Learning Driver
基于NVIDIA论文的简化实现

Author: Research Project
Date: 2024
Purpose: 学习端到端深度学习在自动驾驶中的应用
"""

import carla
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import time
from collections import deque
import matplotlib.pyplot as plt

class NVIDIANet(nn.Module):
    """
    NVIDIA端到端学习网络架构
    输入: RGB图像 (66, 200, 3)
    输出: 转向角 [-1, 1]
    """
    
    def __init__(self):
        super(NVIDIANet, self).__init__()
        
        # 图像预处理层
        self.normalize = nn.Lambda(lambda x: x / 127.5 - 1.0)
        
        # 卷积层 - 特征提取
        self.conv1 = nn.Conv2d(3, 24, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(24, 36, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(36, 48, kernel_size=5, stride=2)
        self.conv4 = nn.Conv2d(48, 64, kernel_size=3, stride=1)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # 全连接层 - 决策输出
        self.fc1 = nn.Linear(1152, 1164)  # 计算flatten后的维度
        self.fc2 = nn.Linear(1164, 100)
        self.fc3 = nn.Linear(100, 50)
        self.fc4 = nn.Linear(50, 10)
        self.fc5 = nn.Linear(10, 1)
        
        # Dropout防止过拟合
        self.dropout = nn.Dropout(0.5)
        
        # 激活函数
        self.relu = nn.ReLU()
        
    def forward(self, x):
        """前向传播"""
        # 输入形状: (batch, 3, 66, 200)
        
        # 卷积特征提取
        x = self.relu(self.conv1(x))  # (batch, 24, 31, 98)
        x = self.relu(self.conv2(x))  # (batch, 36, 14, 47) 
        x = self.relu(self.conv3(x))  # (batch, 48, 5, 22)
        x = self.relu(self.conv4(x))  # (batch, 64, 3, 20)
        x = self.relu(self.conv5(x))  # (batch, 64, 1, 18)
        
        # 展平
        x = x.view(x.size(0), -1)    # (batch, 1152)
        
        # 全连接层
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        
        # 输出转向角 (无激活函数，直接回归)
        steering = self.fc5(x)
        
        return steering

class DrivingDataset(Dataset):
    """
    驾驶数据集类
    用于加载和预处理收集到的驾驶数据
    """
    
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        
        # 加载数据文件列表
        self.image_files = []
        self.steering_angles = []
        
        # 读取数据索引文件
        index_file = os.path.join(data_dir, 'driving_log.csv')
        if os.path.exists(index_file):
            with open(index_file, 'r') as f:
                for line in f.readlines()[1:]:  # 跳过标题行
                    parts = line.strip().split(',')
                    if len(parts) >= 2:
                        self.image_files.append(parts[0])
                        self.steering_angles.append(float(parts[1]))
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # 加载图像
        img_path = os.path.join(self.data_dir, self.image_files[idx])
        image = cv2.imread(img_path)
        
        if image is None:
            # 如果图像加载失败，返回零图像
            image = np.zeros((66, 200, 3), dtype=np.uint8)
        else:
            # 预处理图像
            image = self.preprocess_image(image)
        
        # 获取转向角
        steering = self.steering_angles[idx]
        
        # 转换为张量
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.FloatTensor(image).permute(2, 0, 1)  # (H,W,C) -> (C,H,W)
        
        steering = torch.FloatTensor([steering])
        
        return image, steering
    
    def preprocess_image(self, image):
        """
        图像预处理 - 复现NVIDIA论文的处理流程
        """
        # 1. 转换颜色空间 (BGR -> RGB)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 2. 裁剪图像 (去除天空和车头)
        # 原图: (720, 1280, 3) -> 裁剪: (400, 1280, 3) 
        h, w = image.shape[:2]
        crop_top = int(0.35 * h)
        crop_bottom = int(0.85 * h)
        image = image[crop_top:crop_bottom, :, :]
        
        # 3. 调整大小到网络输入尺寸
        image = cv2.resize(image, (200, 66))
        
        # 4. 数据类型转换
        image = image.astype(np.float32)
        
        return image

class CarlaDataCollector:
    """
    CARLA数据收集器
    用于自动收集训练数据
    """
    
    def __init__(self, host='127.0.0.1', port=2000):
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        
        # 车辆和传感器
        self.vehicle = None
        self.camera = None
        self.collision_sensor = None
        
        # 数据存储
        self.images = []
        self.steering_angles = []
        self.data_dir = 'data/training_data'
        
        # 确保数据目录存在
        os.makedirs(self.data_dir, exist_ok=True)
        
    def setup_vehicle(self):
        """设置车辆和传感器"""
        # 获取车辆蓝图
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
        
        # 生成车辆
        spawn_points = self.world.get_map().get_spawn_points()
        spawn_point = spawn_points[0]
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        
        # 设置摄像头
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '1280')
        camera_bp.set_attribute('image_size_y', '720')
        camera_bp.set_attribute('fov', '90')
        
        # 摄像头位置 (与NVIDIA论文类似)
        camera_transform = carla.Transform(
            carla.Location(x=2.0, z=1.4),  # 车前2米，高1.4米
            carla.Rotation(pitch=-15)       # 向下15度
        )
        
        self.camera = self.world.spawn_actor(
            camera_bp, camera_transform, attach_to=self.vehicle
        )
        
        # 设置图像回调
        self.camera.listen(self.process_image)
        
    def process_image(self, image):
        """处理接收到的图像"""
        # 转换CARLA图像格式
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))[:, :, :3]
        
        # 获取当前转向角
        control = self.vehicle.get_control()
        steering_angle = control.steer  # CARLA转向角范围 [-1, 1]
        
        # 存储数据
        timestamp = int(time.time() * 1000)
        img_filename = f'image_{timestamp}.png'
        img_path = os.path.join(self.data_dir, img_filename)
        
        # 保存图像
        cv2.imwrite(img_path, cv2.cvtColor(array, cv2.COLOR_RGB2BGR))
        
        # 记录数据
        self.images.append(img_filename)
        self.steering_angles.append(steering_angle)
        
        print(f"Collected sample {len(self.images)}: steering = {steering_angle:.3f}")
    
    def collect_data(self, duration=300):  # 默认收集5分钟数据
        """开始数据收集"""
        print("Starting data collection...")
        print("Use WASD keys to control the vehicle")
        print(f"Collecting for {duration} seconds...")
        
        # 启用autopilot进行数据收集
        self.vehicle.set_autopilot(True)
        
        start_time = time.time()
        while time.time() - start_time < duration:
            # 保持连接活跃
            self.world.tick()
            time.sleep(0.1)
        
        # 停止收集
        print(f"Data collection finished. Collected {len(self.images)} samples.")
        self.save_driving_log()
        
    def save_driving_log(self):
        """保存驾驶日志文件"""
        log_file = os.path.join(self.data_dir, 'driving_log.csv')
        with open(log_file, 'w') as f:
            f.write('image,steering\n')
            for img, steering in zip(self.images, self.steering_angles):
                f.write(f'{img},{steering}\n')
        print(f"Driving log saved to {log_file}")
    
    def cleanup(self):
        """清理资源"""
        if self.camera:
            self.camera.destroy()
        if self.vehicle:
            self.vehicle.destroy()

class EndToEndTrainer:
    """
    端到端学习训练器
    """
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        
        # 优化器和损失函数
        self.optimizer = optim.Adam(model.parameters(), lr=0.0001)
        self.criterion = nn.MSELoss()
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self, dataloader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (images, steering) in enumerate(dataloader):
            images = images.to(self.device)
            steering = steering.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            predicted_steering = self.model(images)
            loss = self.criterion(predicted_steering, steering)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Batch {batch_idx}: Loss = {loss.item():.6f}')
        
        avg_loss = total_loss / len(dataloader)
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def validate(self, dataloader):
        """验证模型"""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for images, steering in dataloader:
                images = images.to(self.device)
                steering = steering.to(self.device)
                
                predicted_steering = self.model(images)
                loss = self.criterion(predicted_steering, steering)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        self.val_losses.append(avg_loss)
        return avg_loss
    
    def train(self, train_loader, val_loader, epochs=50):
        """完整训练流程"""
        print(f"Starting training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters())}")
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            print(f'\nEpoch {epoch+1}/{epochs}')
            
            # 训练
            train_loss = self.train_epoch(train_loader)
            
            # 验证
            val_loss = self.validate(val_loader)
            
            print(f'Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'models/best_nvidia_model.pth')
                print('Saved best model!')
        
        # 绘制训练曲线
        self.plot_training_curves()
    
    def plot_training_curves(self):
        """绘制训练曲线"""
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 1, 1)
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_curves.png')
        plt.show()

def main():
    """主函数 - 演示完整流程"""
    
    # 1. 数据收集阶段
    print("=== 阶段1: 数据收集 ===")
    collector = CarlaDataCollector()
    try:
        collector.setup_vehicle()
        collector.collect_data(duration=60)  # 收集1分钟数据用于演示
    except Exception as e:
        print(f"Data collection error: {e}")
    finally:
        collector.cleanup()
    
    # 2. 模型训练阶段
    print("\n=== 阶段2: 模型训练 ===")
    
    # 创建模型
    model = NVIDIANet()
    
    # 创建数据集
    dataset = DrivingDataset('data/training_data')
    
    if len(dataset) > 0:
        # 分割训练和验证集
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        # 数据加载器
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        # 训练模型
        trainer = EndToEndTrainer(model)
        trainer.train(train_loader, val_loader, epochs=10)
        
        print("Training completed!")
    else:
        print("No data found. Please collect data first.")

if __name__ == "__main__":
    # 确保必要目录存在
    os.makedirs('data/training_data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    main()