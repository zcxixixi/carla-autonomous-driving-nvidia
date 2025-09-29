# CARLA Autonomous Driving with NVIDIA

张涔熙的科研项目 - 基于CARLA仿真器和NVIDIA技术的自动驾驶研究项目

## 项目概述 / Project Overview

This research project implements autonomous driving algorithms using the CARLA simulator with NVIDIA GPU acceleration. The project focuses on developing and testing self-driving car algorithms in a realistic simulation environment.

本研究项目使用CARLA仿真器和NVIDIA GPU加速技术实现自动驾驶算法。项目专注于在真实的仿真环境中开发和测试自动驾驶汽车算法。

## 功能特性 / Features

- **CARLA仿真环境** / CARLA Simulation Environment
- **NVIDIA GPU加速** / NVIDIA GPU Acceleration
- **深度学习感知模块** / Deep Learning Perception Modules
- **路径规划算法** / Path Planning Algorithms
- **车辆控制系统** / Vehicle Control Systems
- **数据收集与处理** / Data Collection and Processing
- **性能评估框架** / Performance Evaluation Framework

## 系统要求 / System Requirements

- Python 3.7+
- CARLA Simulator 0.9.10+
- NVIDIA GPU with CUDA support
- PyTorch
- OpenCV
- NumPy

## 安装指南 / Installation Guide

1. Clone the repository:
```bash
git clone https://github.com/zcxixixi/carla-autonomous-driving-nvidia.git
cd carla-autonomous-driving-nvidia
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download and setup CARLA:
```bash
# Follow CARLA installation instructions
# https://carla.readthedocs.io/en/latest/start_quickstart/
```

## 使用方法 / Usage

### 启动仿真 / Start Simulation
```bash
python src/simulation/carla_simulator.py
```

### 运行自动驾驶 / Run Autonomous Driving
```bash
python src/autonomous_driving/main.py
```

### 数据收集 / Data Collection
```bash
python src/data_collection/collect_data.py
```

## 项目结构 / Project Structure

```
carla-autonomous-driving-nvidia/
├── src/
│   ├── autonomous_driving/     # 自动驾驶核心算法
│   ├── perception/            # 感知模块
│   ├── planning/              # 路径规划
│   ├── control/               # 车辆控制
│   ├── simulation/            # CARLA仿真接口
│   └── data_collection/       # 数据收集
├── models/                    # 训练好的模型
├── data/                      # 数据集
├── configs/                   # 配置文件
├── tests/                     # 测试文件
├── docs/                      # 文档
└── requirements.txt           # 依赖包
```

## 研究目标 / Research Goals

1. 开发高效的感知算法 / Develop efficient perception algorithms
2. 实现安全的路径规划 / Implement safe path planning
3. 优化车辆控制策略 / Optimize vehicle control strategies
4. 评估算法性能 / Evaluate algorithm performance
5. 发布研究成果 / Publish research findings

## 贡献 / Contributing

欢迎提交问题和贡献代码！/ Issues and contributions are welcome!

## 许可证 / License

MIT License

## 联系方式 / Contact

**研究者 / Researcher**: 张涔熙 (Zhang Cixi)  
**项目 / Project**: CARLA自动驾驶研究 / CARLA Autonomous Driving Research