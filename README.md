# CARLA Autonomous Driving with NVIDIA Models

A comprehensive end-to-end autonomous driving research project using CARLA simulator with state-of-the-art NVIDIA models including Hydra-MDP and OmniDrive.

## ? Overview

This project implements and tests advanced autonomous driving models in the CARLA simulator:

- **Hydra-MDP**: Multi-teacher knowledge distillation architecture (CVPR 2024 winner)
- **OmniDrive**: End-to-end framework with Large Language Models (CVPR 2025)
- **NVIDIA PilotNet**: Classic end-to-end driving model
- **Custom Models**: Conditional Affordance Learning framework

## ?? Project Structure

```
├── models/                    # Neural network architectures
│   ├── hydra_mdp.py          # Hydra-MDP implementation
│   ├── omnidrive.py          # OmniDrive with LLM reasoning
│   └── nvidia_pilotnet_fixed.py  # Fixed PilotNet architecture
├── algorithms/               # Training and data collection
│   ├── collect_pilotnet_data.py  # Manual driving data collection
│   └── train_pilotnet.py     # Model training pipeline
├── tests/                    # Testing and evaluation
│   ├── test_nvidia_models.py # Real-time model testing
│   └── simple_movement_test.py # Vehicle movement validation
├── config/                   # Configuration files
├── checkpoints/             # Model weights and training results
└── docs/                    # Documentation and research papers
```

## ? Requirements

### System Requirements
- Windows 10/11
- NVIDIA GPU with CUDA support
- 8GB+ RAM
- CARLA 0.9.16

### Python Dependencies
```bash
pip install -r requirements.txt
```

### CARLA Setup
1. Download CARLA 0.9.16 from [official website](https://carla.org/)
2. Extract to `F:\CARLA_0.9.16\` (or update paths accordingly)
3. Start CARLA server: `CarlaUE4.exe`

## ? Quick Start

### 1. Environment Setup
```bash
# Create conda environment
conda create -n carla_env python=3.8
conda activate carla_env

# Install dependencies
pip install -r requirements.txt
```

### 2. Test Model Architectures
```bash
# Test Hydra-MDP
python models/hydra_mdp.py

# Test OmniDrive
python models/omnidrive.py

# Test PilotNet
python models/nvidia_pilotnet_fixed.py
```

### 3. Real-time Testing in CARLA
```bash
# Start CARLA server first
cd F:\CARLA_0.9.16
CarlaUE4.exe

# Test Hydra-MDP model
python test_nvidia_models.py --model hydra --duration 60

# Test OmniDrive model
python test_nvidia_models.py --model omnidrive --duration 60
```

### 4. Data Collection and Training
```bash
# Collect manual driving data
python collect_pilotnet_data.py

# Train PilotNet model
python train_pilotnet.py
```

## ? Model Performance

| Model | Parameters | FPS | Key Features |
|-------|-----------|-----|--------------|
| Hydra-MDP | 3.5M | 60+ | Multi-teacher distillation, safety scoring |
| OmniDrive | 25.5M | 45+ | LLM reasoning, BEV perception |
| PilotNet | 3.0M | 70+ | Classic end-to-end, proven architecture |

## ? Testing Results

### Hydra-MDP Performance
- ? Real-time inference at 60+ FPS
- ? Multi-task outputs (steering, throttle, brake, waypoints)
- ? Safety-aware predictions
- ?? Requires training data for proper vehicle control

### OmniDrive Performance  
- ? Advanced 3D perception and reasoning
- ? Counterfactual analysis capabilities
- ? Text explanation generation
- ?? Higher computational requirements

## ? Research Features

### Advanced Capabilities
- **Multi-teacher Knowledge Distillation**: Learn from both human and rule-based teachers
- **Safety-aware Driving**: Explicit collision risk assessment
- **3D Spatial Reasoning**: Bird's eye view perception
- **Explainable AI**: Natural language decision justification
- **Real-time Performance**: Optimized for live CARLA testing

### Data Collection
- Manual driving with WASD controls
- Automatic labeling with vehicle state
- Multi-modal sensor data (RGB, depth, semantic)
- Scenario-based data organization

## ? Key Files

### Core Models
- `models/hydra_mdp.py` - CVPR 2024 winning architecture
- `models/omnidrive.py` - LLM-enhanced autonomous driving
- `models/nvidia_pilotnet_fixed.py` - Fixed dimension PilotNet

### Testing Suite
- `test_nvidia_models.py` - Comprehensive real-time testing
- `simple_movement_test.py` - Vehicle movement validation
- Manual controls with collision detection

### Training Pipeline
- `collect_pilotnet_data.py` - Interactive data collection
- `train_pilotnet.py` - Complete training with validation
- Automatic checkpoint saving and resume

## ? Known Issues

1. **Vehicle Movement**: Untrained models may produce conservative driving
2. **Data Requirements**: Models need substantial training data for good performance
3. **GPU Memory**: OmniDrive requires significant GPU memory (8GB+ recommended)

## ? Troubleshooting

### CARLA Connection Issues
```bash
# Check if CARLA is running
netstat -an | findstr "2000"

# Restart CARLA server
cd F:\CARLA_0.9.16
CarlaUE4.exe
```

### Model Loading Issues
```bash
# Test CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Test model architectures
python models/hydra_mdp.py
```

## ? Research Papers

- **Hydra-MDP**: "Multi-teacher Knowledge Distillation for End-to-end Autonomous Driving" (CVPR 2024)
- **OmniDrive**: "OmniDrive: A Holistic LLM-Agent Framework for Autonomous Driving" (CVPR 2025)
- **PilotNet**: "End to End Learning for Self-Driving Cars" (NVIDIA 2016)

## ? Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open Pull Request

## ? License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ? Acknowledgments

- NVIDIA for open-source model architectures
- CARLA team for the simulation environment
- Research community for advancing autonomous driving

## ? Contact

For questions or collaboration opportunities, please open an issue or contact the maintainers.

---

**? Star this repository if you find it helpful for your autonomous driving research!**
<<<<<<< HEAD
# carla-autonomous-driving-nvidia
=======
# CARLA Autonomous Driving with NVIDIA Models# Accident-Aware End-to-End Learning for Autonomous Driving



A comprehensive end-to-end autonomous driving research project using CARLA simulator with state-of-the-art NVIDIA models including Hydra-MDP and OmniDrive.## ? 项目概述

本项目研究基于事故学习的自动驾驶安全系统，扩展NVIDIA端到端学习架构，通过学习事故场景提升驾驶安全性。

## ? Overview

## ? 项目结构

This project implements and tests advanced autonomous driving models in the CARLA simulator:

### **核心算法** (`algorithms/`)

- **Hydra-MDP**: Multi-teacher knowledge distillation architecture (CVPR 2024 winner)- `accident_aware_nvidia_clean.py` - **主要算法**：事故感知的端到端学习系统

- **OmniDrive**: End-to-end framework with Large Language Models (CVPR 2025)- `nvidia_end_to_end.py` - 基础NVIDIA架构实现

- **NVIDIA PilotNet**: Classic end-to-end driving model- `collect_research_data.py` - CARLA数据收集脚本

- **Custom Models**: Conditional Affordance Learning framework- `research_pipeline.py` - 完整研究流程



## ?? Project Structure### **文献研究** (`papers/`)

- `accident_aware_paper_draft.md` - 学术论文草稿

```- `conditional_affordance_reading_notes.md` - 核心论文阅读笔记

├── models/                    # Neural network architectures- `literature_comparison_table.md` - 竞争对手分析

│   ├── hydra_mdp.py          # Hydra-MDP implementation- `nvidia_end_to_end_analysis.md` - NVIDIA论文分析

│   ├── omnidrive.py          # OmniDrive with LLM reasoning- `chauffeurnet_analysis.md` - Waymo系统分析

│   └── nvidia_pilotnet_fixed.py  # Fixed PilotNet architecture- `accident_learning_research_analysis.md` - 研究领域分析

├── algorithms/               # Training and data collection

│   ├── collect_pilotnet_data.py  # Manual driving data collection### **配置与数据**

│   └── train_pilotnet.py     # Model training pipeline- `config/` - 系统配置文件

├── tests/                    # Testing and evaluation- `data/` - 仿真数据存储 (深度图像等)

│   ├── test_nvidia_models.py # Real-time model testing- `models/` - 训练好的模型文件

│   └── simple_movement_test.py # Vehicle movement validation- `tests/` - 功能测试脚本

├── config/                   # Configuration files

├── checkpoints/             # Model weights and training results### **研究规划**

└── docs/                    # Documentation and research papers- `RESEARCH_PLAN.md` - **4个月完整研究计划**

```- `week1-2_literature_plan.md` - 文献调研详细计划

- `requirements.txt` - Python依赖包

## ? Requirements

## ? 核心创新

### System Requirements1. **首个事故感知学习系统** - 从事故视频中学习安全驾驶

- Windows 10/112. **双阶段训练方法** - 正常驾驶 + 事故场景学习

- NVIDIA GPU with CUDA support3. **动态安全评估** - 实时预测事故风险

- 8GB+ RAM4. **扩展affordance学习** - 结合条件感知与安全学习

- CARLA 0.9.16

## ? 环境要求

### Python Dependencies- Python 3.8+ 

```bash- PyTorch (GPU支持)

pip install -r requirements.txt- CARLA 0.9.16

```- OpenCV, NumPy

- Conda环境: `carla_env`

### CARLA Setup

1. Download CARLA 0.9.16 from [official website](https://carla.org/)## ? 快速开始

2. Extract to `F:\CARLA_0.9.16\` (or update paths accordingly)

3. Start CARLA server: `CarlaUE4.exe`### 1. 激活环境

```bash

## ? Quick Startconda activate carla_env

```

### 1. Environment Setup

```bash### 2. 验证安装

# Create conda environment```bash

conda create -n carla_env python=3.8python test_environment.py

conda activate carla_envpython test_dl_environment.py

```

# Install dependencies

pip install -r requirements.txt### 3. 运行核心算法

``````bash

python algorithms/accident_aware_nvidia_clean.py

### 2. Test Model Architectures```

```bash

# Test Hydra-MDP## ? 当前进展

python models/hydra_mdp.py- ? 核心算法实现完成

- ? 代码验证通过

# Test OmniDrive- ? 文献调研启动

python models/omnidrive.py- ? 数据集构建中

- ? 论文写作进行中

# Test PilotNet

python models/nvidia_pilotnet_fixed.py## ? 目标会议

```- IROS/ICRA 2025 (主要目标)

- CVPR 2025 (备选)

### 3. Real-time Testing in CARLA- CoRL 2025 (机器人学习专场)

```bash

# Start CARLA server first## ? 研究进展跟踪

cd F:\CARLA_0.9.16查看 `RESEARCH_PLAN.md` 了解详细的4个月研究时间线和里程碑规划。
CarlaUE4.exe

# Test Hydra-MDP model
python test_nvidia_models.py --model hydra --duration 60

# Test OmniDrive model
python test_nvidia_models.py --model omnidrive --duration 60
```

### 4. Data Collection and Training
```bash
# Collect manual driving data
python collect_pilotnet_data.py

# Train PilotNet model
python train_pilotnet.py
```

## ? Model Performance

| Model | Parameters | FPS | Key Features |
|-------|-----------|-----|--------------|
| Hydra-MDP | 3.5M | 60+ | Multi-teacher distillation, safety scoring |
| OmniDrive | 25.5M | 45+ | LLM reasoning, BEV perception |
| PilotNet | 3.0M | 70+ | Classic end-to-end, proven architecture |

## ? Testing Results

### Hydra-MDP Performance
- ? Real-time inference at 60+ FPS
- ? Multi-task outputs (steering, throttle, brake, waypoints)
- ? Safety-aware predictions
- ?? Requires training data for proper vehicle control

### OmniDrive Performance  
- ? Advanced 3D perception and reasoning
- ? Counterfactual analysis capabilities
- ? Text explanation generation
- ?? Higher computational requirements

## ? Research Features

### Advanced Capabilities
- **Multi-teacher Knowledge Distillation**: Learn from both human and rule-based teachers
- **Safety-aware Driving**: Explicit collision risk assessment
- **3D Spatial Reasoning**: Bird's eye view perception
- **Explainable AI**: Natural language decision justification
- **Real-time Performance**: Optimized for live CARLA testing

### Data Collection
- Manual driving with WASD controls
- Automatic labeling with vehicle state
- Multi-modal sensor data (RGB, depth, semantic)
- Scenario-based data organization

## ? Key Files

### Core Models
- `models/hydra_mdp.py` - CVPR 2024 winning architecture
- `models/omnidrive.py` - LLM-enhanced autonomous driving
- `models/nvidia_pilotnet_fixed.py` - Fixed dimension PilotNet

### Testing Suite
- `test_nvidia_models.py` - Comprehensive real-time testing
- `simple_movement_test.py` - Vehicle movement validation
- Manual controls with collision detection

### Training Pipeline
- `collect_pilotnet_data.py` - Interactive data collection
- `train_pilotnet.py` - Complete training with validation
- Automatic checkpoint saving and resume

## ? Known Issues

1. **Vehicle Movement**: Untrained models may produce conservative driving
2. **Data Requirements**: Models need substantial training data for good performance
3. **GPU Memory**: OmniDrive requires significant GPU memory (8GB+ recommended)

## ? Troubleshooting

### CARLA Connection Issues
```bash
# Check if CARLA is running
netstat -an | findstr "2000"

# Restart CARLA server
cd F:\CARLA_0.9.16
CarlaUE4.exe
```

### Model Loading Issues
```bash
# Test CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Test model architectures
python models/hydra_mdp.py
```

## ? Research Papers

- **Hydra-MDP**: "Multi-teacher Knowledge Distillation for End-to-end Autonomous Driving" (CVPR 2024)
- **OmniDrive**: "OmniDrive: A Holistic LLM-Agent Framework for Autonomous Driving" (CVPR 2025)
- **PilotNet**: "End to End Learning for Self-Driving Cars" (NVIDIA 2016)

## ? Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open Pull Request

## ? License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ? Acknowledgments

- NVIDIA for open-source model architectures
- CARLA team for the simulation environment
- Research community for advancing autonomous driving

## ? Contact

For questions or collaboration opportunities, please open an issue or contact the maintainers.

---

**? Star this repository if you find it helpful for your autonomous driving research!**
>>>>>>> ca81866 (Initial commit: CARLA autonomous driving with NVIDIA models)
