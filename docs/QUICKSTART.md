# 快速开始指南 / Quick Start Guide

张涔熙的科研项目 - CARLA自动驾驶系统快速开始指南

## 环境准备 / Environment Setup

### 1. 系统要求 / System Requirements
- Python 3.7+
- NVIDIA GPU with CUDA support (推荐)
- CARLA Simulator 0.9.10+
- 16GB+ RAM (推荐)

### 2. 安装步骤 / Installation Steps

```bash
# 1. 克隆项目 / Clone repository
git clone https://github.com/zcxixixi/carla-autonomous-driving-nvidia.git
cd carla-autonomous-driving-nvidia

# 2. 运行安装脚本 / Run setup script
python setup.py

# 3. 安装CARLA (单独下载) / Install CARLA (separate download)
# Download from: https://carla.readthedocs.io/en/latest/start_quickstart/
```

## 使用方法 / Usage

### 1. 启动CARLA服务器 / Start CARLA Server
```bash
# Linux/Mac
./CarlaUE4.sh

# Windows
CarlaUE4.exe
```

### 2. 运行自动驾驶系统 / Run Autonomous Driving
```bash
# 基本运行 / Basic run
python src/autonomous_driving/main.py

# 自定义配置 / Custom configuration
python src/autonomous_driving/main.py --config configs/custom.yaml --duration 120
```

### 3. 数据收集 / Data Collection
```bash
# 收集训练数据 / Collect training data
python src/data_collection/collect_data.py --scenario free_roam --duration 300

# 不同场景 / Different scenarios
python src/data_collection/collect_data.py --scenario stop_go --duration 180
```

### 4. 系统测试 / System Testing
```bash
# 运行测试 / Run tests
python tests/test_system.py

# 性能测试 / Performance test
python tests/test_system.py --benchmark
```

## 配置说明 / Configuration

编辑 `configs/default.yaml` 文件来自定义系统参数：

```yaml
# 车辆设置 / Vehicle settings
vehicle:
  type: "model3"
  
# 感知设置 / Perception settings  
perception:
  camera:
    confidence_threshold: 0.5
  lidar:
    max_range: 100.0

# 控制设置 / Control settings
control:
  pid:
    speed:
      kp: 0.8
```

## 常见问题 / Troubleshooting

### Q: CARLA连接失败
A: 确保CARLA服务器正在运行，端口2000未被占用

### Q: GPU加速不工作  
A: 检查NVIDIA驱动和CUDA安装，运行 `nvidia-smi` 验证

### Q: 依赖安装失败
A: 使用 `pip install --upgrade pip` 更新pip，然后重新安装

## 项目结构 / Project Structure

```
carla-autonomous-driving-nvidia/
├── src/                    # 源代码 / Source code
│   ├── autonomous_driving/ # 主系统 / Main system
│   ├── perception/         # 感知模块 / Perception
│   ├── planning/           # 规划模块 / Planning  
│   ├── control/           # 控制模块 / Control
│   ├── simulation/        # 仿真接口 / Simulation
│   └── data_collection/   # 数据收集 / Data collection
├── configs/               # 配置文件 / Configuration
├── data/                  # 数据目录 / Data directory
├── models/                # 模型文件 / Model files
└── tests/                 # 测试文件 / Test files
```

## 开发指南 / Development Guide

### 添加新模块 / Adding New Modules
1. 在相应目录创建Python文件
2. 继承基础类（如果有）
3. 实现必要的接口方法
4. 添加单元测试

### 代码规范 / Code Standards
- 使用Python类型提示
- 遵循PEP 8代码风格
- 添加中英文注释
- 编写单元测试

## 联系方式 / Contact

**研究者**: 张涔熙 (Zhang Cixi)  
**项目**: CARLA自动驾驶研究  
**邮箱**: research@example.com