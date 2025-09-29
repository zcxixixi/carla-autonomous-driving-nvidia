#!/usr/bin/env python3
"""
CARLA自动驾驶项目安装脚本 / CARLA Autonomous Driving Project Setup Script
张涔熙的科研项目 - 环境配置和依赖安装

This script sets up the development environment for the autonomous driving project.
"""

import os
import subprocess
import sys
from pathlib import Path

def check_python_version():
    """检查Python版本 / Check Python version"""
    if sys.version_info < (3, 7):
        print("❌ Python 3.7+ is required")
        sys.exit(1)
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")

def check_cuda():
    """检查CUDA是否可用 / Check if CUDA is available"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA version: {torch.version.cuda}")
            return True
        else:
            print("⚠️  CUDA not available, using CPU")
            return False
    except ImportError:
        print("⚠️  PyTorch not installed yet")
        return False

def install_dependencies():
    """安装项目依赖 / Install project dependencies"""
    print("\n📦 Installing dependencies...")
    
    # Install requirements
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    print("✅ Dependencies installed successfully")

def create_directories():
    """创建必要的目录 / Create necessary directories"""
    print("\n📁 Creating project directories...")
    
    directories = [
        "data/logs",
        "data/models", 
        "data/datasets",
        "models",
        "logs",
        "tests/test_logs",
        "docs/images",
        "configs/custom"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   Created: {directory}")
    
    print("✅ Project directories created")

def download_models():
    """下载预训练模型 / Download pre-trained models"""
    print("\n🔽 Setting up model directory...")
    
    # Create placeholder model files (in real project, these would be actual model downloads)
    model_dir = Path("models")
    
    placeholder_models = [
        "camera_perception.pth",
        "lidar_perception.pth", 
        "path_planning.pth"
    ]
    
    for model_name in placeholder_models:
        model_path = model_dir / model_name
        if not model_path.exists():
            # Create empty placeholder file
            model_path.touch()
            print(f"   Created placeholder: {model_name}")
    
    print("✅ Model directory setup complete")

def setup_git_hooks():
    """设置Git钩子 / Setup Git hooks"""
    print("\n🔧 Setting up Git hooks...")
    
    # Create pre-commit hook for code formatting
    hooks_dir = Path(".git/hooks")
    if hooks_dir.exists():
        pre_commit_hook = hooks_dir / "pre-commit"
        hook_content = """#!/bin/sh
# Auto-format code before commit
echo "Running code formatter..."
python -m black src/ tests/ --line-length 100
python -m isort src/ tests/
echo "Code formatting complete"
"""
        pre_commit_hook.write_text(hook_content)
        pre_commit_hook.chmod(0o755)
        print("   ✅ Pre-commit hook created")
    
    print("✅ Git hooks setup complete")

def verify_installation():
    """验证安装 / Verify installation"""
    print("\n🔍 Verifying installation...")
    
    try:
        # Test imports
        import torch
        import numpy as np
        import cv2
        import pandas as pd
        print("   ✅ Core dependencies imported successfully")
        
        # Test CARLA import (might fail if CARLA not installed)
        try:
            import carla
            print("   ✅ CARLA Python API imported successfully")
        except ImportError:
            print("   ⚠️  CARLA Python API not found (install CARLA separately)")
        
        # Test custom modules
        sys.path.append(str(Path.cwd() / "src"))
        
        from perception.camera_perception import CameraPerception
        from planning.path_planner import PathPlanner
        from control.vehicle_controller import VehicleController
        print("   ✅ Custom modules imported successfully")
        
        print("✅ Installation verification complete")
        return True
        
    except Exception as e:
        print(f"   ❌ Verification failed: {e}")
        return False

def print_next_steps():
    """打印后续步骤 / Print next steps"""
    print("\n" + "="*60)
    print("🎉 Setup Complete! / 安装完成！")
    print("="*60)
    print("\n📋 Next Steps / 后续步骤:")
    print("\n1. Install CARLA Simulator:")
    print("   Download from: https://carla.readthedocs.io/en/latest/start_quickstart/")
    print("   Add CARLA Python API to your Python path")
    
    print("\n2. Start CARLA Server:")
    print("   ./CarlaUE4.sh")
    print("   # or on Windows: CarlaUE4.exe")
    
    print("\n3. Run Tests:")
    print("   python tests/test_system.py")
    
    print("\n4. Start Autonomous Driving:")
    print("   python src/autonomous_driving/main.py")
    
    print("\n5. Custom Configuration:")
    print("   Edit configs/default.yaml for your specific setup")
    
    print("\n📖 Documentation:")
    print("   See README.md for detailed usage instructions")
    
    print("\n🐛 Troubleshooting:")
    print("   Check logs/ directory for error messages")
    print("   Ensure NVIDIA drivers are installed for GPU acceleration")

def main():
    """主安装函数 / Main setup function"""
    print("🚗 CARLA自动驾驶项目安装程序")
    print("CARLA Autonomous Driving Project Setup")
    print("张涔熙的科研项目 / Zhang Cixi's Research Project")
    print("="*60)
    
    # Change to project directory
    os.chdir(Path(__file__).parent)
    
    # Run setup steps
    check_python_version()
    create_directories()
    install_dependencies()
    check_cuda()
    download_models()
    setup_git_hooks()
    
    # Verify installation
    if verify_installation():
        print_next_steps()
    else:
        print("\n❌ Setup completed with errors. Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()