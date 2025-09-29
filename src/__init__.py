#!/usr/bin/env python3
"""
张涔熙的科研项目 - CARLA自动驾驶项目初始化文件
CARLA Autonomous Driving Project Initialization

This package implements autonomous driving algorithms using CARLA simulator
with NVIDIA GPU acceleration for research and development purposes.
"""

__version__ = "1.0.0"
__author__ = "张涔熙 (Zhang Cixi)"
__email__ = "research@example.com"
__description__ = "CARLA Autonomous Driving with NVIDIA - Research Project"

# Import main modules
from . import autonomous_driving
from . import perception
from . import planning
from . import control
from . import simulation
from . import data_collection

__all__ = [
    "autonomous_driving",
    "perception", 
    "planning",
    "control",
    "simulation",
    "data_collection"
]