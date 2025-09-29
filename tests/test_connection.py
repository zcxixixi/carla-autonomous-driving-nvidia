# -*- coding: utf-8 -*-
"""
CARLA Connection Test Script
Test basic connection functionality with CARLA server
"""

import sys
import os
import time
from typing import Optional

# Import CARLA (already installed in conda environment)
try:
    import carla
    print("? CARLA module imported successfully")
except ImportError as e:
    print(f"? CARLA module import failed: {e}")
    print("Please ensure:")
    print("1. CARLA server is running")
    print("2. CARLA Python API path is correct")
    print("3. Correct CARLA version is installed")
    sys.exit(1)


def test_carla_connection(host: str = 'localhost', port: int = 2000, timeout: float = 10.0) -> bool:
    """Test CARLA connection"""
    print(f"Connecting to CARLA server {host}:{port}...")
    
    try:
        # 创建客户端
        client = carla.Client(host, port)
        client.set_timeout(timeout)
        
        # 获取世界信息
        world = client.get_world()
        world_map = world.get_map()
        
        print(f"? Connection successful!")
        print(f"  Map name: {world_map.name}")
        print(f"  CARLA version: {client.get_client_version()}")
        print(f"  Server version: {client.get_server_version()}")
        
        # Get some basic info
        blueprint_library = world.get_blueprint_library()
        vehicles = blueprint_library.filter('vehicle.*')
        print(f"  Available vehicles: {len(vehicles)}")
        
        return True
        
    except Exception as e:
        print(f"? Connection failed: {e}")
        print("Please check:")
        print("1. Is CARLA server running? (CarlaUE4.exe)")
        print("2. Is the port correct? (default 2000)")
        print("3. Firewall settings")
        return False


def main():
    """Main function"""
    print("=" * 50)
    print("CARLA Connection Test")
    print("=" * 50)
    
    # Test connection
    success = test_carla_connection()
    
    if success:
        print("\n? All tests passed! CARLA environment is configured correctly.")
        print("You can now start developing CARLA applications!")
    else:
        print("\n? Tests failed. Please check CARLA installation and configuration.")
        return 1
    
    return 0


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)