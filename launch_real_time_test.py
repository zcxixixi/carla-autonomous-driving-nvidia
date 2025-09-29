"""Launch Real-time CARLA Control Test"""
import os
import sys
import subprocess
import time
import psutil

def check_carla_server():
    """Check if CARLA server is running"""
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            if 'carla' in proc.info['name'].lower():
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return False

def start_carla_server():
    """Start CARLA server"""
    carla_path = "F:\\CARLA_0.9.16\\CarlaUE4.exe"
    
    if not os.path.exists(carla_path):
        print("? CARLA executable not found!")
        print(f"Expected: {carla_path}")
        return False
    
    print("? Starting CARLA server...")
    try:
        # Start CARLA in background
        subprocess.Popen([carla_path, "-windowed", "-ResX=800", "-ResY=600"])
        print("? Waiting for CARLA server to initialize...")
        
        # Wait for server to start
        for i in range(30):  # Wait up to 30 seconds
            if check_carla_server():
                print("? CARLA server is running")
                time.sleep(5)  # Additional wait for full initialization
                return True
            time.sleep(1)
            print(f"   Waiting... {i+1}/30")
        
        print("? CARLA server failed to start")
        return False
        
    except Exception as e:
        print(f"? Failed to start CARLA: {e}")
        return False

def run_real_time_test():
    """Run the real-time control test"""
    print("\\n? Running Real-time Control Test")
    print("=" * 40)
    
    # Activate conda environment and run test
    cmd = [
        "conda", "run", "-n", "carla_env", "python", 
        "algorithms/real_time_control.py"
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"? Test failed: {e}")
        return False
    except KeyboardInterrupt:
        print("\\n?? Test interrupted by user")
    
    return True

def main():
    """Main launcher"""
    print("? CARLA Real-time Control Launcher")
    print("=" * 50)
    
    # Check if CARLA is already running
    if check_carla_server():
        print("? CARLA server already running")
    else:
        print("?? CARLA server not found")
        user_input = input("Start CARLA server? (y/n): ").lower().strip()
        
        if user_input == 'y':
            if not start_carla_server():
                return
        else:
            print("? CARLA server required for testing")
            return
    
    # Run the test
    print("\\n? Starting real-time control test...")
    print("Controls during test:")
    print("  - Model will automatically control the vehicle")
    print("  - Press Ctrl+C to stop")
    print("  - Check terminal for real-time statistics")
    
    input("\\nPress Enter to start test...")
    
    run_real_time_test()
    
    print("\\n? Test completed!")

if __name__ == '__main__':
    main()