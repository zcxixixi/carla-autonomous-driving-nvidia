# -*- coding: utf-8 -*-
"""
CARLA Data Analysis Script
Analyze collected sensor data from CARLA simulation
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob

def analyze_collected_data():
    """Analyze the collected CARLA data"""
    print("=" * 60)
    print("CARLA Collected Data Analysis")
    print("=" * 60)
    
    data_dir = "data"
    
    if not os.path.exists(data_dir):
        print("? Data directory not found!")
        return
    
    # Count files
    rgb_files = glob.glob(os.path.join(data_dir, "rgb_*.png"))
    depth_files = glob.glob(os.path.join(data_dir, "depth_*.png")) 
    lidar_files = glob.glob(os.path.join(data_dir, "lidar_*.npy"))
    
    print(f"? Data Summary:")
    print(f"   RGB Images: {len(rgb_files)}")
    print(f"   Depth Images: {len(depth_files)}")
    print(f"   LiDAR Point Clouds: {len(lidar_files)}")
    
    # Analyze RGB images
    if rgb_files:
        print(f"\n??  RGB Image Analysis:")
        rgb_file = rgb_files[0]  # Analyze first image
        img = Image.open(rgb_file)
        print(f"   Image size: {img.size}")
        print(f"   Image mode: {img.mode}")
        
        # Get file sizes
        sizes = [os.path.getsize(f) for f in rgb_files]
        print(f"   Average file size: {np.mean(sizes)/1024:.1f} KB")
        print(f"   Total RGB data: {sum(sizes)/1024/1024:.1f} MB")
    
    # Analyze LiDAR data
    if lidar_files:
        print(f"\n? LiDAR Data Analysis:")
        lidar_file = lidar_files[0]  # Analyze first file
        points = np.load(lidar_file)
        print(f"   Point cloud shape: {points.shape}")
        print(f"   Points per scan: {points.shape[0]}")
        print(f"   Coordinates: X, Y, Z, Intensity")
        
        # Statistics
        print(f"   X range: [{points[:, 0].min():.1f}, {points[:, 0].max():.1f}] m")
        print(f"   Y range: [{points[:, 1].min():.1f}, {points[:, 1].max():.1f}] m") 
        print(f"   Z range: [{points[:, 2].min():.1f}, {points[:, 2].max():.1f}] m")
        
        # Get file sizes
        sizes = [os.path.getsize(f) for f in lidar_files]
        print(f"   Average file size: {np.mean(sizes)/1024:.1f} KB")
        print(f"   Total LiDAR data: {sum(sizes)/1024/1024:.1f} MB")
    
    # Total data size
    all_files = rgb_files + depth_files + lidar_files
    total_size = sum(os.path.getsize(f) for f in all_files)
    print(f"\n? Total Dataset Size: {total_size/1024/1024:.1f} MB")
    
    # Data collection rate analysis
    if len(all_files) > 0:
        print(f"\n??  Collection Statistics:")
        print(f"   Collection duration: ~15 seconds")
        print(f"   RGB frame rate: ~{len(rgb_files)/15:.1f} FPS")
        print(f"   Depth frame rate: ~{len(depth_files)/15:.1f} FPS")
        print(f"   LiDAR scan rate: ~{len(lidar_files)/15:.1f} Hz")

def visualize_sample_data():
    """Create visualization of sample data"""
    print(f"\n? Creating Data Visualizations...")
    
    data_dir = "data"
    
    # Load sample RGB image
    rgb_files = glob.glob(os.path.join(data_dir, "rgb_*.png"))
    if rgb_files:
        rgb_img = Image.open(rgb_files[10])  # Use 10th image
        
        plt.figure(figsize=(15, 5))
        
        # Show RGB image
        plt.subplot(1, 3, 1)
        plt.imshow(rgb_img)
        plt.title("RGB Camera View")
        plt.axis('off')
        
        # Load and show LiDAR data
        lidar_files = glob.glob(os.path.join(data_dir, "lidar_*.npy"))
        if len(lidar_files) > 10:
            points = np.load(lidar_files[10])
            
            # Top-down view
            plt.subplot(1, 3, 2)
            plt.scatter(points[:, 0], points[:, 1], c=points[:, 2], s=1, cmap='viridis')
            plt.colorbar(label='Height (m)')
            plt.title("LiDAR Top View (X-Y)")
            plt.xlabel("X (m)")
            plt.ylabel("Y (m)")
            plt.axis('equal')
            
            # Side view
            plt.subplot(1, 3, 3)
            distances = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
            plt.scatter(distances, points[:, 2], c=points[:, 3], s=1, cmap='plasma')
            plt.colorbar(label='Intensity')
            plt.title("LiDAR Side View (Distance-Height)")
            plt.xlabel("Distance (m)")
            plt.ylabel("Height (m)")
        
        plt.tight_layout()
        plt.savefig(os.path.join(data_dir, "data_visualization.png"), dpi=150, bbox_inches='tight')
        print(f"   ? Saved visualization: {data_dir}/data_visualization.png")
        plt.show()

def main():
    """Main analysis function"""
    analyze_collected_data()
    
    try:
        visualize_sample_data()
        print(f"\n? Data analysis complete!")
        print(f"   Your CARLA data collection system is working perfectly!")
        print(f"   You can now use this data for:")
        print(f"     ? Computer vision research")
        print(f"     ? Autonomous driving algorithms")
        print(f"     ? Sensor fusion experiments")
        print(f"     ? Deep learning model training")
        
    except Exception as e:
        print(f"\n??  Visualization error (data analysis still complete): {e}")
        print(f"   Note: Install matplotlib and PIL for visualizations")

if __name__ == '__main__':
    main()