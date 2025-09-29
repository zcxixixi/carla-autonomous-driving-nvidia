#!/usr/bin/env python3
"""
LiDAR感知模块 / LiDAR Perception Module
张涔熙的科研项目 - 基于点云的感知系统

This module implements LiDAR-based perception for autonomous driving,
including object detection, ground plane estimation, and 3D mapping.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class LidarPerception:
    """
    LiDAR感知类 / LiDAR Perception Class
    
    实现基于LiDAR的感知功能，包括3D目标检测、地面分割、点云处理等
    Implements LiDAR-based perception including 3D object detection, ground segmentation, point cloud processing
    """
    
    def __init__(self, device: torch.device = None):
        """
        初始化LiDAR感知模块 / Initialize LiDAR perception module
        
        Args:
            device: 计算设备 / Computing device
        """
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # LiDAR parameters
        self.max_range = 100.0  # Maximum detection range in meters
        self.min_range = 1.0    # Minimum detection range in meters
        self.grid_size = 0.2    # Voxel grid size in meters
        
        # Initialize models
        self.object_detector_3d = None
        self.ground_segmentor = None
        
        self._load_models()
        
        logger.info(f"LiDAR perception initialized on device: {self.device}")
    
    def _load_models(self):
        """加载3D感知模型 / Load 3D perception models"""
        try:
            # Load 3D object detection model
            self.object_detector_3d = PointNetObjectDetector()
            
            # Load ground segmentation model
            self.ground_segmentor = GroundSegmentor()
            
            # Move models to device
            self.object_detector_3d.to(self.device)
            self.ground_segmentor.to(self.device)
            
            # Set to evaluation mode
            self.object_detector_3d.eval()
            self.ground_segmentor.eval()
            
            logger.info("LiDAR perception models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load LiDAR perception models: {e}")
    
    def process(self, lidar_data: Dict) -> Dict:
        """
        处理LiDAR数据 / Process LiDAR data
        
        Args:
            lidar_data: LiDAR数据字典 / LiDAR data dictionary
            
        Returns:
            感知结果字典 / Perception result dictionary
        """
        if not lidar_data or 'data' not in lidar_data:
            return {}
        
        points = lidar_data['data']
        timestamp = lidar_data.get('timestamp', 0)
        
        try:
            # Preprocess point cloud
            processed_points = self._preprocess_pointcloud(points)
            
            # Ground segmentation
            ground_mask, object_points = self._segment_ground(processed_points)
            
            # 3D object detection
            objects_3d = self._detect_objects_3d(object_points)
            
            # Generate depth map
            depth_map = self._generate_depth_map(processed_points)
            
            # Obstacle detection
            obstacles = self._detect_obstacles(object_points)
            
            result = {
                'timestamp': timestamp,
                'objects': objects_3d,
                'obstacles': obstacles,
                'ground_mask': ground_mask,
                'depth_map': depth_map,
                'point_count': len(processed_points),
                'range_data': self._calculate_range_data(processed_points)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing LiDAR data: {e}")
            return {}
    
    def _preprocess_pointcloud(self, points: np.ndarray) -> np.ndarray:
        """
        点云预处理 / Point cloud preprocessing
        
        Args:
            points: 原始点云数据 / Raw point cloud data
            
        Returns:
            预处理后的点云 / Preprocessed point cloud
        """
        # Remove points outside range
        distances = np.sqrt(points[:, 0]**2 + points[:, 1]**2 + points[:, 2]**2)
        valid_mask = (distances >= self.min_range) & (distances <= self.max_range)
        filtered_points = points[valid_mask]
        
        # Remove points below ground (assuming z < -2.5m is underground)
        height_mask = filtered_points[:, 2] > -2.5
        filtered_points = filtered_points[height_mask]
        
        # Downsample using voxel grid
        downsampled_points = self._voxel_downsample(filtered_points)
        
        return downsampled_points
    
    def _voxel_downsample(self, points: np.ndarray) -> np.ndarray:
        """
        体素网格降采样 / Voxel grid downsampling
        
        Args:
            points: 输入点云 / Input point cloud
            
        Returns:
            降采样后的点云 / Downsampled point cloud
        """
        # Calculate voxel indices
        voxel_indices = np.floor(points[:, :3] / self.grid_size).astype(int)
        
        # Find unique voxels
        unique_indices, inverse_indices = np.unique(voxel_indices, axis=0, return_inverse=True)
        
        # Average points in each voxel
        downsampled_points = []
        for i in range(len(unique_indices)):
            voxel_mask = inverse_indices == i
            voxel_points = points[voxel_mask]
            centroid = np.mean(voxel_points, axis=0)
            downsampled_points.append(centroid)
        
        return np.array(downsampled_points)
    
    def _segment_ground(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        地面分割 / Ground segmentation
        
        Args:
            points: 输入点云 / Input point cloud
            
        Returns:
            地面掩码和物体点云 / Ground mask and object points
        """
        # RANSAC plane fitting for ground detection
        ground_mask = self._ransac_ground_segmentation(points)
        
        # Separate ground and object points
        object_mask = ~ground_mask
        object_points = points[object_mask]
        
        return ground_mask, object_points
    
    def _ransac_ground_segmentation(self, points: np.ndarray, 
                                   threshold: float = 0.2, 
                                   max_iterations: int = 1000) -> np.ndarray:
        """
        RANSAC地面分割 / RANSAC ground segmentation
        
        Args:
            points: 输入点云 / Input point cloud
            threshold: 距离阈值 / Distance threshold
            max_iterations: 最大迭代次数 / Maximum iterations
            
        Returns:
            地面点掩码 / Ground point mask
        """
        best_inliers = 0
        best_mask = np.zeros(len(points), dtype=bool)
        
        for _ in range(max_iterations):
            # Randomly sample 3 points
            sample_indices = np.random.choice(len(points), 3, replace=False)
            sample_points = points[sample_indices]
            
            # Calculate plane parameters
            v1 = sample_points[1] - sample_points[0]
            v2 = sample_points[2] - sample_points[0]
            normal = np.cross(v1, v2)
            
            if np.linalg.norm(normal) == 0:
                continue
                
            normal = normal / np.linalg.norm(normal)
            d = -np.dot(normal, sample_points[0])
            
            # Calculate distances to plane
            distances = np.abs(np.dot(points, normal) + d)
            
            # Find inliers
            inlier_mask = distances < threshold
            num_inliers = np.sum(inlier_mask)
            
            # Update best model
            if num_inliers > best_inliers:
                best_inliers = num_inliers
                best_mask = inlier_mask
        
        return best_mask
    
    def _detect_objects_3d(self, points: np.ndarray) -> List[Dict]:
        """
        3D目标检测 / 3D object detection
        
        Args:
            points: 物体点云 / Object point cloud
            
        Returns:
            3D目标检测结果 / 3D object detection results
        """
        if len(points) == 0:
            return []
        
        # Clustering-based object detection
        clusters = self._euclidean_clustering(points)
        
        objects_3d = []
        for i, cluster in enumerate(clusters):
            if len(cluster) < 10:  # Minimum points per object
                continue
            
            # Calculate bounding box
            min_coords = np.min(cluster, axis=0)
            max_coords = np.max(cluster, axis=0)
            
            # Calculate object properties
            center = (min_coords + max_coords) / 2
            size = max_coords - min_coords
            
            # Estimate object type based on size
            object_type = self._classify_object_by_size(size)
            
            objects_3d.append({
                'id': i,
                'type': object_type,
                'center': center.tolist(),
                'size': size.tolist(),
                'bbox_min': min_coords.tolist(),
                'bbox_max': max_coords.tolist(),
                'point_count': len(cluster),
                'confidence': min(1.0, len(cluster) / 100.0)
            })
        
        return objects_3d
    
    def _euclidean_clustering(self, points: np.ndarray, 
                             cluster_tolerance: float = 0.5,
                             min_cluster_size: int = 10,
                             max_cluster_size: int = 10000) -> List[np.ndarray]:
        """
        欧几里得聚类 / Euclidean clustering
        
        Args:
            points: 输入点云 / Input point cloud
            cluster_tolerance: 聚类容差 / Cluster tolerance
            min_cluster_size: 最小聚类大小 / Minimum cluster size
            max_cluster_size: 最大聚类大小 / Maximum cluster size
            
        Returns:
            聚类结果 / Clustering results
        """
        clusters = []
        processed = np.zeros(len(points), dtype=bool)
        
        for i in range(len(points)):
            if processed[i]:
                continue
            
            # Start new cluster
            cluster_indices = []
            search_queue = [i]
            
            while search_queue:
                current_idx = search_queue.pop(0)
                if processed[current_idx]:
                    continue
                
                processed[current_idx] = True
                cluster_indices.append(current_idx)
                
                # Find neighbors
                current_point = points[current_idx]
                distances = np.linalg.norm(points - current_point, axis=1)
                neighbors = np.where((distances < cluster_tolerance) & (~processed))[0]
                
                search_queue.extend(neighbors.tolist())
            
            # Add cluster if size is within limits
            if min_cluster_size <= len(cluster_indices) <= max_cluster_size:
                clusters.append(points[cluster_indices])
        
        return clusters
    
    def _classify_object_by_size(self, size: np.ndarray) -> str:
        """
        根据尺寸分类物体 / Classify object by size
        
        Args:
            size: 物体尺寸 / Object size
            
        Returns:
            物体类型 / Object type
        """
        length, width, height = size
        
        # Simple size-based classification
        if height < 0.5:
            return 'unknown'
        elif length > 4 and width > 1.5:
            return 'car'
        elif length > 8 and width > 2:
            return 'truck'
        elif height > 3:
            return 'building'
        elif height < 2 and max(length, width) < 2:
            return 'person'
        else:
            return 'obstacle'
    
    def _generate_depth_map(self, points: np.ndarray, 
                           image_width: int = 800, 
                           image_height: int = 600) -> np.ndarray:
        """
        生成深度图 / Generate depth map
        
        Args:
            points: 点云数据 / Point cloud data
            image_width: 图像宽度 / Image width
            image_height: 图像高度 / Image height
            
        Returns:
            深度图 / Depth map
        """
        depth_map = np.zeros((image_height, image_width))
        
        if len(points) == 0:
            return depth_map
        
        # Project 3D points to 2D image plane
        # Simplified projection (assuming front-facing camera)
        fov_rad = np.pi / 3  # 60 degrees field of view
        
        for point in points:
            x, y, z = point[:3]
            
            if z > 0:  # Only forward points
                # Calculate image coordinates
                u = int(image_width / 2 + (x / z) * (image_width / 2) / np.tan(fov_rad / 2))
                v = int(image_height / 2 - (y / z) * (image_height / 2) / np.tan(fov_rad / 2))
                
                # Check bounds
                if 0 <= u < image_width and 0 <= v < image_height:
                    depth = np.sqrt(x**2 + y**2 + z**2)
                    if depth_map[v, u] == 0 or depth < depth_map[v, u]:
                        depth_map[v, u] = depth
        
        return depth_map
    
    def _detect_obstacles(self, points: np.ndarray) -> List[Dict]:
        """
        障碍物检测 / Obstacle detection
        
        Args:
            points: 点云数据 / Point cloud data
            
        Returns:
            障碍物列表 / List of obstacles
        """
        obstacles = []
        
        if len(points) == 0:
            return obstacles
        
        # Grid-based obstacle detection
        grid_resolution = 0.5  # 0.5m grid
        x_min, y_min = np.min(points[:, :2], axis=0)
        x_max, y_max = np.max(points[:, :2], axis=0)
        
        x_bins = int((x_max - x_min) / grid_resolution) + 1
        y_bins = int((y_max - y_min) / grid_resolution) + 1
        
        obstacle_grid = np.zeros((y_bins, x_bins))
        
        for point in points:
            x, y, z = point[:3]
            
            # Calculate grid indices
            x_idx = int((x - x_min) / grid_resolution)
            y_idx = int((y - y_min) / grid_resolution)
            
            if 0 <= x_idx < x_bins and 0 <= y_idx < y_bins:
                if z > -1.5:  # Above ground threshold
                    obstacle_grid[y_idx, x_idx] = 1
        
        # Find obstacle regions
        obstacle_indices = np.where(obstacle_grid == 1)
        
        for y_idx, x_idx in zip(obstacle_indices[0], obstacle_indices[1]):
            x_world = x_min + x_idx * grid_resolution
            y_world = y_min + y_idx * grid_resolution
            
            obstacles.append({
                'position': [x_world, y_world],
                'grid_index': [x_idx, y_idx],
                'type': 'obstacle'
            })
        
        return obstacles
    
    def _calculate_range_data(self, points: np.ndarray) -> Dict:
        """
        计算距离数据统计 / Calculate range data statistics
        
        Args:
            points: 点云数据 / Point cloud data
            
        Returns:
            距离统计信息 / Range statistics
        """
        if len(points) == 0:
            return {}
        
        distances = np.sqrt(points[:, 0]**2 + points[:, 1]**2 + points[:, 2]**2)
        
        return {
            'min_range': float(np.min(distances)),
            'max_range': float(np.max(distances)),
            'mean_range': float(np.mean(distances)),
            'std_range': float(np.std(distances)),
            'median_range': float(np.median(distances))
        }


class PointNetObjectDetector(nn.Module):
    """基于PointNet的3D目标检测模型 / PointNet-based 3D object detection model"""
    
    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.num_classes = num_classes
        
        # Point feature extraction
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        
        # Global feature
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # x shape: (batch_size, 3, num_points)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        
        # Global max pooling
        x = torch.max(x, 2)[0]
        
        # Classification head
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        
        return torch.softmax(x, dim=1)


class GroundSegmentor(nn.Module):
    """地面分割模型 / Ground segmentation model"""
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x shape: (batch_size, 3, num_points)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.sigmoid(self.conv3(x))
        return x


def main():
    """测试函数 / Test function"""
    # Create dummy LiDAR data
    num_points = 10000
    dummy_points = np.random.randn(num_points, 4)  # x, y, z, intensity
    dummy_points[:, 2] += 1.0  # Lift points above ground
    
    lidar_data = {
        'data': dummy_points,
        'timestamp': 1234567890
    }
    
    # Initialize perception module
    perception = LidarPerception()
    
    # Process data
    result = perception.process(lidar_data)
    
    logger.info(f"LiDAR perception result: {len(result.get('objects', []))} objects detected")
    logger.info(f"Obstacles detected: {len(result.get('obstacles', []))}")
    logger.info(f"Processed {result.get('point_count', 0)} points")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()