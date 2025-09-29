#!/usr/bin/env python3
"""
数据收集与记录模块 / Data Collection and Logging Module
张涔熙的科研项目 - 自动驾驶数据记录系统

This module implements data collection and logging for autonomous driving research,
including sensor data recording, performance metrics, and analysis tools.
"""

import json
import logging
import os
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import h5py
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataLogger:
    """
    数据记录类 / Data Logger Class
    
    负责收集和记录自动驾驶系统的各种数据，用于分析和研究
    Responsible for collecting and logging various data from autonomous driving system for analysis and research
    """
    
    def __init__(self, log_dir: str = "data/logs", session_name: str = None):
        """
        初始化数据记录器 / Initialize data logger
        
        Args:
            log_dir: 日志目录 / Log directory
            session_name: 会话名称 / Session name
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create session name if not provided
        if session_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_name = f"session_{timestamp}"
        
        self.session_name = session_name
        self.session_dir = self.log_dir / session_name
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize data structures
        self.step_data = []
        self.performance_metrics = []
        self.sensor_data_buffer = []
        self.control_data_buffer = []
        
        # File paths
        self.metadata_file = self.session_dir / "metadata.json"
        self.step_data_file = self.session_dir / "step_data.h5"
        self.metrics_file = self.session_dir / "metrics.csv"
        self.summary_file = self.session_dir / "summary.json"
        
        # Session metadata
        self.session_start_time = time.time()
        self.total_steps = 0
        self.total_distance = 0.0
        self.max_speed = 0.0
        self.avg_speed = 0.0
        
        # Create metadata file
        self._create_metadata()
        
        logger.info(f"Data logger initialized for session: {session_name}")
        logger.info(f"Log directory: {self.session_dir}")
    
    def _create_metadata(self):
        """创建元数据文件 / Create metadata file"""
        metadata = {
            "session_name": self.session_name,
            "start_time": datetime.fromtimestamp(self.session_start_time).isoformat(),
            "description": "张涔熙的科研项目 - CARLA自动驾驶数据收集",
            "version": "1.0",
            "data_format": "HDF5 + CSV + JSON",
            "coordinate_system": "CARLA (UE4)",
            "units": {
                "distance": "meters",
                "speed": "m/s",
                "acceleration": "m/s²",
                "angle": "degrees",
                "time": "seconds"
            }
        }
        
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    def log_step(self, step_data: Dict):
        """
        记录单步数据 / Log step data
        
        Args:
            step_data: 单步数据字典 / Step data dictionary
        """
        try:
            self.total_steps += 1
            step_data['session_step'] = self.total_steps
            step_data['session_time'] = time.time() - self.session_start_time
            
            # Add to buffer
            self.step_data.append(step_data.copy())
            
            # Extract performance metrics
            self._extract_performance_metrics(step_data)
            
            # Periodic saving to prevent data loss
            if self.total_steps % 100 == 0:
                self._save_incremental_data()
            
        except Exception as e:
            logger.error(f"Error logging step data: {e}")
    
    def _extract_performance_metrics(self, step_data: Dict):
        """
        提取性能指标 / Extract performance metrics
        
        Args:
            step_data: 步骤数据 / Step data
        """
        try:
            # Extract vehicle state
            control = step_data.get('control', {})
            vehicle_speed = 0.0
            
            # Try to get speed from different sources
            if 'sensor_data' in step_data:
                sensor_data = step_data['sensor_data']
                # Speed could be in vehicle state or calculated from velocity
                vehicle_speed = sensor_data.get('vehicle_speed', 0.0)
            
            # Update session statistics
            self.max_speed = max(self.max_speed, vehicle_speed)
            
            # Calculate average speed
            if self.total_steps > 0:
                total_speed = sum([m.get('speed', 0.0) for m in self.performance_metrics])
                self.avg_speed = (total_speed + vehicle_speed) / (len(self.performance_metrics) + 1)
            
            # Create performance metrics entry
            metrics = {
                'step': self.total_steps,
                'timestamp': step_data.get('timestamp', time.time()),
                'session_time': step_data.get('session_time', 0.0),
                'speed': vehicle_speed,
                'throttle': control.get('throttle', 0.0),
                'brake': control.get('brake', 0.0),
                'steer': control.get('steer', 0.0),
                'hand_brake': control.get('hand_brake', False),
                'num_objects_detected': len(step_data.get('perception', {}).get('objects', [])),
                'num_lanes_detected': len(step_data.get('perception', {}).get('lanes', [])),
                'planning_valid': step_data.get('planned_path', {}).get('valid', False),
                'behavior': step_data.get('planned_path', {}).get('behavior', 'unknown')
            }
            
            self.performance_metrics.append(metrics)
            
        except Exception as e:
            logger.error(f"Error extracting performance metrics: {e}")
    
    def _save_incremental_data(self):
        """增量保存数据 / Save data incrementally"""
        try:
            # Save performance metrics as CSV
            if self.performance_metrics:
                df = pd.DataFrame(self.performance_metrics)
                df.to_csv(self.metrics_file, index=False)
            
            logger.debug(f"Incremental save at step {self.total_steps}")
            
        except Exception as e:
            logger.error(f"Error in incremental save: {e}")
    
    def save_sensor_data(self, sensor_type: str, data: np.ndarray, metadata: Dict = None):
        """
        保存传感器数据 / Save sensor data
        
        Args:
            sensor_type: 传感器类型 / Sensor type
            data: 传感器数据 / Sensor data
            metadata: 元数据 / Metadata
        """
        try:
            # Create sensor data directory
            sensor_dir = self.session_dir / "sensor_data" / sensor_type
            sensor_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = sensor_dir / f"{sensor_type}_{timestamp}.npy"
            
            # Save data
            np.save(filename, data)
            
            # Save metadata if provided
            if metadata:
                metadata_filename = sensor_dir / f"{sensor_type}_{timestamp}_meta.json"
                with open(metadata_filename, 'w') as f:
                    json.dump(metadata, f, indent=2)
            
        except Exception as e:
            logger.error(f"Error saving sensor data: {e}")
    
    def save_session(self):
        """保存完整会话数据 / Save complete session data"""
        try:
            logger.info("Saving session data...")
            
            # Save step data to HDF5
            self._save_step_data_hdf5()
            
            # Save final performance metrics
            if self.performance_metrics:
                df = pd.DataFrame(self.performance_metrics)
                df.to_csv(self.metrics_file, index=False)
            
            # Create session summary
            self._create_session_summary()
            
            logger.info(f"Session data saved successfully to {self.session_dir}")
            
        except Exception as e:
            logger.error(f"Error saving session: {e}")
    
    def _save_step_data_hdf5(self):
        """将步骤数据保存为HDF5格式 / Save step data as HDF5 format"""
        try:
            with h5py.File(self.step_data_file, 'w') as f:
                # Create groups for different data types
                perception_group = f.create_group('perception')
                planning_group = f.create_group('planning')
                control_group = f.create_group('control')
                metadata_group = f.create_group('metadata')
                
                # Process each step
                for i, step in enumerate(self.step_data):
                    step_group = f.create_group(f'step_{i:06d}')
                    
                    # Save basic info
                    step_group.attrs['step_number'] = step.get('step', i)
                    step_group.attrs['timestamp'] = step.get('timestamp', 0.0)
                    step_group.attrs['session_time'] = step.get('session_time', 0.0)
                    
                    # Save perception data
                    perception_data = step.get('perception', {})
                    if perception_data:
                        perc_step = perception_group.create_group(f'step_{i:06d}')
                        
                        # Save objects
                        objects = perception_data.get('objects', [])
                        if objects:
                            obj_data = self._serialize_objects(objects)
                            perc_step.create_dataset('objects', data=obj_data)
                        
                        # Save semantic map if available
                        semantic_map = perception_data.get('semantic_map')
                        if semantic_map is not None and isinstance(semantic_map, np.ndarray):
                            perc_step.create_dataset('semantic_map', data=semantic_map)
                    
                    # Save planning data
                    planning_data = step.get('planned_path', {})
                    if planning_data:
                        plan_step = planning_group.create_group(f'step_{i:06d}')
                        plan_step.attrs['behavior'] = planning_data.get('behavior', 'unknown')
                        plan_step.attrs['valid'] = planning_data.get('valid', False)
                        
                        # Save waypoints
                        waypoints = planning_data.get('waypoints', [])
                        if waypoints:
                            wp_data = self._serialize_waypoints(waypoints)
                            plan_step.create_dataset('waypoints', data=wp_data)
                    
                    # Save control data
                    control_data = step.get('control', {})
                    if control_data:
                        ctrl_step = control_group.create_group(f'step_{i:06d}')
                        ctrl_step.attrs['throttle'] = control_data.get('throttle', 0.0)
                        ctrl_step.attrs['brake'] = control_data.get('brake', 0.0)
                        ctrl_step.attrs['steer'] = control_data.get('steer', 0.0)
                        ctrl_step.attrs['hand_brake'] = control_data.get('hand_brake', False)
                        ctrl_step.attrs['reverse'] = control_data.get('reverse', False)
            
        except Exception as e:
            logger.error(f"Error saving HDF5 data: {e}")
    
    def _serialize_objects(self, objects: List[Dict]) -> np.ndarray:
        """序列化目标检测结果 / Serialize object detection results"""
        # Convert objects to structured numpy array
        if not objects:
            return np.array([])
        
        # Create structured array
        dtype = [
            ('class_name', 'U32'),
            ('confidence', 'f4'),
            ('center_x', 'f4'),
            ('center_y', 'f4'),
            ('center_z', 'f4'),
            ('size_x', 'f4'),
            ('size_y', 'f4'),
            ('size_z', 'f4')
        ]
        
        obj_array = np.zeros(len(objects), dtype=dtype)
        
        for i, obj in enumerate(objects):
            obj_array[i]['class_name'] = obj.get('class', obj.get('type', 'unknown'))
            obj_array[i]['confidence'] = obj.get('confidence', 1.0)
            
            center = obj.get('center', [0.0, 0.0, 0.0])
            obj_array[i]['center_x'] = center[0] if len(center) > 0 else 0.0
            obj_array[i]['center_y'] = center[1] if len(center) > 1 else 0.0
            obj_array[i]['center_z'] = center[2] if len(center) > 2 else 0.0
            
            size = obj.get('size', [1.0, 1.0, 1.0])
            obj_array[i]['size_x'] = size[0] if len(size) > 0 else 1.0
            obj_array[i]['size_y'] = size[1] if len(size) > 1 else 1.0
            obj_array[i]['size_z'] = size[2] if len(size) > 2 else 1.0
        
        return obj_array
    
    def _serialize_waypoints(self, waypoints: List[Dict]) -> np.ndarray:
        """序列化航点数据 / Serialize waypoint data"""
        if not waypoints:
            return np.array([])
        
        dtype = [
            ('position_x', 'f4'),
            ('position_y', 'f4'),
            ('speed', 'f4'),
            ('time', 'f4')
        ]
        
        wp_array = np.zeros(len(waypoints), dtype=dtype)
        
        for i, wp in enumerate(waypoints):
            pos = wp.get('position', [0.0, 0.0])
            wp_array[i]['position_x'] = pos[0] if len(pos) > 0 else 0.0
            wp_array[i]['position_y'] = pos[1] if len(pos) > 1 else 0.0
            wp_array[i]['speed'] = wp.get('speed', 0.0)
            wp_array[i]['time'] = wp.get('time', 0.0)
        
        return wp_array
    
    def _create_session_summary(self):
        """创建会话总结 / Create session summary"""
        try:
            session_duration = time.time() - self.session_start_time
            
            # Calculate total distance (simplified)
            if self.performance_metrics:
                speeds = [m['speed'] for m in self.performance_metrics]
                # Approximate distance as sum of speed * time_step
                time_step = 0.05  # Assuming 20 FPS
                self.total_distance = sum(speeds) * time_step
            
            summary = {
                "session_info": {
                    "name": self.session_name,
                    "start_time": datetime.fromtimestamp(self.session_start_time).isoformat(),
                    "end_time": datetime.now().isoformat(),
                    "duration_seconds": session_duration,
                    "total_steps": self.total_steps
                },
                "performance": {
                    "total_distance_m": round(self.total_distance, 2),
                    "max_speed_ms": round(self.max_speed, 2),
                    "avg_speed_ms": round(self.avg_speed, 2),
                    "total_data_points": len(self.step_data)
                },
                "data_files": {
                    "step_data": str(self.step_data_file),
                    "metrics": str(self.metrics_file),
                    "metadata": str(self.metadata_file)
                },
                "statistics": self._calculate_statistics()
            }
            
            with open(self.summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
        except Exception as e:
            logger.error(f"Error creating session summary: {e}")
    
    def _calculate_statistics(self) -> Dict:
        """计算统计信息 / Calculate statistics"""
        try:
            if not self.performance_metrics:
                return {}
            
            df = pd.DataFrame(self.performance_metrics)
            
            stats = {
                "speed_stats": {
                    "mean": float(df['speed'].mean()),
                    "std": float(df['speed'].std()),
                    "min": float(df['speed'].min()),
                    "max": float(df['speed'].max()),
                    "median": float(df['speed'].median())
                },
                "control_stats": {
                    "avg_throttle": float(df['throttle'].mean()),
                    "avg_brake": float(df['brake'].mean()),
                    "avg_steer_abs": float(df['steer'].abs().mean()),
                    "emergency_brakes": int(df['hand_brake'].sum())
                },
                "perception_stats": {
                    "avg_objects_detected": float(df['num_objects_detected'].mean()),
                    "avg_lanes_detected": float(df['num_lanes_detected'].mean()),
                    "planning_success_rate": float(df['planning_valid'].mean())
                },
                "behavior_distribution": df['behavior'].value_counts().to_dict()
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error calculating statistics: {e}")
            return {}
    
    def load_session_data(self, session_name: str) -> Dict:
        """
        加载会话数据 / Load session data
        
        Args:
            session_name: 会话名称 / Session name
            
        Returns:
            会话数据 / Session data
        """
        try:
            session_dir = self.log_dir / session_name
            
            # Load summary
            summary_file = session_dir / "summary.json"
            with open(summary_file, 'r', encoding='utf-8') as f:
                summary = json.load(f)
            
            # Load metrics
            metrics_file = session_dir / "metrics.csv"
            metrics_df = pd.read_csv(metrics_file)
            
            return {
                'summary': summary,
                'metrics': metrics_df,
                'session_dir': str(session_dir)
            }
            
        except Exception as e:
            logger.error(f"Error loading session data: {e}")
            return {}
    
    def get_session_list(self) -> List[str]:
        """
        获取会话列表 / Get session list
        
        Returns:
            会话名称列表 / List of session names
        """
        try:
            sessions = []
            for item in self.log_dir.iterdir():
                if item.is_dir() and (item / "metadata.json").exists():
                    sessions.append(item.name)
            
            return sorted(sessions)
            
        except Exception as e:
            logger.error(f"Error getting session list: {e}")
            return []


def main():
    """测试函数 / Test function"""
    # Initialize data logger
    logger_instance = DataLogger()
    
    # Simulate some data logging
    for i in range(10):
        dummy_step_data = {
            'step': i,
            'timestamp': time.time(),
            'sensor_data': {
                'camera': {'data': np.random.randint(0, 255, (480, 640, 3))},
                'lidar': {'data': np.random.randn(1000, 4)},
                'vehicle_speed': 10.0 + i * 0.5
            },
            'perception': {
                'objects': [
                    {'class': 'car', 'confidence': 0.9, 'center': [10.0, 5.0, 0.0], 'size': [4.0, 2.0, 1.5]}
                ],
                'lanes': [{'points': [(0, 100), (10, 200)]}]
            },
            'planned_path': {
                'waypoints': [{'position': [i*5, 0], 'speed': 15.0, 'time': i*0.5}],
                'behavior': 'cruise',
                'valid': True
            },
            'control': {
                'throttle': 0.5,
                'brake': 0.0,
                'steer': 0.1 * np.sin(i * 0.5),
                'hand_brake': False
            }
        }
        
        logger_instance.log_step(dummy_step_data)
        time.sleep(0.1)
    
    # Save session
    logger_instance.save_session()
    
    # Get session list
    sessions = logger_instance.get_session_list()
    logger.info(f"Available sessions: {sessions}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()