#!/usr/bin/env python3
"""
车辆控制模块 / Vehicle Control Module
张涔熙的科研项目 - 自动驾驶车辆控制系统

This module implements vehicle control algorithms for autonomous driving,
including PID controllers, model predictive control, and safety mechanisms.
"""

import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class VehicleController:
    """
    车辆控制类 / Vehicle Controller Class
    
    实现车辆的纵向和横向控制，包括速度控制、转向控制和安全机制
    Implements longitudinal and lateral vehicle control including speed control, steering control, and safety mechanisms
    """
    
    def __init__(self):
        """初始化车辆控制器 / Initialize vehicle controller"""
        # PID controller parameters for speed control
        self.speed_kp = 0.8
        self.speed_ki = 0.02
        self.speed_kd = 0.1
        self.speed_error_integral = 0.0
        self.speed_error_previous = 0.0
        
        # PID controller parameters for steering control
        self.steer_kp = 1.5
        self.steer_ki = 0.01
        self.steer_kd = 0.05
        self.steer_error_integral = 0.0
        self.steer_error_previous = 0.0
        
        # Control limits
        self.max_throttle = 1.0
        self.max_brake = 1.0
        self.max_steer = 0.8
        self.max_speed = 30.0  # m/s
        
        # Safety parameters
        self.emergency_brake_distance = 5.0  # meters
        self.comfort_deceleration = 3.0      # m/s²
        self.emergency_deceleration = 8.0    # m/s²
        
        # Vehicle parameters
        self.wheelbase = 2.7  # meters
        self.dt = 0.05        # control timestep
        
        # Control history for smoothing
        self.throttle_history = []
        self.brake_history = []
        self.steer_history = []
        self.history_length = 5
        
        logger.info("Vehicle controller initialized")
    
    def compute_control(self, current_state: Dict, planned_path: Dict) -> Dict:
        """
        计算控制命令 / Compute control commands
        
        Args:
            current_state: 当前车辆状态 / Current vehicle state
            planned_path: 规划路径 / Planned path
            
        Returns:
            控制命令 / Control commands
        """
        try:
            if not current_state or not planned_path:
                return self._get_emergency_stop()
            
            # Extract current vehicle state
            current_speed = current_state.get('speed', 0.0)
            current_pos = current_state.get('location', [0.0, 0.0, 0.0])[:2]
            current_yaw = math.radians(current_state.get('rotation', [0.0, 0.0, 0.0])[1])
            
            # Get target waypoints
            waypoints = planned_path.get('waypoints', [])
            behavior = planned_path.get('behavior', 'stop')
            
            if not waypoints:
                return self._get_emergency_stop()
            
            # Safety check
            if self._emergency_stop_required(current_state, planned_path):
                return self._get_emergency_stop()
            
            # Compute longitudinal control (throttle/brake)
            throttle, brake = self._compute_longitudinal_control(
                current_speed, waypoints, behavior
            )
            
            # Compute lateral control (steering)
            steer = self._compute_lateral_control(
                current_pos, current_yaw, waypoints
            )
            
            # Apply control smoothing
            throttle = self._smooth_control(throttle, self.throttle_history)
            brake = self._smooth_control(brake, self.brake_history)
            steer = self._smooth_control(steer, self.steer_history)
            
            # Ensure control limits
            control_command = {
                'throttle': np.clip(throttle, 0.0, self.max_throttle),
                'brake': np.clip(brake, 0.0, self.max_brake),
                'steer': np.clip(steer, -self.max_steer, self.max_steer),
                'hand_brake': False,
                'reverse': False
            }
            
            return control_command
            
        except Exception as e:
            logger.error(f"Error in control computation: {e}")
            return self._get_emergency_stop()
    
    def _compute_longitudinal_control(self, current_speed: float, 
                                    waypoints: List[Dict], 
                                    behavior: str) -> Tuple[float, float]:
        """
        计算纵向控制 / Compute longitudinal control
        
        Args:
            current_speed: 当前速度 / Current speed
            waypoints: 航点列表 / List of waypoints
            behavior: 行为类型 / Behavior type
            
        Returns:
            油门和刹车值 / Throttle and brake values
        """
        # Determine target speed based on behavior
        if behavior == 'emergency_stop':
            target_speed = 0.0
        elif behavior == 'stop':
            target_speed = 0.0
        elif behavior == 'slow_down':
            target_speed = min(5.0, current_speed * 0.7)
        else:
            # Use speed from nearest waypoint
            target_speed = waypoints[0].get('speed', 10.0) if waypoints else 10.0
        
        # Speed error
        speed_error = target_speed - current_speed
        
        # PID control for speed
        self.speed_error_integral += speed_error * self.dt
        speed_error_derivative = (speed_error - self.speed_error_previous) / self.dt
        
        # Anti-windup for integral term
        self.speed_error_integral = np.clip(self.speed_error_integral, -10.0, 10.0)
        
        # Compute PID output
        speed_control = (self.speed_kp * speed_error + 
                        self.speed_ki * self.speed_error_integral + 
                        self.speed_kd * speed_error_derivative)
        
        # Update previous error
        self.speed_error_previous = speed_error
        
        # Convert to throttle/brake commands
        if speed_control > 0:
            throttle = np.clip(speed_control, 0.0, 1.0)
            brake = 0.0
        else:
            throttle = 0.0
            brake = np.clip(-speed_control, 0.0, 1.0)
        
        # Apply emergency braking if needed
        if behavior == 'emergency_stop':
            throttle = 0.0
            brake = 1.0
        
        return throttle, brake
    
    def _compute_lateral_control(self, current_pos: List[float], 
                               current_yaw: float, 
                               waypoints: List[Dict]) -> float:
        """
        计算横向控制 / Compute lateral control
        
        Args:
            current_pos: 当前位置 / Current position
            current_yaw: 当前偏航角 / Current yaw angle
            waypoints: 航点列表 / List of waypoints
            
        Returns:
            转向角度 / Steering angle
        """
        if not waypoints:
            return 0.0
        
        # Use pure pursuit controller
        lookahead_distance = 5.0  # meters
        target_point = self._find_lookahead_point(current_pos, waypoints, lookahead_distance)
        
        if target_point is None:
            return 0.0
        
        # Calculate angle to target
        dx = target_point[0] - current_pos[0]
        dy = target_point[1] - current_pos[1]
        target_angle = math.atan2(dy, dx)
        
        # Calculate heading error
        heading_error = self._normalize_angle(target_angle - current_yaw)
        
        # Pure pursuit steering calculation
        distance_to_target = math.sqrt(dx*dx + dy*dy)
        steer_angle = math.atan2(2.0 * self.wheelbase * math.sin(heading_error), distance_to_target)
        
        # Apply PID for smoothing
        self.steer_error_integral += heading_error * self.dt
        steer_error_derivative = (heading_error - self.steer_error_previous) / self.dt
        
        # Anti-windup
        self.steer_error_integral = np.clip(self.steer_error_integral, -1.0, 1.0)
        
        # PID output
        steer_correction = (self.steer_kp * heading_error + 
                           self.steer_ki * self.steer_error_integral + 
                           self.steer_kd * steer_error_derivative)
        
        self.steer_error_previous = heading_error
        
        # Combine pure pursuit and PID
        final_steer = steer_angle + 0.3 * steer_correction
        
        return final_steer
    
    def _find_lookahead_point(self, current_pos: List[float], 
                            waypoints: List[Dict], 
                            lookahead_distance: float) -> Optional[Tuple[float, float]]:
        """
        查找前瞻点 / Find lookahead point
        
        Args:
            current_pos: 当前位置 / Current position
            waypoints: 航点列表 / List of waypoints
            lookahead_distance: 前瞻距离 / Lookahead distance
            
        Returns:
            前瞻点 / Lookahead point
        """
        for waypoint in waypoints:
            wp_pos = waypoint.get('position', [0.0, 0.0])
            distance = math.sqrt(
                (wp_pos[0] - current_pos[0])**2 + 
                (wp_pos[1] - current_pos[1])**2
            )
            
            if distance >= lookahead_distance:
                return (wp_pos[0], wp_pos[1])
        
        # If no point found at lookahead distance, use the farthest waypoint
        if waypoints:
            last_wp = waypoints[-1]
            wp_pos = last_wp.get('position', [0.0, 0.0])
            return (wp_pos[0], wp_pos[1])
        
        return None
    
    def _normalize_angle(self, angle: float) -> float:
        """
        角度归一化 / Angle normalization
        
        Args:
            angle: 输入角度 / Input angle
            
        Returns:
            归一化角度 / Normalized angle
        """
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle
    
    def _emergency_stop_required(self, current_state: Dict, planned_path: Dict) -> bool:
        """
        检查是否需要紧急停车 / Check if emergency stop is required
        
        Args:
            current_state: 当前状态 / Current state
            planned_path: 规划路径 / Planned path
            
        Returns:
            是否紧急停车 / Whether emergency stop is required
        """
        behavior = planned_path.get('behavior', 'cruise')
        
        # Emergency stop behavior
        if behavior == 'emergency_stop':
            return True
        
        # Check for collision risk (simplified)
        current_speed = current_state.get('speed', 0.0)
        if current_speed > 0:
            # Calculate stopping distance
            stopping_distance = (current_speed * current_speed) / (2 * self.comfort_deceleration)
            
            # Check if emergency stop is needed based on path validity
            if not planned_path.get('valid', False):
                return True
            
            waypoints = planned_path.get('waypoints', [])
            if not waypoints:
                return True
        
        return False
    
    def _smooth_control(self, new_value: float, history: List[float]) -> float:
        """
        控制命令平滑 / Control command smoothing
        
        Args:
            new_value: 新控制值 / New control value
            history: 历史控制值 / Control history
            
        Returns:
            平滑后的控制值 / Smoothed control value
        """
        # Add new value to history
        history.append(new_value)
        
        # Keep only recent history
        if len(history) > self.history_length:
            history.pop(0)
        
        # Apply moving average filter
        if len(history) > 1:
            return sum(history) / len(history)
        else:
            return new_value
    
    def _get_emergency_stop(self) -> Dict:
        """
        获取紧急停车命令 / Get emergency stop command
        
        Returns:
            紧急停车控制命令 / Emergency stop control command
        """
        return {
            'throttle': 0.0,
            'brake': 1.0,
            'steer': 0.0,
            'hand_brake': True,
            'reverse': False
        }
    
    def reset_controllers(self):
        """重置控制器状态 / Reset controller state"""
        self.speed_error_integral = 0.0
        self.speed_error_previous = 0.0
        self.steer_error_integral = 0.0
        self.steer_error_previous = 0.0
        
        self.throttle_history.clear()
        self.brake_history.clear()
        self.steer_history.clear()
        
        logger.info("Vehicle controller reset")
    
    def set_pid_parameters(self, speed_params: Tuple[float, float, float] = None,
                          steer_params: Tuple[float, float, float] = None):
        """
        设置PID参数 / Set PID parameters
        
        Args:
            speed_params: 速度PID参数 (kp, ki, kd) / Speed PID parameters
            steer_params: 转向PID参数 (kp, ki, kd) / Steering PID parameters
        """
        if speed_params:
            self.speed_kp, self.speed_ki, self.speed_kd = speed_params
            logger.info(f"Speed PID parameters updated: {speed_params}")
        
        if steer_params:
            self.steer_kp, self.steer_ki, self.steer_kd = steer_params
            logger.info(f"Steering PID parameters updated: {steer_params}")
    
    def get_control_status(self) -> Dict:
        """
        获取控制器状态 / Get controller status
        
        Returns:
            控制器状态信息 / Controller status information
        """
        return {
            'speed_error_integral': self.speed_error_integral,
            'steer_error_integral': self.steer_error_integral,
            'throttle_history_avg': sum(self.throttle_history) / len(self.throttle_history) if self.throttle_history else 0.0,
            'brake_history_avg': sum(self.brake_history) / len(self.brake_history) if self.brake_history else 0.0,
            'steer_history_avg': sum(self.steer_history) / len(self.steer_history) if self.steer_history else 0.0
        }


class ModelPredictiveController:
    """
    模型预测控制器 / Model Predictive Controller
    
    高级控制器，用于更精确的车辆控制
    Advanced controller for more precise vehicle control
    """
    
    def __init__(self, prediction_horizon: int = 10):
        """
        初始化MPC控制器 / Initialize MPC controller
        
        Args:
            prediction_horizon: 预测时域 / Prediction horizon
        """
        self.prediction_horizon = prediction_horizon
        self.dt = 0.1
        
        # Vehicle model parameters
        self.mass = 1500.0        # kg
        self.drag_coefficient = 0.3
        self.rolling_resistance = 0.01
        
        logger.info("Model Predictive Controller initialized")
    
    def compute_optimal_control(self, current_state: Dict, 
                              reference_trajectory: List[Dict]) -> Dict:
        """
        计算最优控制 / Compute optimal control
        
        Args:
            current_state: 当前状态 / Current state
            reference_trajectory: 参考轨迹 / Reference trajectory
            
        Returns:
            最优控制命令 / Optimal control command
        """
        # Simplified MPC implementation
        # In practice, this would use optimization solvers like CVXPY or CasADi
        
        if not reference_trajectory:
            return {'throttle': 0.0, 'brake': 0.0, 'steer': 0.0, 'hand_brake': False, 'reverse': False}
        
        # Extract current state
        current_speed = current_state.get('speed', 0.0)
        current_pos = current_state.get('location', [0.0, 0.0, 0.0])[:2]
        
        # Get target from reference trajectory
        target = reference_trajectory[0]
        target_speed = target.get('speed', 10.0)
        target_pos = target.get('position', [0.0, 0.0])
        
        # Simple proportional control as MPC approximation
        speed_error = target_speed - current_speed
        throttle = np.clip(speed_error * 0.5, 0.0, 1.0)
        brake = np.clip(-speed_error * 0.8, 0.0, 1.0)
        
        # Lateral control
        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]
        lateral_error = math.atan2(dy, dx)
        steer = np.clip(lateral_error * 0.5, -0.8, 0.8)
        
        return {
            'throttle': throttle,
            'brake': brake,
            'steer': steer,
            'hand_brake': False,
            'reverse': False
        }


def main():
    """测试函数 / Test function"""
    controller = VehicleController()
    
    # Create dummy state and path
    current_state = {
        'speed': 10.0,
        'location': [0.0, 0.0, 0.0],
        'rotation': [0.0, 45.0, 0.0],  # 45 degree yaw
        'velocity': [7.07, 7.07, 0.0],
        'acceleration': [0.0, 0.0, 0.0]
    }
    
    planned_path = {
        'waypoints': [
            {'position': [10.0, 10.0], 'speed': 15.0, 'time': 1.0},
            {'position': [20.0, 20.0], 'speed': 15.0, 'time': 2.0},
            {'position': [30.0, 30.0], 'speed': 15.0, 'time': 3.0}
        ],
        'behavior': 'cruise',
        'valid': True
    }
    
    # Compute control
    control = controller.compute_control(current_state, planned_path)
    
    logger.info(f"Control command: {control}")
    logger.info(f"Controller status: {controller.get_control_status()}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()