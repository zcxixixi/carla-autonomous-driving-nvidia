#!/usr/bin/env python3
"""
路径规划模块 / Path Planning Module
张涔熙的科研项目 - 自动驾驶路径规划算法

This module implements path planning algorithms for autonomous driving,
including A* pathfinding, trajectory optimization, and behavior planning.
"""

import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class PathPlanner:
    """
    路径规划类 / Path Planning Class
    
    实现自动驾驶的路径规划功能，包括全局路径规划和局部路径规划
    Implements path planning for autonomous driving including global and local path planning
    """
    
    def __init__(self):
        """初始化路径规划器 / Initialize path planner"""
        # Planning parameters
        self.planning_horizon = 50.0  # Planning horizon in meters
        self.time_horizon = 5.0       # Time horizon in seconds
        self.safety_margin = 2.0      # Safety margin in meters
        self.max_speed = 15.0         # Maximum speed in m/s
        self.max_acceleration = 3.0   # Maximum acceleration in m/s²
        self.max_deceleration = 5.0   # Maximum deceleration in m/s²
        self.max_steering_angle = 0.5 # Maximum steering angle in radians
        
        # Grid parameters for A* planning
        self.grid_resolution = 0.5    # Grid resolution in meters
        self.grid_size = 200          # Grid size
        
        # Current path
        self.current_path = []
        self.current_trajectory = []
        
        logger.info("Path planner initialized")
    
    def plan(self, current_pose: Dict, perception_data: Dict, goal: Dict) -> Dict:
        """
        主路径规划函数 / Main path planning function
        
        Args:
            current_pose: 当前位姿 / Current pose
            perception_data: 感知数据 / Perception data
            goal: 目标位置 / Goal location
            
        Returns:
            规划结果 / Planning result
        """
        try:
            # Extract current position and orientation
            if not current_pose:
                logger.warning("No current pose provided")
                return self._get_empty_plan()
            
            current_pos = self._extract_position(current_pose)
            current_yaw = self._extract_yaw(current_pose)
            goal_pos = self._extract_goal_position(goal)
            
            # Global path planning
            global_path = self._plan_global_path(current_pos, goal_pos, perception_data)
            
            # Local path planning and trajectory optimization
            local_trajectory = self._plan_local_trajectory(
                current_pos, current_yaw, global_path, perception_data
            )
            
            # Behavior planning
            behavior = self._plan_behavior(perception_data, local_trajectory)
            
            # Generate control waypoints
            waypoints = self._generate_waypoints(local_trajectory)
            
            return {
                'global_path': global_path,
                'local_trajectory': local_trajectory,
                'waypoints': waypoints,
                'behavior': behavior,
                'planning_time': self.time_horizon,
                'valid': len(waypoints) > 0
            }
            
        except Exception as e:
            logger.error(f"Error in path planning: {e}")
            return self._get_empty_plan()
    
    def _extract_position(self, pose: Dict) -> Tuple[float, float]:
        """提取位置信息 / Extract position information"""
        if hasattr(pose, 'location'):
            return (pose.location.x, pose.location.y)
        elif 'location' in pose:
            loc = pose['location']
            return (loc[0], loc[1])
        else:
            return (0.0, 0.0)
    
    def _extract_yaw(self, pose: Dict) -> float:
        """提取偏航角 / Extract yaw angle"""
        if hasattr(pose, 'rotation'):
            return math.radians(pose.rotation.yaw)
        elif 'rotation' in pose:
            rot = pose['rotation']
            return math.radians(rot[1])  # yaw is second element
        else:
            return 0.0
    
    def _extract_goal_position(self, goal: Dict) -> Tuple[float, float]:
        """提取目标位置 / Extract goal position"""
        if not goal:
            return (100.0, 0.0)  # Default goal
        
        if hasattr(goal, 'x'):
            return (goal.x, goal.y)
        elif isinstance(goal, (list, tuple)) and len(goal) >= 2:
            return (goal[0], goal[1])
        else:
            return (100.0, 0.0)
    
    def _plan_global_path(self, start: Tuple[float, float], 
                         goal: Tuple[float, float], 
                         perception_data: Dict) -> List[Tuple[float, float]]:
        """
        全局路径规划 / Global path planning
        
        Args:
            start: 起始位置 / Start position
            goal: 目标位置 / Goal position
            perception_data: 感知数据 / Perception data
            
        Returns:
            全局路径点列表 / List of global path points
        """
        # Create occupancy grid from perception data
        occupancy_grid = self._create_occupancy_grid(perception_data)
        
        # A* path finding
        path = self._a_star_search(start, goal, occupancy_grid)
        
        # Smooth the path
        smoothed_path = self._smooth_path(path)
        
        return smoothed_path
    
    def _create_occupancy_grid(self, perception_data: Dict) -> np.ndarray:
        """
        创建占用栅格地图 / Create occupancy grid map
        
        Args:
            perception_data: 感知数据 / Perception data
            
        Returns:
            占用栅格 / Occupancy grid
        """
        grid = np.zeros((self.grid_size, self.grid_size))
        
        # Mark obstacles from perception data
        objects = perception_data.get('objects', [])
        obstacles = perception_data.get('obstacles', [])
        
        # Process 3D objects
        for obj in objects:
            if 'center' in obj:
                center = obj['center']
                x_grid = int(center[0] / self.grid_resolution + self.grid_size // 2)
                y_grid = int(center[1] / self.grid_resolution + self.grid_size // 2)
                
                # Mark object area as occupied
                size = obj.get('size', [2, 2, 2])
                x_size = max(1, int(size[0] / self.grid_resolution))
                y_size = max(1, int(size[1] / self.grid_resolution))
                
                for dx in range(-x_size//2, x_size//2 + 1):
                    for dy in range(-y_size//2, y_size//2 + 1):
                        nx, ny = x_grid + dx, y_grid + dy
                        if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                            grid[ny, nx] = 1
        
        # Process 2D obstacles
        for obstacle in obstacles:
            if 'position' in obstacle:
                pos = obstacle['position']
                x_grid = int(pos[0] / self.grid_resolution + self.grid_size // 2)
                y_grid = int(pos[1] / self.grid_resolution + self.grid_size // 2)
                
                if 0 <= x_grid < self.grid_size and 0 <= y_grid < self.grid_size:
                    grid[y_grid, x_grid] = 1
        
        # Apply safety inflation
        grid = self._inflate_obstacles(grid)
        
        return grid
    
    def _inflate_obstacles(self, grid: np.ndarray) -> np.ndarray:
        """
        障碍物膨胀 / Obstacle inflation
        
        Args:
            grid: 原始栅格 / Original grid
            
        Returns:
            膨胀后的栅格 / Inflated grid
        """
        inflation_radius = int(self.safety_margin / self.grid_resolution)
        inflated_grid = grid.copy()
        
        obstacle_indices = np.where(grid == 1)
        
        for y, x in zip(obstacle_indices[0], obstacle_indices[1]):
            for dy in range(-inflation_radius, inflation_radius + 1):
                for dx in range(-inflation_radius, inflation_radius + 1):
                    if dx*dx + dy*dy <= inflation_radius*inflation_radius:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < grid.shape[0] and 0 <= nx < grid.shape[1]:
                            inflated_grid[ny, nx] = 1
        
        return inflated_grid
    
    def _a_star_search(self, start: Tuple[float, float], 
                      goal: Tuple[float, float], 
                      grid: np.ndarray) -> List[Tuple[float, float]]:
        """
        A*搜索算法 / A* search algorithm
        
        Args:
            start: 起始位置 / Start position
            goal: 目标位置 / Goal position
            grid: 占用栅格 / Occupancy grid
            
        Returns:
            路径点列表 / List of path points
        """
        # Convert world coordinates to grid coordinates
        start_grid = (
            int(start[0] / self.grid_resolution + self.grid_size // 2),
            int(start[1] / self.grid_resolution + self.grid_size // 2)
        )
        goal_grid = (
            int(goal[0] / self.grid_resolution + self.grid_size // 2),
            int(goal[1] / self.grid_resolution + self.grid_size // 2)
        )
        
        # Check bounds
        if not (0 <= start_grid[0] < self.grid_size and 0 <= start_grid[1] < self.grid_size):
            logger.warning("Start position out of grid bounds")
            return [start, goal]
        
        if not (0 <= goal_grid[0] < self.grid_size and 0 <= goal_grid[1] < self.grid_size):
            logger.warning("Goal position out of grid bounds")
            return [start, goal]
        
        # A* implementation
        open_set = [start_grid]
        came_from = {}
        g_score = {start_grid: 0}
        f_score = {start_grid: self._heuristic(start_grid, goal_grid)}
        
        while open_set:
            # Find node with lowest f_score
            current = min(open_set, key=lambda x: f_score.get(x, float('inf')))
            
            if current == goal_grid:
                # Reconstruct path
                path = []
                while current in came_from:
                    # Convert back to world coordinates
                    world_x = (current[0] - self.grid_size // 2) * self.grid_resolution
                    world_y = (current[1] - self.grid_size // 2) * self.grid_resolution
                    path.append((world_x, world_y))
                    current = came_from[current]
                
                # Add start position
                path.append(start)
                path.reverse()
                return path
            
            open_set.remove(current)
            
            # Check neighbors
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                neighbor = (current[0] + dx, current[1] + dy)
                
                # Check bounds and obstacles
                if (0 <= neighbor[0] < self.grid_size and 
                    0 <= neighbor[1] < self.grid_size and 
                    grid[neighbor[1], neighbor[0]] == 0):
                    
                    # Calculate tentative g_score
                    diagonal = (dx != 0 and dy != 0)
                    tentative_g = g_score[current] + (1.414 if diagonal else 1.0)
                    
                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        f_score[neighbor] = tentative_g + self._heuristic(neighbor, goal_grid)
                        
                        if neighbor not in open_set:
                            open_set.append(neighbor)
        
        # No path found, return direct line
        logger.warning("No path found using A*, returning direct line")
        return [start, goal]
    
    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """A*启发式函数 / A* heuristic function"""
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    
    def _smooth_path(self, path: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        路径平滑 / Path smoothing
        
        Args:
            path: 原始路径 / Original path
            
        Returns:
            平滑后的路径 / Smoothed path
        """
        if len(path) < 3:
            return path
        
        smoothed = [path[0]]
        
        for i in range(1, len(path) - 1):
            # Apply simple smoothing filter
            prev_point = path[i - 1]
            curr_point = path[i]
            next_point = path[i + 1]
            
            smooth_x = 0.25 * prev_point[0] + 0.5 * curr_point[0] + 0.25 * next_point[0]
            smooth_y = 0.25 * prev_point[1] + 0.5 * curr_point[1] + 0.25 * next_point[1]
            
            smoothed.append((smooth_x, smooth_y))
        
        smoothed.append(path[-1])
        return smoothed
    
    def _plan_local_trajectory(self, current_pos: Tuple[float, float], 
                              current_yaw: float,
                              global_path: List[Tuple[float, float]], 
                              perception_data: Dict) -> List[Dict]:
        """
        局部轨迹规划 / Local trajectory planning
        
        Args:
            current_pos: 当前位置 / Current position
            current_yaw: 当前偏航角 / Current yaw angle
            global_path: 全局路径 / Global path
            perception_data: 感知数据 / Perception data
            
        Returns:
            局部轨迹 / Local trajectory
        """
        trajectory = []
        
        if not global_path:
            return trajectory
        
        # Find closest point on global path
        closest_idx = self._find_closest_point(current_pos, global_path)
        
        # Generate trajectory points
        dt = 0.2  # Time step in seconds
        current_speed = 10.0  # Current speed in m/s (simplified)
        
        for i in range(int(self.time_horizon / dt)):
            t = i * dt
            
            # Get target point from global path
            target_idx = min(closest_idx + i, len(global_path) - 1)
            target_point = global_path[target_idx]
            
            # Calculate desired speed based on obstacles
            desired_speed = self._calculate_desired_speed(target_point, perception_data)
            
            # Generate trajectory point
            traj_point = {
                'time': t,
                'position': target_point,
                'speed': desired_speed,
                'acceleration': 0.0,  # Simplified
                'steering_angle': 0.0  # Simplified
            }
            
            trajectory.append(traj_point)
        
        return trajectory
    
    def _find_closest_point(self, pos: Tuple[float, float], 
                           path: List[Tuple[float, float]]) -> int:
        """查找最近路径点 / Find closest path point"""
        min_dist = float('inf')
        closest_idx = 0
        
        for i, point in enumerate(path):
            dist = math.sqrt((pos[0] - point[0])**2 + (pos[1] - point[1])**2)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        
        return closest_idx
    
    def _calculate_desired_speed(self, target_point: Tuple[float, float], 
                               perception_data: Dict) -> float:
        """
        计算期望速度 / Calculate desired speed
        
        Args:
            target_point: 目标点 / Target point
            perception_data: 感知数据 / Perception data
            
        Returns:
            期望速度 / Desired speed
        """
        base_speed = self.max_speed
        
        # Check for obstacles near target point
        objects = perception_data.get('objects', [])
        
        for obj in objects:
            if 'center' in obj:
                obj_pos = obj['center']
                distance = math.sqrt(
                    (target_point[0] - obj_pos[0])**2 + 
                    (target_point[1] - obj_pos[1])**2
                )
                
                # Reduce speed based on distance to obstacle
                if distance < 20.0:
                    speed_reduction = (20.0 - distance) / 20.0
                    base_speed *= (1.0 - 0.8 * speed_reduction)
        
        # Check traffic lights
        traffic_lights = perception_data.get('traffic_lights', [])
        for tl in traffic_lights:
            if tl.get('color') == 'red':
                base_speed *= 0.3  # Slow down for red lights
            elif tl.get('color') == 'yellow':
                base_speed *= 0.7  # Slow down for yellow lights
        
        return max(0.0, min(base_speed, self.max_speed))
    
    def _plan_behavior(self, perception_data: Dict, trajectory: List[Dict]) -> str:
        """
        行为规划 / Behavior planning
        
        Args:
            perception_data: 感知数据 / Perception data
            trajectory: 轨迹 / Trajectory
            
        Returns:
            行为类型 / Behavior type
        """
        # Simplified behavior planning
        objects = perception_data.get('objects', [])
        traffic_lights = perception_data.get('traffic_lights', [])
        
        # Check for emergency stop conditions
        for obj in objects:
            if 'center' in obj and obj.get('type') == 'person':
                distance = math.sqrt(obj['center'][0]**2 + obj['center'][1]**2)
                if distance < 5.0:
                    return 'emergency_stop'
        
        # Check traffic lights
        for tl in traffic_lights:
            if tl.get('color') == 'red':
                return 'stop'
            elif tl.get('color') == 'yellow':
                return 'slow_down'
        
        # Check for lane change opportunities
        if len(objects) > 0:
            return 'lane_follow'
        
        return 'cruise'
    
    def _generate_waypoints(self, trajectory: List[Dict]) -> List[Dict]:
        """
        生成控制航点 / Generate control waypoints
        
        Args:
            trajectory: 轨迹 / Trajectory
            
        Returns:
            航点列表 / List of waypoints
        """
        waypoints = []
        
        for i, traj_point in enumerate(trajectory[:10]):  # Use first 10 points
            waypoint = {
                'position': traj_point['position'],
                'speed': traj_point['speed'],
                'time': traj_point['time'],
                'index': i
            }
            waypoints.append(waypoint)
        
        return waypoints
    
    def _get_empty_plan(self) -> Dict:
        """获取空规划结果 / Get empty planning result"""
        return {
            'global_path': [],
            'local_trajectory': [],
            'waypoints': [],
            'behavior': 'stop',
            'planning_time': 0.0,
            'valid': False
        }


def main():
    """测试函数 / Test function"""
    planner = PathPlanner()
    
    # Create dummy data
    current_pose = {
        'location': [0.0, 0.0, 0.0],
        'rotation': [0.0, 0.0, 0.0]
    }
    
    perception_data = {
        'objects': [
            {'center': [10.0, 5.0, 0.0], 'type': 'car', 'size': [4.0, 2.0, 1.5]},
            {'center': [20.0, -3.0, 0.0], 'type': 'truck', 'size': [8.0, 2.5, 3.0]}
        ],
        'obstacles': [],
        'traffic_lights': [{'color': 'green', 'confidence': 0.9}]
    }
    
    goal = [50.0, 0.0]
    
    # Plan path
    result = planner.plan(current_pose, perception_data, goal)
    
    logger.info(f"Planning result: {len(result['waypoints'])} waypoints generated")
    logger.info(f"Behavior: {result['behavior']}")
    logger.info(f"Valid plan: {result['valid']}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()