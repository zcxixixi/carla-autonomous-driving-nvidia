"""Simple controller wrapper for using affordances to compute control commands.

This module provides a basic PID lateral controller and a distance-based
longitudinal controller. When a risk_score exceeds a threshold, the controller
switches to a conservative mode.
"""
from dataclasses import dataclass
import math


@dataclass
class Control:
    throttle: float = 0.0
    brake: float = 0.0
    steer: float = 0.0


class SimplePID:
    def __init__(self, kp=1.0, ki=0.0, kd=0.1):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.prev_error = None

    def step(self, error, dt=0.1):
        self.integral += error * dt
        derivative = 0.0 if self.prev_error is None else (error - self.prev_error) / dt
        self.prev_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative


class AffordanceController:
    def __init__(self, risk_threshold=0.7):
        self.pid = SimplePID(kp=0.8, kd=0.05)
        self.risk_threshold = risk_threshold

    def control(self, affordances: dict, dt=0.1) -> Control:
        # affordances expected: lane_offset, front_distance, speed_limit, risk_score
        lane_offset = affordances.get('lane_offset', None)
        front_distance = affordances.get('front_distance', None)
        speed_limit = affordances.get('speed_limit', None)
        risk = affordances.get('risk_score', 0.0)

        ctrl = Control()

        # Lateral: simple PID on lane offset
        if lane_offset is not None:
            steer = -self.pid.step(lane_offset, dt=dt)
            ctrl.steer = max(-1.0, min(1.0, steer))

        # Longitudinal: conservative policy with risk switching
        target_speed = 10.0  # default m/s
        if speed_limit is not None and speed_limit > 0:
            target_speed = speed_limit / 3.6

        if front_distance is None:
            # no front vehicle: maintain target speed
            desired_speed = target_speed
        else:
            # simple distance-based rule: slow down if too close
            desired_speed = min(target_speed, front_distance * 2.0)

        # If risk high, be conservative
        if risk is not None and risk > self.risk_threshold:
            desired_speed = min(desired_speed, 3.0)

        # Assume current speed unknown in this wrapper; output throttle/brake as proxy
        # Simple proportional mapping (to be replaced with model-based controller)
        speed_error = desired_speed - (affordances.get('ego_speed', desired_speed))
        if speed_error > 0.1:
            ctrl.throttle = max(0.0, min(1.0, speed_error / 3.0))
            ctrl.brake = 0.0
        else:
            ctrl.throttle = 0.0
            ctrl.brake = max(0.0, min(1.0, -speed_error / 3.0))

        return ctrl


if __name__ == '__main__':
    # quick sanity run
    c = AffordanceController()
    aff = {'lane_offset': 0.2, 'front_distance': 8.0, 'speed_limit': 50, 'risk_score': 0.1, 'ego_speed': 8.0}
    print(c.control(aff))
