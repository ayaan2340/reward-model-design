"""
Robosuite 1.5.x: JointVelocityController assigns `self.torque_compensation = ...` (a bool from
`use_torque_compensation`), but `Controller` already defines `torque_compensation` as a read-only
@property returning gravity torques (`qfrc_bias`). That makes construction fail with:
`AttributeError: property 'torque_compensation' ... has no setter`.

Upstream still has the same pattern; this module fixes it by using `_use_torque_compensation` for the
flag and the inherited `torque_compensation` property for gravity in `run_controller`.

See: https://robosuite.ai/docs/modules/controllers.html (Joint Space Control - Velocity)
"""

from __future__ import annotations

import numpy as np

_PATCHED = False


def apply_joint_velocity_controller_fix() -> None:
    """Idempotent: safe to call before every `robosuite.make(...)`."""
    global _PATCHED
    if _PATCHED:
        return

    from robosuite.controllers.parts.controller import Controller
    from robosuite.controllers.parts.generic import joint_vel
    from robosuite.utils.buffers import RingBuffer

    JV = joint_vel.JointVelocityController

    def __init__(
        self,
        sim,
        joint_indexes,
        actuator_range,
        ref_name=None,
        input_max=1,
        input_min=-1,
        output_max=1,
        output_min=-1,
        kp=0.25,
        policy_freq=20,
        lite_physics=True,
        velocity_limits=None,
        interpolator=None,
        **kwargs,
    ):
        use_torque_compensation = kwargs.pop("use_torque_compensation", True)

        super(JV, self).__init__(
            sim,
            ref_name=ref_name,
            joint_indexes=joint_indexes,
            actuator_range=actuator_range,
            part_name=kwargs.get("part_name", None),
            naming_prefix=kwargs.get("naming_prefix", None),
            lite_physics=lite_physics,
        )
        self.control_dim = len(joint_indexes["joints"])

        self.input_max = self.nums2array(input_max, self.joint_dim)
        self.input_min = self.nums2array(input_min, self.joint_dim)
        self.output_max = self.nums2array(output_max, self.joint_dim)
        self.output_min = self.nums2array(output_min, self.joint_dim)

        self.kp = self.nums2array(kp, self.joint_dim)
        if type(kp) is float or type(kp) is int:
            low, high = self.actuator_limits
            self.kp = kp * (high - low)

        self.ki = self.kp * 0.005
        self.kd = self.kp * 0.001
        self.last_err = np.zeros(self.joint_dim)
        self.derr_buf = RingBuffer(dim=self.joint_dim, length=5)
        self.summed_err = np.zeros(self.joint_dim)
        self.saturated = False
        self.last_joint_vel = np.zeros(self.joint_dim)

        self.velocity_limits = np.array(velocity_limits) if velocity_limits is not None else None

        self.control_freq = policy_freq

        self.interpolator = interpolator

        self.goal_vel = None
        self.current_vel = np.zeros(self.joint_dim)
        self.torques = None

        self._use_torque_compensation = bool(use_torque_compensation)

    def run_controller(self):
        if self.goal_vel is None:
            self.set_goal(np.zeros(self.joint_dim))

        self.update()

        if self.interpolator is not None:
            if self.interpolator.order == 1:
                self.current_vel = self.interpolator.get_interpolated_goal()
        else:
            self.current_vel = np.array(self.goal_vel)

        err = self.current_vel - self.joint_vel
        derr = err - self.last_err
        self.last_err = err
        self.derr_buf.push(derr)

        if not self.saturated:
            self.summed_err += err

        pid = self.kp * err + self.ki * self.summed_err + self.kd * self.derr_buf.average
        if self._use_torque_compensation:
            gravity = Controller.torque_compensation.fget(self)
            torques = pid + gravity
        else:
            torques = pid

        self.torques = self.clip_torques(torques)

        self.saturated = False if np.sum(np.abs(self.torques - torques)) == 0 else True

        super(JV, self).run_controller()

        return self.torques

    JV.__init__ = __init__
    JV.run_controller = run_controller
    _PATCHED = True
