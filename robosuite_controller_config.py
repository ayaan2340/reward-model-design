"""
Build robosuite composite controller configs for Panda + JOINT_VELOCITY the way the docs describe:
start from shipped defaults, then override.

- Full BASIC JSON (type BASIC + body_parts): passed through `load_composite_controller_config` like the tutorial.
- Arm-only overrides JSON: merged onto `load_part_controller_config(default_controller="JOINT_VELOCITY")`
  plus Panda gripper, then swapped into `load_composite_controller_config(robot="Panda")`.

See https://robosuite.ai/docs/modules/controllers.html (Loading a Controller, Controller Settings).
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

_RD = Path(__file__).resolve().parent
if str(_RD) not in sys.path:
    sys.path.insert(0, str(_RD))

from robosuite_patches.joint_velocity_controller import apply_joint_velocity_controller_fix


def load_pi05_panda_composite_config(config_path: Path) -> dict[str, Any]:
    """
    Returns a dict suitable for `robosuite.make(..., controller_configs=...)`.

    Applies `apply_joint_velocity_controller_fix()` first (robosuite JOINT_VELOCITY bugfix).
    """
    apply_joint_velocity_controller_fix()

    if not config_path.is_file():
        raise FileNotFoundError(f"Controller config not found: {config_path}")

    with open(config_path) as f:
        data = json.load(f)

    if isinstance(data, dict) and data.get("type") == "BASIC" and "body_parts" in data:
        from robosuite import load_composite_controller_config

        return load_composite_controller_config(controller=str(config_path.resolve()))

    return _panda_joint_velocity_from_arm_overrides(data)


def _panda_joint_velocity_from_arm_overrides(arm_overrides: dict[str, Any]) -> dict[str, Any]:
    from robosuite import load_composite_controller_config
    from robosuite.controllers import load_part_controller_config

    composite = load_composite_controller_config(robot="Panda")
    arm = copy.deepcopy(load_part_controller_config(default_controller="JOINT_VELOCITY"))
    arm["gripper"] = {"type": "GRIP"}
    arm.update(arm_overrides)
    composite["body_parts"]["right"] = arm
    return composite
