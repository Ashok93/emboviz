"""Robot catalog — resolve a robot to a :class:`RobotKinematics`.

Two entry points, one flow:

  * **Preconfigured** common robots: the caller passes a catalog name
    (``"franka_panda"``). The official URDF is resolved through
    ``robot_descriptions`` (cloned and cached on first use); the end-effector
    frame and controlled-joint list are fixed here.
  * **Custom / uncommon** robots: the caller passes a URDF path plus the
    end-effector frame and joint names. Same ``RobotKinematics``, same ``fk``.

Adding a preconfigured robot is one ``_CatalogEntry`` — the ``robot_descriptions``
description name, the end-effector frame, and the controlled joints. Frame and
joint names must be verified against the shipped URDF, never guessed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from emboviz_robot.kinematics import RobotKinematics


@dataclass(frozen=True)
class _CatalogEntry:
    description: str          # robot_descriptions description module name
    ee_frame: str            # end-effector frame in that URDF
    joint_names: list[str]   # controlled joints, in policy order


# Verified against the URDF each ``robot_descriptions`` description ships.
#
# franka_panda: example-robot-data ``panda.urdf`` exposes panda_joint1..7
# (revolute) and the flange frame ``panda_link8`` (child of the fixed
# ``panda_joint8``). DROID logs ``cartesian_position`` as the panda_link8 pose
# in the panda_link0 base (Polymetis ``ee_link_name: panda_link8``), so this is
# the frame a DROID-trained policy's Cartesian conditioning refers to.
_CATALOG: dict[str, _CatalogEntry] = {
    "franka_panda": _CatalogEntry(
        description="panda_description",
        ee_frame="panda_link8",
        joint_names=[f"panda_joint{i}" for i in range(1, 8)],
    ),
}

#: Convenience aliases for catalog names.
_ALIASES: dict[str, str] = {
    "panda": "franka_panda",
    "franka": "franka_panda",
}


def available_robots() -> list[str]:
    """Names accepted by :func:`load_kinematics` (catalog + aliases)."""
    return sorted(set(_CATALOG) | set(_ALIASES))


def _resolve_catalog_urdf(description: str) -> str:
    """Return the cached URDF path for a ``robot_descriptions`` description."""
    from importlib import import_module

    module = import_module(f"robot_descriptions.{description}")
    urdf_path = getattr(module, "URDF_PATH", None)
    if urdf_path is None:
        raise ValueError(
            f"robot_descriptions.{description} has no URDF_PATH (it may be a "
            "MuJoCo-only description). Use a URDF-backed description."
        )
    return str(urdf_path)


def load_kinematics(
    robot: Optional[str] = None,
    *,
    urdf: Optional[str] = None,
    ee_frame: Optional[str] = None,
    joint_names: Optional[list[str]] = None,
) -> RobotKinematics:
    """Build a :class:`RobotKinematics` for a preconfigured or custom robot.

    Preconfigured::

        load_kinematics("franka_panda")

    Custom (supply all three)::

        load_kinematics(urdf="/path/arm.urdf", ee_frame="tool0",
                        joint_names=["j1", ..., "j6"])

    Exactly one of ``robot`` or the ``urdf``/``ee_frame``/``joint_names`` triple
    must be given — mixing them is rejected rather than silently resolved.
    """
    custom = (urdf, ee_frame, joint_names)
    has_custom = any(x is not None for x in custom)

    if robot is not None:
        if has_custom:
            raise ValueError(
                "load_kinematics: pass EITHER a catalog name OR a custom "
                "urdf/ee_frame/joint_names triple, not both."
            )
        key = _ALIASES.get(robot, robot)
        entry = _CATALOG.get(key)
        if entry is None:
            raise ValueError(
                f"load_kinematics: unknown robot {robot!r}. Preconfigured: "
                f"{available_robots()}. For a robot not in the catalog, pass "
                "urdf + ee_frame + joint_names."
            )
        return RobotKinematics(
            _resolve_catalog_urdf(entry.description), entry.ee_frame, entry.joint_names
        )

    if not has_custom:
        raise ValueError(
            "load_kinematics: pass a catalog name (one of "
            f"{available_robots()}) or a custom urdf + ee_frame + joint_names."
        )
    if urdf is None or ee_frame is None or not joint_names:
        raise ValueError(
            "load_kinematics: a custom robot needs all of urdf, ee_frame, and "
            f"joint_names; got urdf={urdf!r}, ee_frame={ee_frame!r}, "
            f"joint_names={joint_names!r}."
        )
    return RobotKinematics(urdf, ee_frame, list(joint_names))


__all__ = ["available_robots", "load_kinematics"]
