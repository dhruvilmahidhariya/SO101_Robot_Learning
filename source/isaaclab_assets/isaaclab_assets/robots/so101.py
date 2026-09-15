# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the LeRobot SO-101 arm (6-DOF + gripper)."""

import math
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

# Robot mesh/articulation only — camera frame is NOT baked into this USD.
SO101_USD_PATH = str(Path(__file__).resolve().parents[4] / "so101_new_calib.usd")

# PhysX Inspector limits for gripper: close=-10 deg, open=100 deg.
SO101_GRIPPER_OPEN_RAD = math.radians(95.0)
SO101_GRIPPER_CLOSE_RAD = math.radians(-10.0)


@configclass
class So101WristCameraCalibCfg:
    """Runtime wrist-camera calibration (replace with your own hand-eye / intrinsics).

    Extrinsics are ``parent_link`` → camera optical frame (ROS/OpenCV: x-right,
    y-down, z-forward). Intrinsics are the 3x3 camera matrix in row-major form
    ``[fx, 0, cx, 0, fy, cy, 0, 0, 1]`` (same layout as OpenCV / ROS CameraInfo).
    """

    # Link the camera is rigidly mounted on (must exist in the USD).
    parent_link: str = "wrist_link"
    # Prim name created under parent_link at spawn time.
    prim_name: str = "wrist_cam"

    # Hand-eye: xyz (m) and quaternion wxyz for optical frame in parent.
    # Final TF: wrist_link -> camera_optical (static_transform_publisher / so101_camera.xacro)
    pos: tuple[float, float, float] = (-0.050360, -0.045513, 0.018273)
    # From rpy xyz (rad) = (1.090172, 1.532300, -0.331997)  [URDF fixed XYZ]
    rot_wxyz: tuple[float, float, float, float] = (
        0.5482868223698861,
        0.4664484448691112,
        0.52299998823884,
        -0.4563753071726101,
    )
    convention: str = "ros"

    width: int = 640
    height: int = 480
    # From Docs/calibration_patterns/intrinsics/innomaker_640x480.yaml (physical board)
    intrinsic_matrix: tuple[float, ...] = (
        550.821730,
        0.0,
        340.207457,
        0.0,
        550.402808,
        263.941152,
        0.0,
        0.0,
        1.0,
    )
    clipping_range: tuple[float, float] = (0.01, 2.0)


# Default calib shipped for this workspace's Innomaker mount. Copy/edit for yours.
SO101_WRIST_CAMERA_CALIB = So101WristCameraCalibCfg()


def make_so101_wrist_camera_cfg(
    robot_prim_path: str = "{ENV_REGEX_NS}/Robot",
    usd_root_prim: str = "so101_new_calib",
    calib: So101WristCameraCalibCfg | None = None,
) -> CameraCfg:
    """Build a :class:`CameraCfg` from calibration (spawns under the parent link)."""
    calib = calib if calib is not None else SO101_WRIST_CAMERA_CALIB
    prim_path = f"{robot_prim_path}/{usd_root_prim}/{calib.parent_link}/{calib.prim_name}"
    return CameraCfg(
        prim_path=prim_path,
        update_period=0.0,
        height=calib.height,
        width=calib.width,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
            intrinsic_matrix=list(calib.intrinsic_matrix),
            width=calib.width,
            height=calib.height,
            clipping_range=calib.clipping_range,
        ),
        offset=CameraCfg.OffsetCfg(
            pos=calib.pos,
            rot=calib.rot_wxyz,
            convention=calib.convention,
        ),
    )


SO101_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=SO101_USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=1,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "shoulder_pan": 0.0,
            "shoulder_lift": 0.0,
            "elbow_flex": math.radians(-10.0),
            "elbow_rotate": 0.0,
            "wrist_flex": math.radians(60.0),
            "wrist_roll": math.radians(90.0),
            "gripper": SO101_GRIPPER_OPEN_RAD,
        },
    ),
    actuators={
        # Only shiver fix: softer PD + lower vel/effort than Franka defaults.
        "arm": ImplicitActuatorCfg(
            joint_names_expr=[
                "shoulder_pan",
                "shoulder_lift",
                "elbow_flex",
                "elbow_rotate",
                "wrist_flex",
                "wrist_roll",
            ],
            effort_limit_sim=5.0,
            velocity_limit_sim=3.0,
            stiffness=40.0,
            damping=12.0,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["gripper"],
            effort_limit_sim=2.0,
            velocity_limit_sim=2.0,
            stiffness=200.0,
            damping=40.0,
        ),
    },
    soft_joint_pos_limit_factor=0.95,
)
"""Configuration of LeRobot SO-101 unimanual arm."""
