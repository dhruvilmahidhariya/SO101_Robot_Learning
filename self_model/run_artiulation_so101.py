"""Launch IsaacSim"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Spawn SO101 6DOF robot arm")
parser.add_argument(
    "--num_envs", type=int, default=1, help="Number of environments to spawn."
)
parser.add_argument(
    "--goal_hold_ticks",
    type=int,
    default=8,
    help="How many 10 Hz ticks to keep the same EEF goal (iterative IK). 8 ≈ 0.8 s.",
)
parser.add_argument(
    "--max_ctrl_steps",
    type=int,
    default=0,
    help="Stop after this many 10 Hz world-model updates (0 = run until window closed).",
)
parser.add_argument(
    "--save_path",
    type=str,
    default="self_model/checkpoints/world_model.pt",
    help="Where to save the world-model weights on exit / periodically.",
)
AppLauncher.add_app_launcher_args(parser)
# parse the args
args_cli = parser.parse_args()
# launch omniverse
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms

# same import pattern as the working lift env
from isaaclab_assets.robots.so101 import SO101_CFG

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from network.nn.mlp import WorldModel

# EEF workspace (table at z=0): |x|,|y| ≤ 0.35 m, z ≥ 0.01 m
EEF_X_MAX = 0.35
EEF_Y_MAX = 0.35
EEF_Z_MIN = 0.01
EEF_Z_MAX = 0.35
CONTROL_HZ = 10.0


@configclass
class So101SceneCfg(InteractiveSceneCfg):
    """Minimal scene — same robot spawn as Isaac-Lift-Cube-SO101-v0."""

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg()
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )
    # identical to lift: SO101_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot = SO101_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    robot = scene["robot"]
    device = robot.device
    model = WorldModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    vel_weight = 0.1
    ARM_JOINT_NAMES = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "elbow_rotate",
        "wrist_flex",
        "wrist_roll",
    ]
    arm_ids, _ = robot.find_joints(ARM_JOINT_NAMES, preserve_order=True)
    ee_body_ids, _ = robot.find_bodies(["eef_frame_link"], preserve_order=True)
    ee_body_id = ee_body_ids[0]
    ee_jacobi_idx = ee_body_id - 1 if robot.is_fixed_base else ee_body_id

    diff_ik = DifferentialIKController(
        DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
        num_envs=scene.num_envs,
        device=device,
    )

    sim_dt = sim.get_physics_dt()
    hold_steps = max(1, int(round((1.0 / CONTROL_HZ) / sim_dt)))  # 10 @ dt=0.01
    goal_hold_ticks = max(1, args_cli.goal_hold_ticks)
    action = None
    hold_left = 0
    ik_cmd = None
    goal_left = 0
    q_ctrl = qd_ctrl = dq_hat = dqd_hat = None
    count = 0
    ctrl_count = 0
    print(
        f"[INFO] : control={CONTROL_HZ} Hz, hold_steps={hold_steps}, "
        f"goal_hold_ticks={goal_hold_ticks}, sim_dt={sim_dt}, num_envs={scene.num_envs}"
    )

    while simulation_app.is_running():
        if count % 500 == 0:
            count = 0
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])
            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            scene.reset()
            diff_ik.reset()
            action = None
            hold_left = 0
            ik_cmd = None
            goal_left = 0
            q_ctrl = qd_ctrl = dq_hat = dqd_hat = None
            print(f"[INFO] : Resetting the robot state")

        # --- 10 Hz control tick ---
        if action is None or hold_left <= 0:
            if q_ctrl is not None:
                q2 = robot.data.joint_pos[:, arm_ids]
                qd2 = robot.data.joint_vel[:, arm_ids]
                dq, dqd = q2 - q_ctrl, qd2 - qd_ctrl
                loss = ((dq_hat - dq) ** 2).mean() + vel_weight * ((dqd_hat - dqd) ** 2).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                ctrl_count += 1
                if ctrl_count % 10 == 0:
                    with torch.no_grad():
                        ee_pos_w = robot.data.body_pose_w[:, ee_body_id, :3] - scene.env_origins
                        pos_mae = (dq_hat - dq).abs().mean().item()
                        vel_mae = (dqd_hat - dqd).abs().mean().item()
                        id_pos = dq.abs().mean().item()
                        id_vel = dqd.abs().mean().item()
                    print(
                        f"[{ctrl_count}] loss={loss.item():.5f}  "
                        f"pos_mae={pos_mae:.5f} (id={id_pos:.5f})  "
                        f"vel_mae={vel_mae:.5f} (id={id_vel:.5f})  "
                        f"ee={ee_pos_w[0].cpu().numpy()}"
                    )
                if args_cli.max_ctrl_steps > 0 and ctrl_count >= args_cli.max_ctrl_steps:
                    save_path = Path(args_cli.save_path)
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save({"model": model.state_dict(), "ctrl_count": ctrl_count}, save_path)
                    print(f"[INFO] : saved {save_path} after {ctrl_count} ctrl steps")
                    break

            # new Cartesian goal only when previous goal budget is spent
            root_pose_w = robot.data.root_pose_w
            ee_pose_w = robot.data.body_pose_w[:, ee_body_id]
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
            )
            if ik_cmd is None or goal_left <= 0:
                target_pos = torch.empty(scene.num_envs, 3, device=device)
                target_pos[:, 0] = (torch.rand(scene.num_envs, device=device) * 2.0 - 1.0) * EEF_X_MAX
                target_pos[:, 1] = (torch.rand(scene.num_envs, device=device) * 2.0 - 1.0) * EEF_Y_MAX
                target_pos[:, 2] = EEF_Z_MIN + torch.rand(scene.num_envs, device=device) * (
                    EEF_Z_MAX - EEF_Z_MIN
                )
                ik_cmd = torch.cat([target_pos, ee_quat_b], dim=-1)
                goal_left = goal_hold_ticks

            # iterative IK: same goal, re-solve from current pose each 10 Hz tick
            diff_ik.set_command(ik_cmd)
            jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, arm_ids]
            joint_pos = robot.data.joint_pos[:, arm_ids]
            action = diff_ik.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
            low = robot.data.soft_joint_pos_limits[:, arm_ids, 0]
            high = robot.data.soft_joint_pos_limits[:, arm_ids, 1]
            action = action.clamp(low, high)

            q_ctrl = robot.data.joint_pos[:, arm_ids].clone()
            qd_ctrl = robot.data.joint_vel[:, arm_ids].clone()
            pred = model(torch.cat([q_ctrl, qd_ctrl, action], dim=-1))
            dq_hat, dqd_hat = pred[:, :6], pred[:, 6:]
            hold_left = hold_steps
            goal_left -= 1

        # command arm only; gripper keeps default target
        full_target = robot.data.default_joint_pos.clone()
        full_target[:, arm_ids] = action
        robot.set_joint_position_target(full_target)

        scene.write_data_to_sim()
        sim.step()
        count += 1
        scene.update(sim_dt)
        hold_left -= 1


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)  # 100 Hz physics → 10 steps / control
    sim = SimulationContext(sim_cfg)
    if not args_cli.headless:
        sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])
    scene = InteractiveScene(So101SceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0))
    sim.reset()
    print(f"[INFO] : setup complete")
    run_simulator(sim, scene)


if __name__ == "__main__":

    main()
    simulation_app.close()
