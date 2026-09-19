"""Frozen eval of the SO101 world-model checkpoint (no training).

Compares next-step predictions vs the identity baseline (Δ = 0).
Optionally runs a short recursive rollout.

Example:
  ./isaaclab.sh -p self_model/eval_world_model.py \\
    --checkpoint self_model/checkpoints/world_model.pt \\
    --num_envs 64 --headless --max_ctrl_steps 500
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Eval frozen SO101 world model")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--goal_hold_ticks", type=int, default=8)
parser.add_argument("--max_ctrl_steps", type=int, default=500, help="Eval control ticks (10 Hz).")
parser.add_argument(
    "--checkpoint",
    type=str,
    default="self_model/checkpoints/world_model.pt",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import sys
from pathlib import Path

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab_assets.robots.so101 import SO101_CFG

sys.path.insert(0, str(Path(__file__).resolve().parent))
from network.nn.mlp import WorldModel

EEF_X_MAX, EEF_Y_MAX = 0.35, 0.35
EEF_Z_MIN, EEF_Z_MAX = 0.01, 0.35
CONTROL_HZ = 10.0
ARM_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "elbow_rotate",
    "wrist_flex",
    "wrist_roll",
]


@configclass
class So101SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )
    robot = SO101_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def run_eval(sim: SimulationContext, scene: InteractiveScene):
    robot = scene["robot"]
    device = robot.device

    ckpt_path = Path(args_cli.checkpoint)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = WorldModel().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[INFO] : loaded {ckpt_path} (trained ctrl_count={ckpt.get('ctrl_count', '?')})")

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
    hold_steps = max(1, int(round((1.0 / CONTROL_HZ) / sim_dt)))
    goal_hold_ticks = max(1, args_cli.goal_hold_ticks)

    action = None
    hold_left = 0
    ik_cmd = None
    goal_left = 0
    q_ctrl = qd_ctrl = None
    count = 0
    ctrl_count = 0

    # running sums for mean metrics
    sum_pos = sum_id_pos = sum_vel = sum_id_vel = 0.0

    print(
        f"[INFO] : EVAL control={CONTROL_HZ} Hz, hold_steps={hold_steps}, "
        f"goal_hold_ticks={goal_hold_ticks}, num_envs={scene.num_envs}"
    )

    with torch.no_grad():
        while simulation_app.is_running():
            if count % 500 == 0:
                count = 0
                root_state = robot.data.default_root_state.clone()
                root_state[:, :3] += scene.env_origins
                robot.write_root_pose_to_sim(root_state[:, :7])
                robot.write_root_velocity_to_sim(root_state[:, 7:])
                robot.write_joint_state_to_sim(
                    robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
                )
                scene.reset()
                diff_ik.reset()
                action = None
                hold_left = 0
                ik_cmd = None
                goal_left = 0
                q_ctrl = qd_ctrl = None

            if action is None or hold_left <= 0:
                if q_ctrl is not None:
                    q2 = robot.data.joint_pos[:, arm_ids]
                    qd2 = robot.data.joint_vel[:, arm_ids]
                    dq, dqd = q2 - q_ctrl, qd2 - qd_ctrl

                    pred = model(torch.cat([q_ctrl, qd_ctrl, action], dim=-1))
                    dq_hat, dqd_hat = pred[:, :6], pred[:, 6:]

                    pos_mae = (dq_hat - dq).abs().mean().item()
                    vel_mae = (dqd_hat - dqd).abs().mean().item()
                    id_pos = dq.abs().mean().item()
                    id_vel = dqd.abs().mean().item()
                    sum_pos += pos_mae
                    sum_id_pos += id_pos
                    sum_vel += vel_mae
                    sum_id_vel += id_vel
                    ctrl_count += 1

                    if ctrl_count % 50 == 0:
                        print(
                            f"[{ctrl_count}/{args_cli.max_ctrl_steps}]  "
                            f"pos_mae={pos_mae:.5f} (id={id_pos:.5f})  "
                            f"vel_mae={vel_mae:.5f} (id={id_vel:.5f})"
                        )

                    if ctrl_count >= args_cli.max_ctrl_steps:
                        n = float(ctrl_count)
                        print("\n========== FROZEN EVAL SUMMARY ==========")
                        print(f"steps={ctrl_count}  envs={scene.num_envs}")
                        print(
                            f"mean pos_mae={sum_pos / n:.5f}  id={sum_id_pos / n:.5f}  "
                            f"ratio={sum_id_pos / max(sum_pos, 1e-12):.2f}x better than id"
                        )
                        print(
                            f"mean vel_mae={sum_vel / n:.5f}  id={sum_id_vel / n:.5f}  "
                            f"ratio={sum_id_vel / max(sum_vel, 1e-12):.2f}x better than id"
                        )
                        beat = (sum_pos / n) < (sum_id_pos / n)
                        print(f"PASS (beat identity on pos): {beat}")
                        print("=========================================\n")
                        return

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

                diff_ik.set_command(ik_cmd)
                jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, arm_ids]
                joint_pos = robot.data.joint_pos[:, arm_ids]
                action = diff_ik.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
                action = action.clamp(
                    robot.data.soft_joint_pos_limits[:, arm_ids, 0],
                    robot.data.soft_joint_pos_limits[:, arm_ids, 1],
                )

                q_ctrl = robot.data.joint_pos[:, arm_ids].clone()
                qd_ctrl = robot.data.joint_vel[:, arm_ids].clone()
                hold_left = hold_steps
                goal_left -= 1

            full_target = robot.data.default_joint_pos.clone()
            full_target[:, arm_ids] = action
            robot.set_joint_position_target(full_target)
            scene.write_data_to_sim()
            sim.step()
            count += 1
            scene.update(sim_dt)
            hold_left -= 1


def main():
    sim = SimulationContext(sim_utils.SimulationCfg(dt=0.01, device=args_cli.device))
    if not args_cli.headless:
        sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])
    scene = InteractiveScene(So101SceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0))
    sim.reset()
    print("[INFO] : setup complete")
    run_eval(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
