'''Launch IsaacSim'''
import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(description= "Spawn SO101 6DOF robot arm")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
AppLauncher.add_app_launcher_args(parser)
#parse the args
args_cli = parser.parse_args()
#launch omniverse
app_launcher=AppLauncher(args_cli)
simulation_app=app_launcher.app

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass

# same import pattern as the working lift env
from isaaclab_assets.robots.so101 import SO101_CFG


@configclass
class So101SceneCfg(InteractiveSceneCfg):
    """Minimal scene — same robot spawn as Isaac-Lift-Cube-SO101-v0."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )
    # identical to lift: SO101_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot = SO101_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    robot = scene["robot"]
    sim_dt = sim.get_physics_dt()
    count = 0
    while simulation_app.is_running():
        if count % 500 == 0:
            count = 0
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])
            joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
            joint_pos += torch.rand_like(joint_pos) * 0.1
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            scene.reset()
            print(f"[INFO] : Resetting the robot state")
        # SO101 uses ImplicitActuatorCfg with stiffness>0 → position PD, not effort (cartpole tutorial).
        targets = robot.data.default_joint_pos.clone()
        targets += (torch.rand_like(targets) - 0.5) * 0.2
        robot.set_joint_position_target(targets)
        scene.write_data_to_sim()
        sim.step()
        count += 1
        scene.update(sim_dt)
        if count % 50 == 0:
            print(f"q={robot.data.joint_pos[0, :6].cpu().numpy()}")


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])
    scene = InteractiveScene(So101SceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0))
    sim.reset()
    print(f"[INFO] : setup complete")
    run_simulator(sim, scene)

if __name__ == "__main__":

    main()
    simulation_app.close()
