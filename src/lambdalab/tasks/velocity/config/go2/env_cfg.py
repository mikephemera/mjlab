"""Unitree Go2 velocity environment configurations."""

from lambdalab.envs import ManagerBasedRlEnvCfg
from lambdalab.envs.mdp.actions import JointPositionActionCfg
from lambdalab.managers.manager_term_config import TerminationTermCfg
from lambdalab.managers.scene_entity_config import SceneEntityCfg
from lambdalab.sensor import ContactMatch, ContactSensorCfg
from lambdalab.asset_zoo.robots import get_go2_robot_cfg, GO2_ACTION_SCALE
from lambdalab.tasks.velocity import mdp
from lambdalab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg


def go2_velocity_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree Go2 rough terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  # Increase MuJoCo contact constraint limits for Go2
  # Go2 has complex geometry and generates many contacts, especially with terrain
  cfg.sim.njmax = 300  # Max number of constraints per world (default 300 is too small)
  cfg.sim.nconmax = 35  # Max number of contacts to allocate per world

  # Set Go2 robot
  cfg.scene.entities = {"robot": get_go2_robot_cfg()}

  # Define foot site names (now available in go2.xml)
  site_names = ("FR", "FL", "RR", "RL")
  # Define foot geom names for Go2
  foot_geom_names = ("FR", "FL", "RR", "RL")

  # Configure foot sensors for Go2
  # Use geom-based contact sensor (like Go1) for better accuracy
  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="geom", pattern=foot_geom_names, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  # Detect illegal contacts on non-foot bodies
  # For Go2, check base, hip, and thigh bodies only
  # Do NOT include calf, as calf body contains foot geoms and will cause false positives
  nonfoot_ground_cfg = ContactSensorCfg(
    name="nonfoot_ground_touch",
    primary=ContactMatch(
      mode="body",
      entity="robot",
      # Only check base, hip, and thigh (NOT calf!)
      pattern=("base_link", ".*_hip", ".*_thigh"),
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )
  cfg.scene.sensors = (feet_ground_cfg, nonfoot_ground_cfg)

  # Configure action scale for Go2 (use computed scale like Go1)
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = GO2_ACTION_SCALE

  # Set viewer configuration
  cfg.viewer.body_name = "base_link"
  cfg.viewer.distance = 1.5
  cfg.viewer.elevation = -10.0

  # Configure foot_height observation with foot sites
  cfg.observations["critic"].terms["foot_height"].params[
    "asset_cfg"
  ].site_names = site_names

  # Configure site-based reward terms with foot sites
  for reward_name in ["foot_clearance", "foot_swing_height", "foot_slip"]:
    if reward_name in cfg.rewards:
      cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

  # Configure event terms with Go2-specific geom names
  # Note: Go2 uses geom names like "FR", "FL", etc. (not "*_foot_collision")
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = ("FR", "FL", "RR", "RL")

  # Set pose reward std for Go2
  cfg.rewards["pose"].params["std_standing"] = {
    r".*(FL|FR|RL|RR)_(hip|thigh)_joint.*": 0.05,
    r".*(FL|FR|RL|RR)_calf_joint.*": 0.1,
  }
  cfg.rewards["pose"].params["std_walking"] = {
    r".*(FL|FR|RL|RR)_(hip|thigh)_joint.*": 0.3,
    r".*(FL|FR|RL|RR)_calf_joint.*": 0.6,
  }
  cfg.rewards["pose"].params["std_running"] = {
    r".*(FL|FR|RL|RR)_(hip|thigh)_joint.*": 0.3,
    r".*(FL|FR|RL|RR)_calf_joint.*": 0.6,
  }

  # Configure body-based reward terms (align with Go1)
  cfg.rewards["upright"].params["asset_cfg"].body_names = ("base_link",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("base_link",)

  # Set reward weights (align with Go1)
  cfg.rewards["body_ang_vel"].weight = 0.0
  cfg.rewards["angular_momentum"].weight = 0.0
  cfg.rewards["air_time"].weight = 0.0

  # Configure terrain curriculum for progressive difficulty
  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  # Enable illegal_contact termination (aligned with Go1)
  cfg.terminations["illegal_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": nonfoot_ground_cfg.name},
  )

  # Apply play mode overrides
  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["policy"].enable_corruption = False
    cfg.events.pop("push_robot", None)

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def go2_velocity_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree Go2 flat terrain velocity configuration."""
  cfg = go2_velocity_rough_env_cfg(play=False)  # Don't pass play to rough cfg

  # Switch to flat terrain
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Disable terrain curriculum
  if cfg.curriculum is not None and "terrain_levels" in cfg.curriculum:
    del cfg.curriculum["terrain_levels"]

  # Apply play mode overrides for flat terrain
  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["policy"].enable_corruption = False
    cfg.events.pop("push_robot", None)

  return cfg
