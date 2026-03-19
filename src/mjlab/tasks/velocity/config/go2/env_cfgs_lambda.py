"""Unitree Go2 velocity environment configurations for lambdalab reproduction."""

from mjlab.asset_zoo.robots.unitree_go2 import go2_constants_lambda
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, ObjRef
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg


def unitree_go2_flat_env_cfg_lambda(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree Go2 flat terrain velocity configuration matching lambdalab.

  This configuration replicates the exact parameters used in lambdalab training,
  including PD gains, action scaling, sensor naming, and termination conditions.
  """
  cfg = make_velocity_env_cfg()

  # Increase MuJoCo contact constraint limits for Go2 (lambdalab values)
  cfg.sim.njmax = 300  # Max number of constraints per world
  cfg.sim.nconmax = 35  # Max number of contacts to allocate per world
  # Disable NATIVECCD to allow non-zero margin (lambdalab uses margin=0.001)
  cfg.sim.mujoco.disableflags = ("nativeccd",)

  # Set Go2 robot using lambdalab parameters
  cfg.scene.entities = {"robot": go2_constants_lambda.get_go2_robot_cfg()}

  # Define foot geom names (as used in lambdalab's go2.xml)
  foot_geom_names = ("FR", "FL", "RR", "RL")
  site_names = ("FR", "FL", "RR", "RL")

  # Configure foot sensors for Go2 (geom-based contact sensor like lambdalab)
  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="geom", pattern=foot_geom_names, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  # Detect illegal contacts on non-foot bodies (base, hip, thigh only)
  nonfoot_ground_cfg = ContactSensorCfg(
    name="nonfoot_ground_touch",
    primary=ContactMatch(
      mode="body",
      entity="robot",
      # Only check base, hip, and thigh (NOT calf!) per lambdalab
      pattern=("base_link", ".*_hip", ".*_thigh"),
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )
  # Replace all sensors with Go2-specific contact sensors (as in lambdalab)
  cfg.scene.sensors = (feet_ground_cfg, nonfoot_ground_cfg)
  # Remove height scan observation (no terrain scanning in lambdalab)
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]

  # Configure action scale for Go2 (use computed scale from lambdalab)
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = go2_constants_lambda.GO2_ACTION_SCALE

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

  # Configure event terms with Go2-specific geom names (lambdalab uses geom names directly)
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = foot_geom_names

  # Set pose reward std for Go2 (lambdalab values)
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

  # Configure body-based reward terms (align with lambdalab)
  cfg.rewards["upright"].params["asset_cfg"].body_names = ("base_link",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("base_link",)

  # Set reward weights (align with lambdalab)
  cfg.rewards["body_ang_vel"].weight = 0.0
  cfg.rewards["angular_momentum"].weight = 0.0
  cfg.rewards["air_time"].weight = 0.0

  # Configure terrain curriculum (disabled for flat terrain)
  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  # Enable illegal_contact termination (lambdalab style, no force_threshold)
  cfg.terminations["illegal_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": nonfoot_ground_cfg.name},
  )

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
    # In mjlab, the policy observation group is keyed as "actor" (not "policy")
    if "actor" in cfg.observations:
      cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)

  return cfg
