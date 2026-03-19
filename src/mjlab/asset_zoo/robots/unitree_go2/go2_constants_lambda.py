"""Unitree Go2 constants for lambdalab configuration."""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import XmlPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets

##
# MJCF and assets.
##

GO2_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "unitree_go2_lambda" / "xmls" / "go2.xml"
)
assert GO2_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
  """Load assets for the Go2 robot."""
  assets: dict[str, bytes] = {}
  update_assets(assets, GO2_XML.parent / "assets", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  """Load Go2 MJCF as an MjSpec."""
  spec = mujoco.MjSpec.from_file(str(GO2_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


##
# Actuator config (matching lambdalab parameters)
##

# PD parameters defined in XML (kp and kv)
# Hip/Thigh: kp=4.44, kv=0.28
# Knee: kp=6.66, kv=0.42
STIFFNESS_HIP = 4.44
DAMPING_HIP = 0.28
STIFFNESS_KNEE = 6.66
DAMPING_KNEE = 0.42

# Effort limits from XML (ctrlrange)
HIP_EFFORT_LIMIT = 23.7
KNEE_EFFORT_LIMIT = 45.43

# Use XML-defined position actuators with PD control (parameters already in XML)
GO2_ACTUATOR_CFG = XmlPositionActuatorCfg(
  target_names_expr=(".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"),
)

GO2_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(GO2_ACTUATOR_CFG,),
)


##
# Action scaling (matching lambdalab approach)
##

# Calculate action scale: 0.25 * effort_limit / stiffness
GO2_ACTION_SCALE: dict[str, float] = {
  ".*_hip_joint": 0.25 * HIP_EFFORT_LIMIT / STIFFNESS_HIP,
  ".*_thigh_joint": 0.25 * HIP_EFFORT_LIMIT / STIFFNESS_HIP,
  ".*_calf_joint": 0.25 * KNEE_EFFORT_LIMIT / STIFFNESS_KNEE,
}


def get_go2_robot_cfg() -> EntityCfg:
  """Get Go2 robot configuration as an EntityCfg.

  This uses the "home" keyframe pose from the original Unitree MJCF, where
  hip joints are at 0, thigh joints at ~0.9 rad, and calf joints at ~-1.8 rad.
  """
  return EntityCfg(
    spec_fn=get_spec,
    articulation=GO2_ARTICULATION,
    init_state=EntityCfg.InitialStateCfg(
      # Root position & orientation: floating base standing above the ground.
      pos=(0.0, 0.0, 0.278),  # Increased to 0.278 to match Go1 (lambdalab value)
      rot=(1.0, 0.0, 0.0, 0.0),
      # Initialize all hip/thigh/calf joints to a comfortable standing pose.
      joint_pos={
        # Thigh joints.
        ".*_thigh_joint": 0.9,
        # Knee joints.
        ".*_calf_joint": -1.8,
        # Abduction (hip roll) joints - add slight outward angle for stability.
        "(FR|RR)_hip_joint": 0.1,  # Right side: outward
        "(FL|RL)_hip_joint": -0.1,  # Left side: inward
      },
      joint_vel={".*": 0.0},
    ),
  )
