"""Go2 constants and entity configuration."""

from pathlib import Path

import mujoco

from lambdalab import LAMBDALAB_SRC_PATH
from lambdalab.actuator import XmlPositionActuatorCfg
from lambdalab.entity.entity import EntityArticulationInfoCfg, EntityCfg
from lambdalab.utils.actuator import ElectricActuator, reflected_inertia


##
# MJCF and assets.
##

GO2_XML: Path = (
  LAMBDALAB_SRC_PATH / "asset_zoo" / "robots" / "unitree_go2" / "xmls" / "go2.xml"
)
assert GO2_XML.exists()


def get_spec() -> mujoco.MjSpec:
  """Load Go2 MJCF as an MjSpec."""
  spec = mujoco.MjSpec.from_file(str(GO2_XML))
  return spec


##
# Actuator config (matching Go1 style for consistency)
##

# Rotor inertia - assume similar to Go1
ROTOR_INERTIA = 0.000111842

# Gearbox ratios (Go2 similar to Go1)
HIP_GEAR_RATIO = 6
KNEE_GEAR_RATIO = HIP_GEAR_RATIO * 1.5

HIP_ACTUATOR = ElectricActuator(
  reflected_inertia=reflected_inertia(ROTOR_INERTIA, HIP_GEAR_RATIO),
  velocity_limit=30.1,
  effort_limit=23.7,  # From XML: ctrlrange="-23.7 23.7"
)
KNEE_ACTUATOR = ElectricActuator(
  reflected_inertia=reflected_inertia(ROTOR_INERTIA, KNEE_GEAR_RATIO),
  velocity_limit=20.06,
  effort_limit=45.43,  # From XML: ctrlrange="-45.43 45.43"
)

# PD parameters defined in XML (kp and kv)
# Hip/Thigh: kp=4.44, kv=0.28
# Knee: kp=6.66, kv=0.42
STIFFNESS_HIP = 4.44
DAMPING_HIP = 0.28
STIFFNESS_KNEE = 6.66
DAMPING_KNEE = 0.42

# Use XML-defined position actuators with PD control
GO2_ACTUATOR_CFG = XmlPositionActuatorCfg(
  target_names_expr=(".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"),
)

GO2_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(GO2_ACTUATOR_CFG,),
)


##
# Action scaling (matching Go1 approach)
##

# Calculate action scale: 0.25 * effort_limit / stiffness
GO2_ACTION_SCALE: dict[str, float] = {
  ".*_hip_joint": 0.25 * HIP_ACTUATOR.effort_limit / STIFFNESS_HIP,
  ".*_thigh_joint": 0.25 * HIP_ACTUATOR.effort_limit / STIFFNESS_HIP,
  ".*_calf_joint": 0.25 * KNEE_ACTUATOR.effort_limit / STIFFNESS_KNEE,
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
      pos=(0.0, 0.0, 0.278),  # Increased to 0.278 to match Go1
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
