from lambdalab.tasks.registry import register_lambdalab_task
from lambdalab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfg import (
  go2_velocity_flat_env_cfg,
  go2_velocity_rough_env_cfg,
  )
from .rl_cfg import go2_ppo_cfg

register_lambdalab_task(
  task_id="Lambdalab-Velocity-Flat-Unitree-Go2",
  env_cfg=go2_velocity_flat_env_cfg(),
  play_env_cfg=go2_velocity_flat_env_cfg(play=True),
  rl_cfg=go2_ppo_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_lambdalab_task(
  task_id="Lambdalab-Velocity-Rough-Unitree-Go2",
  env_cfg=go2_velocity_rough_env_cfg(),
  play_env_cfg=go2_velocity_rough_env_cfg(play=True),
  rl_cfg=go2_ppo_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
