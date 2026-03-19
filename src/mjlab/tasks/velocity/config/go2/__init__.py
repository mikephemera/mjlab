from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import unitree_go2_flat_env_cfg
from .rl_cfg import unitree_go2_ppo_runner_cfg
from .env_cfgs_lambda import unitree_go2_flat_env_cfg_lambda
from .rl_cfg_lambda import unitree_go2_ppo_runner_cfg_lambda

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Unitree-Go2",
  env_cfg=unitree_go2_flat_env_cfg(),
  play_env_cfg=unitree_go2_flat_env_cfg(play=True),
  rl_cfg=unitree_go2_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Unitree-Go2-Lambda",
  env_cfg=unitree_go2_flat_env_cfg_lambda(),
  play_env_cfg=unitree_go2_flat_env_cfg_lambda(play=True),
  rl_cfg=unitree_go2_ppo_runner_cfg_lambda(),
  runner_cls=VelocityOnPolicyRunner,
)
