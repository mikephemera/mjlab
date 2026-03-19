"""Register all direct (MuJoCo-based) tasks for isaaclab_slim.

Importing this module triggers the Gym registrations defined in each
isaaclab_slim.tasks.direct.<task> package.
"""

# Import direct-task packages so that their __init__.py runs gym.register.
import lambdalab.tasks.direct.cartpole  # 注册 cartpole 任务
import lambdalab.tasks.direct.go2       # 注册 Go2 任务
import lambdalab.tasks.direct.g1