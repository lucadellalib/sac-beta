# ==============================================================================
# Copyright 2024 Luca Della Libera.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Environment definition."""

# Adapted from:
# https://github.com/thu-ml/tianshou/blob/v0.5.0/examples/mujoco/mujoco_env.py

import warnings

import gymnasium as gym
from tianshou.env import ShmemVectorEnv, VectorEnvNormObs


try:
    import envpool
except ImportError:
    envpool = None


__all__ = ["make_mujoco_env"]


def make_mujoco_env(task, seed, training_num, test_num, obs_norm):
    """Wrapper function for MuJoCo env.

    If EnvPool is installed, it will automatically switch to EnvPool's Mujoco env.

    :return: a tuple of (single env, training envs, test envs).
    """
    if envpool is not None:
        train_envs = env = envpool.make_gymnasium(
            task, num_envs=training_num, seed=seed
        )
        test_envs = envpool.make_gymnasium(task, num_envs=test_num, seed=seed)
    else:
        warnings.warn(
            "Recommend using envpool (pip install envpool) "
            "to run MuJoCo environments more efficiently."
        )
        env = gym.make(task)
        train_envs = ShmemVectorEnv(
            [lambda: gym.make(task) for _ in range(training_num)]
        )
        test_envs = ShmemVectorEnv([lambda: gym.make(task) for _ in range(test_num)])
        # env.seed(seed)
        train_envs.seed(seed)
        test_envs.seed(seed)
    if obs_norm:
        # obs norm wrapper
        train_envs = VectorEnvNormObs(train_envs)
        test_envs = VectorEnvNormObs(test_envs, update_obs_rms=False)
        test_envs.set_obs_rms(train_envs.get_obs_rms())
    return env, train_envs, test_envs
