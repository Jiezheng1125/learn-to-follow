import numpy as np
import gymnasium
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box, Dict

from follower.planning import ResettablePlanner, PlannerConfig


class PreprocessorConfig(PlannerConfig):
    network_input_radius: int = 5
    intrinsic_target_reward: float = 0.01


def follower_preprocessor(env, algo_config):
    env = wrap_preprocessors(env, algo_config.training_config.preprocessing)
    return env


def wrap_preprocessors(env, config: PreprocessorConfig, auto_reset=False):
    env = FollowerWrapper(env=env, config=config)
    env = CutObservationWrapper(env, target_observation_radius=config.network_input_radius)
    env = ConcatPositionalFeatures(env)
    if auto_reset:
        env = AutoResetWrapper(env)
    return env


class FollowerWrapper(ObservationWrapper):

    """
    基于内置规划器ResettablePlanner为每个智能体生成通往全局目标的“子目标路径”，并叠加到局部珊格
    若上一时刻子目标达成，则给予内在奖励
    """
    def __init__(self, env, config: PreprocessorConfig):
        super().__init__(env)
        self._cfg: PreprocessorConfig = config
        self.re_plan = ResettablePlanner(self._cfg)
        self.prev_goals = None
        self.intrinsic_reward = None

    @staticmethod
    def get_relative_xy(x, y, tx, ty, obs_radius):
        dx, dy = x - tx, y - ty
        if dx > obs_radius or dx < -obs_radius or dy > obs_radius or dy < -obs_radius:
            return None, None
        return obs_radius - dx, obs_radius - dy

    def observation(self, observations):
        """
        处理环境返回的观察值，为每个智能体生成路径规划信息，并更新观察值
        
        """

        # Update cost penalties based on the current observations, independently for each agent.
        self.re_plan.update(observations)

        # 获取每个智能体到目标点的最短路径。
        paths = self.re_plan.get_path()

        new_goals = []  # Initialize a list to store new goals for each agent.
        intrinsic_rewards = []  # Initialize a list to store intrinsic rewards for each agent.

        # Iterate through agents and their respective paths.
        for k, path in enumerate(paths):
            obs = observations[k]

            # Check if there is no valid path available.
            if path is None:
                new_goals.append(obs['target_xy'])  # Use the target position as a new goal.
                path = []
            else:
                # 检查智能体是否达成了上一时刻的子目标
                subgoal_achieved = self.prev_goals and obs['xy'] == self.prev_goals[k]
                # 如果达成，给予内在奖励（intrinsic_target_reward）；否则奖励为 0.0。
                intrinsic_rewards.append(self._cfg.intrinsic_target_reward if subgoal_achieved else 0.0)
                # 更新新的子目标为路径中的下一个点
                new_goals.append(path[1])

            # 将观察值中的障碍物值设置为负数，用于后续的处理
            obs['obstacles'][obs['obstacles'] > 0] *= -1

            # Adding path to the observation, setting path values to +1.0.
            r = obs['obstacles'].shape[0] // 2
            for idx, (gx, gy) in enumerate(path):
                x, y = self.get_relative_xy(*obs['xy'], gx, gy, r)
                if x is not None and y is not None:
                    obs['obstacles'][x, y] = 1.0
                else:
                    break
            # print(obs['obstacles'])
        # Update the previous goals and intrinsic rewards for the next step.
        self.prev_goals = new_goals
        self.intrinsic_reward = intrinsic_rewards

        return observations

    def get_intrinsic_rewards(self, reward):
        for agent_idx, r in enumerate(reward):
            reward[agent_idx] = self.intrinsic_reward[agent_idx]
        return reward

    def step(self, action):
        observation, reward, done, tr, info = self.env.step(action)
        return self.observation(observation), self.get_intrinsic_rewards(reward), done, tr, info

    def reset_state(self):
        self.re_plan.reset_states()
        self.re_plan._agent.add_grid_obstacles(self.get_global_obstacles(), self.get_global_agents_xy())

        self.prev_goals = None
        self.intrinsic_reward = None

    def reset(self, **kwargs):
        observations, infos = self.env.reset(**kwargs)
        self.reset_state()
        return self.observation(observations), infos


class CutObservationWrapper(ObservationWrapper):

    """
    把局部观测裁减到network_input_radius=r的正方形窗口，降低输入尺寸与噪声
    """
    def __init__(self, env, target_observation_radius):
        super().__init__(env)
        self._target_obs_radius = target_observation_radius
        self._initial_obs_radius = self.env.observation_space['obstacles'].shape[0] // 2

        for key, value in self.observation_space.items():
            d = self._initial_obs_radius * 2 + 1
            if value.shape == (d, d):
                r = self._target_obs_radius
                self.observation_space[key] = Box(0.0, 1.0, shape=(r * 2 + 1, r * 2 + 1))

    def observation(self, observations):
        tr = self._target_obs_radius
        ir = self._initial_obs_radius
        d = ir * 2 + 1

        for obs in observations:
            for key, value in obs.items():
                if hasattr(value, 'shape') and value.shape == (d, d):
                    obs[key] = value[ir - tr:ir + tr + 1, ir - tr:ir + tr + 1]

        return observations


class ConcatPositionalFeatures(ObservationWrapper):
    """
    把多种二维特征图（障碍物，其他智能体、路径等）按通道维拼成一个单个向量供卷积网络处理    
    """
    def __init__(self, env):
        super().__init__(env)
        self.to_concat = []

        observation_space = Dict()
        full_size = self.env.observation_space['obstacles'].shape[0]

        for key, value in self.observation_space.items():
            if value.shape == (full_size, full_size):
                self.to_concat.append(key)
            else:
                observation_space[key] = value

        obs_shape = (len(self.to_concat), full_size, full_size)
        observation_space['obs'] = Box(0.0, 1.0, shape=obs_shape)
        self.to_concat.sort(key=self.key_comparator)
        self.observation_space = observation_space

    def observation(self, observations):
        for agent_idx, obs in enumerate(observations):
            main_obs = np.concatenate([obs[key][None] for key in self.to_concat])
            for key in self.to_concat:
                del obs[key]

            for key in obs:
                obs[key] = np.array(obs[key], dtype=np.float32)
            observations[agent_idx]['obs'] = main_obs.astype(np.float32)
        return observations

    @staticmethod
    def key_comparator(x):
        if x == 'obstacles':
            return '0_' + x
        elif 'agents' in x:
            return '1_' + x
        return '2_' + x


class AutoResetWrapper(gymnasium.Wrapper):
    def step(self, action):
        observations, rewards, terminated, truncated, infos = self.env.step(action)
        if all(terminated) or all(truncated):
            observations, _ = self.env.reset()
        return observations, rewards, terminated, truncated, infos
