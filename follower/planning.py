from pogema import GridConfig

# noinspection PyUnresolvedReferences
import cppimport.import_hook
# noinspection PyUnresolvedReferences
from follower_cpp.planner import planner

from pydantic import BaseModel

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


class PlannerConfig(BaseModel):
    use_static_cost: bool = True
    use_dynamic_cost: bool = True
    reset_dynamic_cost: bool = True


class Planner:
    """
    负责协调的一个底层路径规划器的初始化、环境配置、动态成本更新以及路径生成等功能
    为每个智能体计算避障且代价最优的路径
    """
    def __init__(self, cfg: PlannerConfig):
        self.planner = None  # 存储每个智能体的底层路径规划器，负责具体路径计算
        self.obstacles = None  # 障碍物
        self.starts = None  # 起始位置
        self.cfg = cfg   # 存储路径规划的配置参数（如是否启用静态/动态成本、是否重置动态成本等）

    # 为路径规划器提供环境信息（障碍物和起始位置）。
    def add_grid_obstacles(self, obstacles, starts):
        self.obstacles = obstacles
        self.starts = starts
        self.planner = None

    def update(self, obs):
        num_agents = len(obs)
        obs_radius = len(obs[0]['obstacles']) // 2
        if self.planner is None:
            self.planner = [planner(self.obstacles, self.cfg.use_static_cost, self.cfg.use_dynamic_cost, self.cfg.reset_dynamic_cost) for _ in range(num_agents)]
            # 设置坐标偏移
            for i, p in enumerate(self.planner):
                p.set_abs_start(self.starts[i])
            # 预计算静态惩罚矩阵
            if self.cfg.use_static_cost:
                pen_calc = planner(self.obstacles, self.cfg.use_static_cost, self.cfg.use_dynamic_cost, self.cfg.reset_dynamic_cost)
                penalties = pen_calc.precompute_penalty_matrix(obs_radius)
                for p in self.planner:
                    p.set_penalties(penalties)

        for k in range(num_agents):
            if obs[k]['xy'] == obs[k]['target_xy']:
                continue
            # 动态调整路径，避免与其他智能体冲突
            obs[k]['agents'][obs_radius][obs_radius] = 0
            self.planner[k].update_occupations(obs[k]['agents'], (obs[k]['xy'][0] - obs_radius, obs[k]['xy'][1] - obs_radius), obs[k]['target_xy'])
            obs[k]['agents'][obs_radius][obs_radius] = 1
            self.planner[k].update_path(obs[k]['xy'], obs[k]['target_xy'])

    # 获取所有智能体路径
    def get_path(self):
        results = []
        for idx in range(len(self.planner)):
            results.append(self.planner[idx].get_path())
        return results


class ResettablePlanner:
    def __init__(self, cfg: PlannerConfig):
        self._cfg = cfg
        self._agent = None

    def update(self, observations):
        return self._agent.update(observations)

    def get_path(self):
        return self._agent.get_path()

    def reset_states(self, ):
        self._agent = Planner(self._cfg)
