#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>
#include <vector>
#include <queue>
#include <cmath>
#include <set>
#include <map>
#include <list>
#include <iostream>
#define INF 1000000000
namespace py = pybind11;

/*
planner 类是一个路径规划器，用于在网格环境中计算从起点到目标点的最短路径。
支持静态成本（例如障碍物）和动态成本（例如其他智能体的占用情况），并结合启发式算法（如 A*）进行路径规划
*/


struct Node {
    Node(int _i = INF, int _j = INF, float _g = INF, float _h = 0) : i(_i), j(_j), g(_g), h(_h), f(_g+_h){}
    int i;
    int j; // 节点的坐标
    float g;  // 从起点到当前节点的代价
    float h;  // 启发式代价
    float f;  // 总代价 (f = g + h)
    std::pair<int, int> parent;
    bool operator<(const Node& other) const
    {
        return this->f < other.f or (std::abs(this->f - other.f) < 1e-5 and this->g < other.g);
    }
    bool operator>(const Node& other) const
    {
        return this->f > other.f or (std::abs(this->f - other.f) < 1e-5 and this->g > other.g);
    }
    bool operator==(const Node& other) const
    {
        return this->i == other.i and this->j == other.j;
    }
    bool operator==(const std::pair<int, int> &other) const
    {
        return this->i == other.first and this->j == other.second;
    }
};

class planner
{
    std::pair<int, int> start;  // 起点
    std::pair<int, int> goal;   // 目标点
    // 用于存储当前的偏移量（绝对坐标）
    // 例如，起点在 (0, 0)，偏移量为 (2, 3)，则实际起点为 (2, 3)
    // 这样可以在不同的网格中复用同一个规划器
    std::pair<int, int> abs_offset;
    std::priority_queue<Node, std::vector<Node>, std::greater<Node>> OPEN; // A * 算法的开放列表（优先队列，最小堆）
    std::vector<std::vector<int>> grid;  // 网格地图，grid[i][j]表示可通行，非零表示障碍物
    std::vector<std::vector<float>> num_occupations;  // 动态代价矩阵：记录每个网格被占用的次数
    std::vector<std::vector<float>> penalties;  // 静态惩罚矩阵：记录每个网格的静态代价（例如障碍物的代价）
    std::vector<std::vector<float>> h_values;   //预计算的启发式代价矩阵
    std::vector<std::vector<Node>> nodes;  // 存储所有节点的“最优状态”（包括最新的g值和父节点）
    bool use_static_cost;
    bool use_dynamic_cost;
    bool reset_dynamic_cost;  // 是否在每次更新路径时重置动态代价矩阵
    inline float h(std::pair<int, int> n)
    {    
        /*
        返回节点到目标点的启发式代价，实际使用预计算的h_values矩阵
        */
        //return abs(n.first - goal.first) + abs(n.second - goal.second); 欧式距离
        return h_values[n.first][n.second];
    }
    // 获取邻居节点，过滤障碍物节点（grid值非 0 的节点）
    std::vector<std::pair<int,int>> get_neighbors(std::pair<int, int> node)
    {
        std::vector<std::pair<int,int>> neighbors;
        std::vector<std::pair<int,int>> deltas = {{0,1},{1,0},{-1,0},{0,-1}};
        for(auto d:deltas)
        {
            std::pair<int,int> n(node.first + d.first, node.second + d.second);
            if(grid[n.first][n.second] == 0)
                neighbors.push_back(n);
        }
        return neighbors;
    }
    // 计算从起点到目标点的最短路径，避免障碍物和动态冲突
    void compute_shortest_path()
    {
        Node current;
        while(!OPEN.empty() and !(current == goal))
        {
            current = OPEN.top();
            OPEN.pop();

            
            if(nodes[current.i][current.j].g < current.g)
                continue;

            // 如果邻居节点的代价更低，则更新副节点和代价，并加入到优先队列
            for(auto n: get_neighbors({current.i, current.j})) {
                float cost(1);
                if(use_static_cost)
                    cost = penalties[n.first][n.second];
                if(use_dynamic_cost)
                    cost += num_occupations[n.first][n.second];
                if(nodes[n.first][n.second].g > current.g + cost)
                {
                    OPEN.push(Node(n.first, n.second, current.g + cost, h(n)));
                    nodes[n.first][n.second].g = current.g + cost;
                    nodes[n.first][n.second].parent = {current.i, current.j};
                }
            }
        }
    }

    float get_avg_distance(int si, int sj)
    {
        /*
        计算从给定起点 (si, sj) 到所有可达节点的平均距离。
        */
        std::queue<std::pair<int, int>> fringe;  // BFS队列，存储待探索的节点坐标
        fringe.emplace(si, sj);  // 将起点加入队列
        auto result = std::vector<std::vector<int>>(grid.size(), std::vector<int>(grid.front().size(), -1)); // 距离矩阵
        result[si][sj] = 0;  //起点到自身的距离设置为0
        std::vector<std::pair<int, int>> moves = {{0,1},{1,0},{-1,0},{0,-1}};
        while(!fringe.empty())
        {
            auto pos = fringe.front();
            fringe.pop();
            for(const auto& move: moves)
            {
                int new_i(pos.first + move.first), new_j(pos.second + move.second);  //声明并初始化
                
                // 检查 邻居节点是否可访问
                if(grid[new_i][new_j] == 0 && result[new_i][new_j] < 0)
                {
                    result[new_i][new_j] = result[pos.first][pos.second] + 1;
                    fringe.emplace(new_i, new_j);
                }
            }
        }
        float avg_dist(0), total_nodes(0);
        for(size_t i = 0; i < grid.size(); i++)
            for(size_t j = 0; j < grid[0].size(); j++)
                if(result[i][j] > 0)
                {
                    avg_dist += result[i][j];
                    total_nodes++;
                }
        return avg_dist/total_nodes;
    }

    void update_h_values(std::pair<int, int> g)
    {
        /*
        基于目标点，用类 Dijkstra 算法预计算所有可通行节点到目标点的最短路径成本，
        作为h_values（保证启发式的可采纳性，确保 A * 算法最优性
        */

        std::priority_queue<Node, std::vector<Node>, std::greater<Node>> open;
        h_values = std::vector<std::vector<float>>(grid.size(), std::vector<float>(grid.front().size(), INF));
        h_values[g.first][g.second] = 0;
        open.push(Node(g.first, g.second, 0, 0));
        while(!open.empty())
        {
            Node current = open.top();
            open.pop();
            for(auto n: get_neighbors({current.i, current.j})) {
                float cost(1);
                if(use_static_cost)
                    cost = penalties[n.first][n.second];
                if(h_values[n.first][n.second] > current.g + cost)
                {
                    open.push(Node(n.first, n.second, current.g + cost, 0));
                    h_values[n.first][n.second] = current.g + cost;
                }
            }
        }
    }

    void reset()
    {
        /*
        重置节点状态和open列表，初始化起点节点
        */
        nodes = std::vector<std::vector<Node>>(grid.size(), std::vector<Node>(grid.front().size(), Node()));
        OPEN = std::priority_queue<Node, std::vector<Node>, std::greater<Node>>();
        Node s = Node(start.first, start.second, 0, h(start));
        OPEN.push(s);
    }

public:
    // 构造函数
    planner(std::vector<std::vector<int>> _grid={}, float _use_static_cost=1.0, float _use_dynamic_cost=1.0, bool _reset_dynamic_cost=true):
    grid(_grid), use_static_cost(_use_static_cost), use_dynamic_cost(_use_dynamic_cost), reset_dynamic_cost(_reset_dynamic_cost)
    {
        abs_offset = {0, 0};
        goal = {0,0};
        start = {0, 0};
        nodes = std::vector<std::vector<Node>>(grid.size(), std::vector<Node>(grid.front().size(), Node()));
        num_occupations = std::vector<std::vector<float>>(grid.size(), std::vector<float>(grid.front().size(), 0));
        penalties = std::vector<std::vector<float>>(grid.size(), std::vector<float>(grid.front().size(), 1));
    }

    std::vector<std::vector<float>> get_num_occupied_matrix()
    {
        return num_occupations;
    }

    std::vector<std::vector<float>> precompute_penalty_matrix(int obs_radius)
    {
        /*
        实现一个静态惩罚矩阵（penalties）的预计算功能，用于路径规划中的代价评估
        其核心逻辑是通过计算网格中每个可通行节点到其他节点的平均距离，生成一个 “惩罚值” 矩阵 
        ——平均距离越小的节点（越孤立的区域），惩罚值越高，从而引导路径规划算法优先选择可达性更好的区域
        */
        penalties = std::vector<std::vector<float>>(grid.size(), std::vector<float>(grid.front().size(), 0));
        float max_avg_dist(0);  // // 用于记录所有节点中最大的平均距离
        for(size_t i = obs_radius; i < grid.size() - obs_radius; i++)
            for(size_t j = obs_radius; j < grid.front().size() - obs_radius; j++)
                if(grid[i][j] == 0)  //仅处理可通行节点（grid[i][j] == 0表示非障碍物）
                {
                    penalties[i][j] = get_avg_distance(i, j);
                    max_avg_dist = std::fmax(max_avg_dist, penalties[i][j]);
                }
        // 再次遍历非边界可通行节点，将平均距离转换为惩罚值
        for(size_t i = obs_radius; i < grid.size() - obs_radius; i++)
            for(size_t j = obs_radius; j < grid.front().size() - obs_radius; j++)
                if(grid[i][j] == 0)
                    // 惩罚值 = 最大平均距离 / 当前节点的平均距离（平均距离越小，惩罚值越高）
                    penalties[i][j] = max_avg_dist / penalties[i][j];
        return penalties;
    }

    void set_penalties(std::vector<std::vector<float>> _penalties)
    {
        /*
        收一个外部定义的二维浮点型矩阵 _penalties，
        并将其赋值给类的私有成员变量 penalties（静态惩罚矩阵），覆盖原有的矩阵值。
        */
        penalties = std::move(_penalties);
    }

    // 更新网格中被占用单元格的计数
    void update_occupied_cells(const std::list<std::pair<int, int>>& _occupied_cells, std::pair<int, int> cur_goal)
    {
        if(reset_dynamic_cost)
            if(goal.first != cur_goal.first || goal.second != cur_goal.second)
                num_occupations = std::vector<std::vector<float>>(grid.size(), std::vector<float>(grid.front().size(), 0));
        for(auto o:_occupied_cells)
            num_occupations[o.first][o.second] += 1.0;
    }

    void update_occupations(py::array_t<double> array, std::pair<int, int> cur_pos, std::pair<int, int> cur_goal)
    {
        // 将其转化为绝对坐标
        cur_goal = {cur_goal.first + abs_offset.first, cur_goal.second + abs_offset.second};
        if(reset_dynamic_cost)
            if(goal.first != cur_goal.first || goal.second != cur_goal.second)
                num_occupations = std::vector<std::vector<float>>(grid.size(), std::vector<float>(grid.front().size(), 0));
        py::buffer_info buf = array.request();  // 获取python数组的缓冲区信息
        std::list<std::pair<int, int>> occupied_cells;
        double *ptr = (double *) buf.ptr;
        cur_pos = {cur_pos.first + abs_offset.first, cur_pos.second + abs_offset.second};
        for(size_t i = 0; i < static_cast<size_t>(buf.shape[0]); i++)
            for(size_t j = 0; j < static_cast<size_t>(buf.shape[1]); j++)
                if(ptr[i*buf.shape[1] + j] != 0)
                    occupied_cells.push_back({cur_pos.first + i, cur_pos.second + j});
        for(auto o:occupied_cells)
            num_occupations[o.first][o.second]+= 1.0;
    }

    void update_path(std::pair<int, int> s, std::pair<int, int> g)
    {
        s = {s.first + abs_offset.first, s.second + abs_offset.second};  // 将起点坐标转换为绝对坐标
        g = {g.first + abs_offset.first, g.second + abs_offset.second};  // 将目标点坐标转换为绝对坐标
        start = s;
        if(goal != g)
            update_h_values(g);
        goal = g;
        reset();
        compute_shortest_path();
    }

    std::list<std::pair<int, int>> get_path()
    {
        /*
        return:一个存储坐标对的列表，表示从起点到目标点的路径
        */
        std::list<std::pair<int, int>> path; // 存储路径节点
        std::pair<int, int> next_node(INF,INF);
        // 检验目标点是否可达
        if(nodes[goal.first][goal.second].g < INF)
            next_node = goal;
        if(next_node.first < INF and (next_node.first != start.first or next_node.second != start.second))
        {
            // 从目标点开始，沿着每个节点的父节点回溯，直到回到起点
            while (nodes[next_node.first][next_node.second].parent != start) {
                path.push_back(next_node);
                next_node = nodes[next_node.first][next_node.second].parent;
            }
            path.push_back(next_node);
            path.push_back(start);
            path.reverse();
        }
        for(auto it = path.begin(); it != path.end(); it++)
        {
            it->first -= abs_offset.first;
            it->second -= abs_offset.second;
        }
        return path;
    }

    // 用来获取路径规划中从当前起点出发的下一个节点坐标
    std::pair<std::pair<int, int>, std::pair<int, int>> get_next_node()
    {
        /*
        工作原理是：从目标点回溯路径，
        直到找到距离起点最近的那个节点（即起点的直接后继节点），然后将这两个节点的坐标（经过偏移校正后）返回。
        */
        std::pair<int, int> next_node(INF, INF);
        if(nodes[goal.first][goal.second].g < INF)
            next_node = goal;
        if(next_node.first < INF and (next_node.first != start.first or next_node.second != start.second))
            while (nodes[next_node.first][next_node.second].parent != start)
                next_node = nodes[next_node.first][next_node.second].parent;
        if(next_node == start)  // 处理节点与目标点重合的情况
            next_node = {INF, INF};
        if(next_node.first < INF)
            return {{start.first - abs_offset.first, start.second - abs_offset.second},
                    {next_node.first - abs_offset.first, next_node.second - abs_offset.second}};
        return {{INF, INF}, {INF, INF}};
    }

    // 设置坐标偏移量，支持相对坐标与绝对坐标转换；
    void set_abs_start(std::pair<int, int> offset)
    {
        abs_offset = offset;
    }
};
