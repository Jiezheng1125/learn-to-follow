from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import config


class CommLayer(nn.Module):
    """
    单次"点对点注意力通信"的基本单元。每个agent 作为查询方，从其他智能体接受信息
    结合相对位置进行注意力聚合，并用GRUcell进行更新
    """
    # 参数说明：
    def __init__(self, input_dim=config.hidden_dim, message_dim=32, pos_embed_dim=16, num_heads=4):
        super().__init__()
        self.input_dim = input_dim
        self.message_dim = message_dim
        self.pos_embed_dim = pos_embed_dim
        self.num_heads = num_heads

        self.norm = nn.LayerNorm(input_dim)

        self.position_embeddings = nn.Linear((2*config.obs_radius+1)**2, pos_embed_dim)

        self.message_key = nn.Linear(input_dim+pos_embed_dim, message_dim * num_heads)
        self.message_value = nn.Linear(input_dim+pos_embed_dim, message_dim * num_heads)
        self.hidden_query = nn.Linear(input_dim, message_dim * num_heads)

        self.head_agg = nn.Linear(message_dim * num_heads, message_dim * num_heads)

        self.update = nn.GRUCell(num_heads*message_dim, input_dim)

    def position_embed(self, relative_pos, dtype, device):
        """
        将相对位置编码为一组可用于神经网络的嵌入向量，
        用于后续的消息传递和注意力机制  
        原理：将二维相对位置转化为one-hot，在通过线性层得到低维嵌入
        input:
        + 相对向量
        + 数据类型
        + 设备
        """
        batch_size, num_agents, _, _ = relative_pos.size()
        # 将视野（FOV）之外的智能体的相对位置 置为 0
        relative_pos[(relative_pos.abs() > config.obs_radius).any(3)] = 0
        #  创建一个 one-hot 编码矩阵，表示每个智能体相对于其他智能体的位置
        one_hot_position = torch.zeros((batch_size*num_agents*num_agents, 9*9), dtype=dtype, device=device)
        
        relative_pos += config.obs_radius
        relative_pos = relative_pos.reshape(batch_size*num_agents*num_agents, 2)
        relative_pos_idx = relative_pos[:, 0] + relative_pos[:, 1]*9
        one_hot_position[torch.arange(batch_size*num_agents*num_agents), relative_pos_idx.long()] = 1
        position_embedding = self.position_embeddings(one_hot_position)

        return position_embedding

    def forward(self, hidden, relative_pos, comm_mask):
        """
        实现多智能体之间的消息传递与注意力机制，根据通信掩码和相对位置，对每个智能体进行更新
        """

        batch_size, num_agents, hidden_dim = hidden.size()
        #生成掩码注意力的掩码，用来屏蔽不可通信的智能体对 comm_mask: [batch_size, num_agents, num_agents] 表示那些智能体之间可以通信
        attn_mask = (comm_mask==False).unsqueeze(3).unsqueeze(4)  #  并在第三和第四和个维度上扩展维度
        relative_pos = relative_pos.clone()

        # 将智能体之间的相对位置编码为嵌入向量，作为后续消息传递的辅助信息
        position_embedding = self.position_embed(relative_pos, hidden.dtype, hidden.device)

        input = hidden  # 保存原始隐藏状态，后续用于 GRUCell 更新和未更新智能体的回填。

        hidden = self.norm(hidden)  # 使用 LayerNorm对每个智能体的隐藏状态进行标准化，提升训练的稳定性

        # 将归一化后的隐藏状态映射为“查询向量”，用于注意力机制。原理：通过线性层得到 query，reshape 以便后续与 key 做点积。
        hidden_q = self.hidden_query(hidden).view(batch_size, 1, num_agents, self.num_heads, self.message_dim) # batch_size x num_agents x message_dim*num_heads

        """作用：为每对智能体构造消息输入，将自身隐藏状态和位置嵌入拼接。
           原理：每个智能体都要和其他智能体通信，repeat_interleave 实现所有组合，拼接位置嵌入后 reshape 成三维结构。
        """
        message_input = hidden.repeat_interleave(num_agents, dim=1).view(batch_size*num_agents*num_agents, hidden_dim)
        message_input = torch.cat((message_input, position_embedding), dim=1)
        message_input = message_input.view(batch_size, num_agents, num_agents, self.input_dim+self.pos_embed_dim)
        
        """
        作用：将消息输入分别映射为 key 和 value，用于注意力机制。
        原理：通过线性层分别得到 key 和 value，reshape 以支持多头注意力
        """
        message_k = self.message_key(message_input).view(batch_size, num_agents, num_agents, self.num_heads, self.message_dim)
        message_v = self.message_value(message_input).view(batch_size, num_agents, num_agents, self.num_heads, self.message_dim)

        # attention
        attn_score = (hidden_q * message_k).sum(4, keepdim=True) / self.message_dim**0.5 # batch_size x num_agents x num_agents x self.num_heads x 1
        attn_score.masked_fill_(attn_mask, torch.finfo(attn_score.dtype).min)  # 屏蔽不可通信的智能体对
        attn_weights = F.softmax(attn_score, dim=1) # softmax 得到每个智能体对的权重

        # agg  用注意力权重加权聚合 value，得到每个智能体的聚合消息，再通过线性层整合多头信息
        agg_message = (message_v * attn_weights).sum(1).view(batch_size, num_agents, self.num_heads*self.message_dim)
        agg_message = self.head_agg(agg_message)

        # update hidden with request message   用聚合后的消息和原始隐藏状态，通过 GRUCell 更新每个智能体的隐藏状态
        input = input.view(-1, hidden_dim)
        agg_message = agg_message.view(batch_size*num_agents, self.num_heads*self.message_dim)
        updated_hidden = self.update(agg_message, input)

        """
        作用：对于没有收到任何消息的智能体，保持原始隐藏状态不变。
        原理：update_mask 标记哪些智能体收到消息，torch.where 实现条件更新，最后 reshape 回原始结构。
        """
        # some agents may not receive message, keep it as original
        update_mask = comm_mask.any(1).view(-1, 1)
        hidden = torch.where(update_mask, updated_hidden, input)
        hidden = hidden.view(batch_size, num_agents, hidden_dim)

        return hidden



class CommBlock(nn.Module):
    def __init__(self, hidden_dim=config.hidden_dim, message_dim=128, pos_embed_dim=16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.message_dim = message_dim
        self.pos_embed_dim = pos_embed_dim

        self.request_comm = CommLayer()
        self.reply_comm = CommLayer()


    def forward(self, latent, relative_pos, comm_mask):
        '''
        latent shape: batch_size x num_agents x latent_dim
        relative_pos shape: batch_size x num_agents x num_agents x 2
        comm_mask shape: batch_size x num_agents x num_agents
        '''
        
        batch_size, num_agents, latent_dim = latent.size()

        assert relative_pos.size() == (batch_size, num_agents, num_agents, 2), relative_pos.size()
        assert comm_mask.size() == (batch_size, num_agents, num_agents), comm_mask.size()

        # 没有任何智能体通信，则直接返回原始的latent，不做任何处理
        if torch.sum(comm_mask).item() == 0:
            return latent

        hidden = self.request_comm(latent, relative_pos, comm_mask)

        comm_mask = torch.transpose(comm_mask, 1, 2)

        hidden = self.reply_comm(hidden, relative_pos, comm_mask)

        return hidden


# 主网络
class Network(nn.Module):
    def __init__(self, input_shape=config.obs_shape, selective_comm=config.selective_comm):
        super().__init__()

        self.hidden_dim = config.hidden_dim
        self.latent_dim = self.hidden_dim + 5
        self.obs_shape = input_shape
        self.selective_comm = selective_comm

        self.obs_encoder = nn.Sequential(
            nn.Conv2d(input_shape[0], 64, 3, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 3, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 192, 3, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(192, 256, 3, 1),
            nn.LeakyReLU(0.2, True),
            nn.Flatten(),
        )

        self.recurrent = nn.GRUCell(self.latent_dim, self.hidden_dim)
        self.comm = CommBlock(self.hidden_dim)

        self.hidden = None

        # dueling q structure
        self.adv = nn.Linear(self.hidden_dim, 5)
        self.state = nn.Linear(self.hidden_dim, 1)

    @torch.no_grad() # 关闭梯度计算
    def step(self, obs, last_act, pos):
        """
        input:
        + obs: 所有智能体的观测
        + last_act: 所有智能体的上一步动作
        + pos: 所有智能体的位置
        output:
        + actions: 所有智能体的动作
        + q_val: 所有智能体的动作值函数
        """
        num_agents = obs.size(0) # 智能体数量
        agent_indexing = torch.arange(num_agents)  # 用于后续索引
        relative_pos = pos.unsqueeze(0)-pos.unsqueeze(1)  # 计算每对智能体的相对位置，形状为 [num_agents, num_agents, 2]
        
        in_obs_mask = (relative_pos.abs() <= config.obs_radius).all(2)  # 判断每对智能体是否在彼此范围内，得到布尔掩码
        in_obs_mask[agent_indexing, agent_indexing] = 0 # 将自己设置为0,表示自己不能和自己通信

        if self.selective_comm:
            test_mask = in_obs_mask.clone()
            test_mask[agent_indexing, agent_indexing] = 1
            num_in_obs_agents = test_mask.sum(1)  # 统计每个智能体视野内的智能体数量
            origin_agent_idx = torch.zeros(num_agents, dtype=torch.long)
            for i in range(num_agents-1):  # 计算每个智能体在展开后的张量中的起始索引
                origin_agent_idx[i+1] = test_mask[i, i:].sum() + test_mask[i+1, :i+1].sum() + origin_agent_idx[i]
            test_obs = torch.repeat_interleave(obs, num_agents, dim=0).view(num_agents, num_agents, *config.obs_shape)[test_mask]

            test_relative_pos = relative_pos[test_mask]  # 得到每对可观测智能体的相对位置，并整体平移为非负
            test_relative_pos += config.obs_radius

            # 观测向两种对应的相对位置为0
            test_obs[torch.arange(num_in_obs_agents.sum()), 0, test_relative_pos[:, 0], test_relative_pos[:, 1]] = 0

            test_last_act = torch.repeat_interleave(last_act, num_in_obs_agents, dim=0)
            if self.hidden is None:
                test_hidden = torch.zeros((num_in_obs_agents.sum(), self.hidden_dim))
            else:
                test_hidden = torch.repeat_interleave(self.hidden, num_in_obs_agents, dim=0)

            test_latent = self.obs_encoder(test_obs)
            test_latent = torch.cat((test_latent, test_last_act), dim=1)

            test_hidden = self.recurrent(test_latent, test_hidden)
            self.hidden = test_hidden[origin_agent_idx]

            # 计算Dueling Q网络的优势值、状态值和Q值
            adv_val = self.adv(test_hidden)
            state_val = self.state(test_hidden)
            test_q_val = (state_val + adv_val - adv_val.mean(1, keepdim=True))
            test_actions = torch.argmax(test_q_val, 1)

            actions_mat = torch.ones((num_agents, num_agents), dtype=test_actions.dtype) * -1
            actions_mat[test_mask] = test_actions
            diff_action_mask = actions_mat != actions_mat[agent_indexing, agent_indexing].unsqueeze(1)

            # 通信掩码为“在视野内且动作不同”的智能体对
            assert (in_obs_mask[agent_indexing, agent_indexing] == 0).all()
            comm_mask = torch.bitwise_and(in_obs_mask, diff_action_mask)

        else:

            latent = self.obs_encoder(obs)
            latent = torch.cat((latent, last_act), dim=1)

            # mask out agents that are far away
            dist_mat = (relative_pos[:, :, 0]**2 + relative_pos[:, :, 1]**2)
            _, ranking = dist_mat.topk(min(config.max_comm_agents, num_agents), dim=1, largest=False)
            dist_mask = torch.zeros((num_agents, num_agents), dtype=torch.bool)
            dist_mask.scatter_(1, ranking, True)
            comm_mask = torch.bitwise_and(in_obs_mask, dist_mask)
            # comm_mask[torch.arange(num_agents), torch.arange(num_agents)] = 0

            if self.hidden is None:
                self.hidden = self.recurrent(latent)
            else:
                self.hidden = self.recurrent(latent, self.hidden)
            
        assert (comm_mask[agent_indexing, agent_indexing] == 0).all()

        # 调用通信模块，进行一次多智能体通信，并更新隐藏状态
        self.hidden = self.comm(self.hidden.unsqueeze(0), relative_pos.unsqueeze(0), comm_mask.unsqueeze(0))
        self.hidden = self.hidden.squeeze(0)

        adv_val = self.adv(self.hidden)
        state_val = self.state(self.hidden)

        q_val = (state_val + adv_val - adv_val.mean(1, keepdim=True))

        actions = torch.argmax(q_val, 1).tolist()

        return actions, q_val.numpy(), self.hidden.squeeze(0).numpy(), relative_pos.numpy(), comm_mask.numpy()
    
    def reset(self):
        self.hidden = None

    @autocast()  # 让函数内部自动使用混合精度计算，提高效率，常用于训练阶段，推理阶段一般不用
    def forward(self, obs, last_act, steps, hidden, relative_pos, comm_mask):
        '''
        used for training
        '''
        # obs shape: seq_len, batch_size, num_agents, obs_shape
        # relative_pos shape: batch_size, seq_len, num_agents, num_agents, 2
        seq_len, batch_size, num_agents, *_ = obs.size()

        obs = obs.view(seq_len*batch_size*num_agents, *self.obs_shape)
        last_act = last_act.view(seq_len*batch_size*num_agents, config.action_dim)

        latent = self.obs_encoder(obs)
        latent = torch.cat((latent, last_act), dim=1)
        latent = latent.view(seq_len, batch_size*num_agents, self.latent_dim)

        hidden_buffer = []
        for i in range(seq_len):
            # hidden size: batch_size*num_agents x self.hidden_dim
            hidden = self.recurrent(latent[i], hidden)
            hidden = hidden.view(batch_size, num_agents, self.hidden_dim)
            hidden = self.comm(hidden, relative_pos[:, i], comm_mask[:, i])
            # only hidden from agent 0
            hidden_buffer.append(hidden[:, 0])
            hidden = hidden.view(batch_size*num_agents, self.hidden_dim)

        # hidden buffer size: batch_size x seq_len x self.hidden_dim
        hidden_buffer = torch.stack(hidden_buffer).transpose(0, 1)

        # hidden size: batch_size x self.hidden_dim
        hidden = hidden_buffer[torch.arange(config.batch_size), steps-1]

        adv_val = self.adv(hidden)
        state_val = self.state(hidden)

        q_val = state_val + adv_val - adv_val.mean(1, keepdim=True)

        return q_val
    
