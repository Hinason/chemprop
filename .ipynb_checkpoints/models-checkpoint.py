# mindspore_chem/models.py

import mindspore
from mindspore import nn, ops

from utils import ATOM_FDIM, BOND_FDIM

class MPN(nn.Cell):
    """
    MPN模型，用于二分类任务
    """
    def __init__(self, hidden_size, depth, dropout_prob):
        super(MPN, self).__init__()
        self.hidden_size = hidden_size
        self.depth = depth

        self.W_i_atom = nn.Dense(ATOM_FDIM, hidden_size, has_bias=False)
        self.W_i_bond = nn.Dense(BOND_FDIM, hidden_size, has_bias=False)
        self.W_m = nn.Dense(hidden_size, hidden_size) 
        self.W_h = nn.Dense(hidden_size, hidden_size)

        self.dropout = nn.Dropout(p=dropout_prob)
        self.act_fn = nn.ReLU()

        self.ffn = nn.SequentialCell(
            nn.Dense(hidden_size, hidden_size),
            nn.ReLU(),
            self.dropout,
            nn.Dense(hidden_size, 1)
        )

    def construct(self, graph):
        # 原子和化学键特征的初始嵌入
        h_atom = self.W_i_atom(graph['atom_features']) 
        h_bond_in = self.W_i_bond(graph['bond_features']) 

        b2a_indices = graph['b2a']

        # 第一次消息传递
        # 获取与每条边相关的邻居原子的表示
        h_bond_neighbor_atom = h_atom[b2a_indices] 
        messages = self.act_fn(h_bond_in + h_bond_neighbor_atom) 

        # 消息传递
        for _ in range(self.depth - 1):
            messages_transformed = self.W_m(messages)
            h_atom_new = ops.unsorted_segment_sum(messages_transformed,
                                                  b2a_indices,
                                                  h_atom.shape[0])
            h_atom = self.act_fn(h_atom + h_atom_new)

            h_bond_neighbor_atom = h_atom[b2a_indices]
            messages = self.act_fn(h_bond_in + h_bond_neighbor_atom)

        # 图级别表示
        mol_vecs = []
        atom_start_idx = 0
        for n_atoms_in_mol in graph['n_atoms_scope']: 
            # 提取当前分子的原子表示
            mol_atoms_h = h_atom[atom_start_idx : atom_start_idx + n_atoms_in_mol]
            # 对分子内的原子表示进行平均池化，得到该分子的表示
            mol_vec = ops.ReduceMean(keep_dims=False)(mol_atoms_h, axis=0)
            mol_vecs.append(mol_vec)
            atom_start_idx += n_atoms_in_mol

        mol_vecs_stacked = ops.stack(mol_vecs) 
        mol_vecs_dropped_out = self.dropout(mol_vecs_stacked)

        prediction = self.ffn(mol_vecs_dropped_out) 
        return prediction