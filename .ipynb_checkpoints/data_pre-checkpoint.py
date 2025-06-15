# -*- coding: utf-8 -*-
# mindspore_chem/data_pre.py

import numpy as np
import pandas as pd
from collections import defaultdict

# MindSpore
import mindspore
from mindspore import Tensor

# RDKit
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

# Scikit-learn
from sklearn.model_selection import train_test_split

from utils import mol_to_graph

# =================================================================================
# 数据集划分
# =================================================================================
def scaffold_split(dataset, smiles_list, sizes, seed=42):
    """按分子骨架划分数据集"""
    train_size, val_size, test_size = sizes[0], sizes[1], sizes[2]
    scaffolds = defaultdict(list)
    for i, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
            scaffolds[scaffold].append(i)

    scaffold_sets = sorted(list(scaffolds.values()), key=len, reverse=True)
    
    train_idx, val_idx, test_idx = [], [], []
    train_count, val_count, test_count = 0, 0, 0
    total_count = len(dataset)

    # 分配验证集和测试集
    for group in scaffold_sets:
        if val_count < total_count * val_size:
            val_idx.extend(group)
            val_count += len(group)
        elif test_count < total_count * test_size:
            test_idx.extend(group)
            test_count += len(group)
        else:
            train_idx.extend(group)
            train_count += len(group)
    
    return [dataset[i] for i in train_idx], [dataset[i] for i in val_idx], [dataset[i] for i in test_idx]

def split_data(df, smiles_column, target_column, split_type, split_sizes):
    """根据参数划分数据的主函数"""
    print(f"Splitting data with method: '{split_type}'")
    
    dataset = list(zip(df[smiles_column], df[target_column].astype(np.float32)))
    
    if split_type == 'random':
        train_val_data, test_data = train_test_split(dataset, test_size=split_sizes[2], random_state=42)
        train_data, val_data = train_test_split(train_val_data, test_size=split_sizes[1]/(split_sizes[0]+split_sizes[1]), random_state=42)
    elif split_type == 'scaffold':
        smiles_list = df[smiles_column].tolist()
        train_data, val_data, test_data = scaffold_split(dataset, smiles_list, split_sizes)
    else:
        raise ValueError(f"Split type '{split_type}' not supported.")
        
    print(f"Data split sizes: Train={len(train_data)}, Validation={len(val_data)}, Test={len(test_data)}")
    return train_data, val_data, test_data

# =================================================================================
# 自定义数据加载器类
# =================================================================================
class MoleculeDataLoader:
    """自定义分子数据加载器，处理变长序列"""
    def __init__(self, data, batch_size, shuffle=False):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(data)))
        
    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        
        for i in range(0, len(self.indices), self.batch_size):
            batch_indices = self.indices[i:i + self.batch_size]
            batch_data = [self.data[idx] for idx in batch_indices]
            yield self._collate_batch(batch_data)
    
    def __len__(self):
        return (len(self.data) + self.batch_size - 1) // self.batch_size
    
    def _collate_batch(self, batch_data):
        """处理一个批次的数据"""
        valid_graphs = []
        valid_labels = []
        
        for smiles, label in batch_data:
            graph = mol_to_graph(smiles)
            if graph is not None:
                valid_graphs.append(graph)
                valid_labels.append(label)
        
        if not valid_graphs:
            return None
        
        # 合并所有图数据
        all_atom_features = []
        all_bond_features = []
        all_b2a = []
        n_atoms_scope = []
        
        atom_offset = 0
        for graph in valid_graphs:
            atom_features = graph['atom_features']
            bond_features = graph['bond_features']
            b2a = graph['b2a']
            
            all_atom_features.append(atom_features)
            all_bond_features.append(bond_features)
            
            # 调整键到原子的索引
            adjusted_b2a = b2a + atom_offset
            all_b2a.append(adjusted_b2a)
            
            n_atoms_scope.append(len(atom_features))
            atom_offset += len(atom_features)
        
        # 合并所有特征
        batched_atom_features = np.concatenate(all_atom_features, axis=0)
        batched_bond_features = np.concatenate(all_bond_features, axis=0) if all_bond_features[0].size > 0 else np.empty((0, all_bond_features[0].shape[1]))
        batched_b2a = np.concatenate(all_b2a, axis=0) if all_b2a[0].size > 0 else np.empty((0,), dtype=np.int32)
        
        # 构建图字典
        batch_graph = {
            'atom_features': Tensor(batched_atom_features.astype(np.float32), mindspore.float32),
            'bond_features': Tensor(batched_bond_features.astype(np.float32), mindspore.float32),
            'b2a': Tensor(batched_b2a.astype(np.int32), mindspore.int32),
            'n_atoms_scope': n_atoms_scope
        }
        
        # 标签
        labels = Tensor(np.array(valid_labels, dtype=np.float32).reshape(-1, 1), mindspore.float32)
        
        return batch_graph, labels

def create_data_loader(data, batch_size, shuffle=False):
    """创建数据加载器的工厂函数"""
    return MoleculeDataLoader(data, batch_size, shuffle)