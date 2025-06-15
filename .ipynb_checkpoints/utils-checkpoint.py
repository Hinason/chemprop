# -*- coding: utf-8 -*-
# mindspore_chem/utils.py

import numpy as np
from rdkit import Chem
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

# MindSpore
import mindspore
from mindspore import ops
from mindspore import Tensor

# =================================================================================
# 分子特征化 
# =================================================================================
def one_of_k_encoding(x, allowable_set):
    """将标签转换为 one-hot 编码"""
    return [x == s for s in allowable_set]

def get_atom_features(atom):
    """从RDKit的atom对象中提取原子特征"""
    possible_atom = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I', 'Si', 'B', 'H']
    features = (
        one_of_k_encoding(atom.GetSymbol(), possible_atom) +
        one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6]) +
        one_of_k_encoding(atom.GetFormalCharge(), [-1, -2, 1, 2, 0]) +
        one_of_k_encoding(int(atom.GetChiralTag()), [0, 1, 2, 3]) +
        [atom.GetIsAromatic()]
    )
    return np.array(features, dtype=np.float32) 

def get_bond_features(bond):
    """从RDKit的bond对象中提取化学键特征"""
    bond_type = bond.GetBondType()
    features = [
        bond_type == Chem.rdchem.BondType.SINGLE,
        bond_type == Chem.rdchem.BondType.DOUBLE,
        bond_type == Chem.rdchem.BondType.TRIPLE,
        bond_type == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()
    ]
    return np.array(features, dtype=np.float32)

# --- 定义特征维度常量 ---
DUMMY_ATOM = Chem.MolFromSmiles('C').GetAtomWithIdx(0)
DUMMY_BOND = Chem.MolFromSmiles('CC').GetBondWithIdx(0)
ATOM_FDIM = len(get_atom_features(DUMMY_ATOM))
BOND_FDIM = len(get_bond_features(DUMMY_BOND))

def mol_to_graph(smiles: str):
    """将SMILES字符串转换为图表示"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: 
        return None
    
    atom_features = np.array([get_atom_features(atom) for atom in mol.GetAtoms()], dtype=np.float32)
    
    bond_features, b2a = [], []
    for bond in mol.GetBonds():
        f_bond = get_bond_features(bond)
        bond_features.extend([f_bond, f_bond])
        b2a.extend([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    
    if not b2a: 
        return None
    
    return {
        'atom_features': atom_features, 
        'bond_features': np.array(bond_features, dtype=np.float32),  
        'b2a': np.array(b2a, dtype=np.int32)                     
    }
# =================================================================================
# 评估指标类
# =================================================================================
class Metric:
    """计算二分类任务的评估指标"""
    def __init__(self):
        self.reset()

    def update(self, preds, labels):
        if not isinstance(preds, Tensor):
            preds = Tensor(preds)
        if not isinstance(labels, Tensor):
            labels = Tensor(labels)

        probs = ops.Sigmoid()(preds).asnumpy()
        self.all_probs.extend(probs.flatten().tolist())
        self.all_preds.extend((probs > 0.5).astype(int).flatten().tolist())
        self.all_labels.extend(labels.asnumpy().astype(int).flatten().tolist())

    def compute(self, metric_name='AUC'):
        if not self.all_labels or not self.all_probs:
            print(f"Warning: No data to compute metric '{metric_name}'. Returning 0.0.")
            return 0.0
        if len(set(self.all_labels)) <= 1:
            print(f"Warning: All labels are the same. Metric '{metric_name}' might be ill-defined. Returning 0.0 for AUC.")
            if metric_name == 'AUC':
                return 0.0
            elif metric_name == 'Accuracy':
                return accuracy_score(self.all_labels, self.all_preds)
        try:
            if metric_name == 'AUC':
                return roc_auc_score(self.all_labels, self.all_probs)
            elif metric_name == 'Accuracy':
                return accuracy_score(self.all_labels, self.all_preds)
            elif metric_name == 'Precision':
                from sklearn.metrics import precision_score
                return precision_score(self.all_labels, self.all_preds, zero_division=0)
            elif metric_name == 'Recall':
                from sklearn.metrics import recall_score
                return recall_score(self.all_labels, self.all_preds, zero_division=0)
            elif metric_name == 'AveragePrecision':
                 return average_precision_score(self.all_labels, self.all_probs)
            else:
                raise ValueError(f"Metric '{metric_name}' not supported.")
        except ValueError as e:
            print(f"ValueError computing metric '{metric_name}': {e}. Returning 0.0")
            return 0.0

    def reset(self):
        self.all_preds = []
        self.all_labels = []
        self.all_probs = []

    def get_labels_and_probs(self):
        """返回收集到的所有标签和概率"""
        return self.all_labels, self.all_probs
    
# =================================================================================
# 可视化模块
# =================================================================================
def plot_roc_curve(y_true, y_probs, ax=None, label='Model', title='ROC Curve', save_path=None):
    """
    绘制 ROC 曲线。

    参数:
    y_true (array-like): 真实标签。
    y_probs (array-like): 模型预测的阳性类概率。
    ax (matplotlib.axes.Axes, optional): 用于绘图的 Axes 对象。如果为 None，则创建新的 Figure 和 Axes。
    label (str, optional): 曲线的标签。
    title (str, optional): 图表标题。
    save_path (str, optional): 保存图像的路径。如果为 None，则显示图像。
    """
    if len(set(y_true)) <= 1:
        print("Cannot plot ROC curve: only one class present in y_true.")
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot([0, 1], [0, 1], linestyle='--', color='navy', label='No Skill')
        ax.text(0.5, 0.5, "ROC N/A\n(Single Class)", ha="center", va="center", fontsize=12, color="red")
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(title)
        ax.legend(loc='lower right')
        if save_path:
            plt.savefig(save_path)
            print(f"ROC curve placeholder saved to {save_path}")
            plt.close()
        else:
            plt.show()
        return

    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(fpr, tpr, lw=2, label=f'{label} (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], linestyle='--', color='navy', lw=2, label='No Skill')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc='lower right')
    ax.grid(True)

    if save_path and ax is None:
        plt.savefig(save_path)
        print(f"ROC curve saved to {save_path}")
        plt.close()
    elif ax is None:
        plt.show()

def plot_precision_recall_curve(y_true, y_probs, ax=None, label='Model', title='Precision-Recall Curve', save_path=None):
    """
    绘制 Precision-Recall 曲线。
    """
    if not hasattr(y_true, '__len__') or isinstance(y_true, (int, float)):
        print(f"Error: y_true is not a list or array-like object. Got type: {type(y_true)}, value: {y_true}")
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "Error: Invalid y_true data", ha="center", va="center", fontsize=12, color="red")
        if save_path and ax is None:
            plt.savefig(save_path)
            plt.close()
        elif ax is None:
            plt.show()
        return

    if not y_true: 
        print("Cannot plot Precision-Recall curve: y_true is empty.")
        return

    y_true_np = np.array(y_true)

    if len(set(y_true_np)) <= 1:
        print("Cannot plot Precision-Recall curve: only one class present in y_true.")
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        positive_class_count = np.sum(y_true_np == 1)
        total_count = len(y_true_np)
        no_skill = positive_class_count / total_count if total_count > 0 else 0

        ax.plot([0, 1], [no_skill, no_skill], linestyle='--', color='navy', label=f'No Skill (AP = {no_skill:.2f})')
        ax.text(0.5, 0.5, "PR Curve N/A\n(Single Class)", ha="center", va="center", fontsize=12, color="red")
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(title)
        ax.legend(loc='best')
        if save_path and ax is None:
            plt.savefig(save_path)
            print(f"PR curve placeholder saved to {save_path}")
            plt.close()
        elif ax is None:
            plt.show()
        return

    precision, recall, _ = precision_recall_curve(y_true_np, y_probs)
    avg_precision = average_precision_score(y_true_np, y_probs)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(recall, precision, lw=2, label=f'{label} (AP = {avg_precision:.2f})')


    positive_class_count = np.sum(y_true_np == 1)
    total_count = len(y_true_np)
    no_skill = positive_class_count / total_count if total_count > 0 else 0 # 防除零
    ax.plot([0, 1], [no_skill, no_skill], linestyle='--', color='navy', lw=2, label=f'No Skill (AP = {no_skill:.2f})')

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_ylim([0.0, 1.05])
    ax.set_xlim([0.0, 1.0])
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True)

    if save_path and ax is None:
        plt.savefig(save_path)
        print(f"Precision-Recall curve saved to {save_path}")
        plt.close()
    elif ax is None:
        plt.show()

def plot_evaluation_summary(y_true, y_probs, model_name='Model', save_dir=None):
    """
    在一张图中绘制 ROC 曲线和 Precision-Recall 曲线。

    参数:
    y_true (array-like): 真实标签。
    y_probs (array-like): 模型预测的阳性类概率。
    model_name (str, optional): 模型名称，用于图例和文件名。
    save_dir (str, optional): 保存图像的目录。如果为 None，则显示图像。
                                文件将命名为 {model_name}_roc_pr_curves.png。
    """
    if not y_true or not y_probs:
        print("No data provided for plotting.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'{model_name} - Evaluation Curves', fontsize=16)

    # ROC Curve
    plot_roc_curve(y_true, y_probs, ax=axes[0], label=model_name, title='ROC Curve')

    # Precision-Recall Curve
    plot_precision_recall_curve(y_true, y_probs, ax=axes[1], label=model_name, title='Precision-Recall Curve')

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, f"{model_name.replace(' ', '_').lower()}_evaluation_curves.png")
        plt.savefig(file_path)
        print(f"Evaluation summary plot saved to {file_path}")
        plt.close(fig)
    else:
        plt.show()