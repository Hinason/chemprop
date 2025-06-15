## 项目简介

本项目是对发表于 *Nature* 的论文 **"Discovery of a structural class of antibiotics with explainable deep learning"** (Wong, F., et al., 2023) 的部分代码复现。

该项目利用 **有向消息传递神经网络 (D-MPNN)** 来学习分子结构与生物活性之间的关系，并对分子的特定属性进行预测。本代码库专注于实现模型在分类任务上的训练和评估流程，例如预测一个分子是否具有抗菌活性。

代码库包含了数据预处理、模型构建、训练和评估的完整脚本，并提供了在 BACE、BBBP 和 MRSA 三个公开数据集上的训练示例。

## 背景

寻找具有新颖化学结构的新型抗生素对于应对全球性的抗生素耐药性危机至关重要。传统的药物发现方法周期长、成本高。深度学习，特别是图神经网络（GNN），为高效筛选和发现活性化合物提供了强大的工具。该研究展示了一种可解释的深度学习方法，不仅能预测分子活性，还能揭示其背后的化学结构原理，从而加速新药发现。

## 方法论

模型的核心是 **有向消息传递神经网络 (Directed Message-Passing Neural Network, D-MPNN)**，这是一种为分子图定制的图神经网络。其工作流程如下：

1. **分子图构建**: 每个分子被转换成一个图结构，其中原子是节点，化学键则被视为有向边。
2. **消息传递**: 模型沿着化学键（有向边）迭代地传递和更新特征信息。在每一步，每个键的特征向量会聚合来自其起始原子和相关键的信息，从而学习到原子所处的复杂化学环境。
3. **读出阶段**: 在多轮消息传递后，所有键的特征向量被聚合（例如求和或取平均），形成一个代表整个分子的全局特征向量。
4. **预测**: 该全局特征向量被送入一个标准的前馈神经网络，以输出最终的预测结果。在本项目中，输出的是一个二分类概率（例如，有活性/无活性）。

## 文件结构

```
.
├── .gitignore               # Git忽略文件配置
├── data/
│   ├── bace.csv             # BACE 数据集
│   ├── bbbp.csv             # BBBP 数据集
│   └── mrsa.csv             # MRSA 数据集
├── data_pre.py              # 数据预处理和封装脚本
├── models.py                # D-MPNN 模型架构定义
├── utils.py                 # 工具函数 (如 AverageMeter)
├── train.py                 # 主要的训练脚本
├── train_bace.ipynb         # 在 BACE 数据集上进行训练的Jupyter Notebook
├── train_bbbp.ipynb         # 在 BBBP 数据集上进行训练的Jupyter Notebook
└── train_mrsa.ipynb         # 在 MRSA 数据集上进行训练的Jupyter Notebook
```

## 环境配置

我们推荐使用 `conda` 来管理您的项目环境，以确保依赖的兼容性。

Bash

```
# 1. 使用 Conda 创建并激活一个新环境
conda create -n chemprop python=3.8
conda activate chemprop

# 2. 安装核心依赖库
pip install torch numpy pandas scikit-learn tqdm

# 3. 安装 RDKit (用于处理分子化学信息)
pip install rdkit-pypi
```

## 使用说明

您可以通过两种方式来运行代码的训练流程。

### 1. 使用 Jupyter Notebooks (推荐)

对于初次使用者，我们强烈推荐通过 `Jupyter Notebook` 来运行代码。这种方式更加直观，可以分步骤执行和观察结果。

- `train_bace.ipynb`: 在 BACE 数据集上进行训练的完整示例。
- `train_bbbp.ipynb`: 在 BBBP 数据集上进行训练的完整示例。
- `train_mrsa.ipynb`: 在 MRSA 数据集上进行训练的完整示例。

请在您的环境中启动 Jupyter，并打开相应的 `.ipynb` 文件，然后按顺序执行所有代码单元格。

### 2. 使用命令行脚本

您也可以直接使用 `train.py` 脚本来启动训练。**请注意：此脚本专用于分类任务。**

**训练 BACE 数据集模型:**

```
python train.py \
    --data_path data/bace.csv \
    --save_dir checkpoints/bace \
    --epochs 30 \
    --batch_size 50
```

**训练 BBBP 数据集模型 (使用不同的超参数):**

```
python train.py \
    --data_path data/bbbp.csv \
    --save_dir checkpoints/bbbp \
    --epochs 50 \
    --hidden_size 400 \
    --depth 4 \
    --dropout 0.2
```

#### 命令行参数说明

以下是 `train.py` 脚本支持的所有参数：

- `--data_path`: **(必需)** `.csv` 数据集文件的路径。
- `--save_dir`: **(必需)** 用于保存模型检查点和日志的目录。
- `--epochs`: 训练的总轮数 (默认: `30`)。
- `--batch_size`: 批处理大小 (默认: `50`)。
- `--hidden_size`: 模型隐藏层的维度 (默认: `300`)。
- `--depth`: D-MPNN 消息传递的深度/步数 (默认: `3`)。
- `--dropout`: Dropout 的比率 (默认: `0.0`)。
- `--seed`: 用于复现结果的随机种子 (默认: `0`)。

## 预期结果

脚本在训练过程中，会实时打印每个 epoch 的训练损失和验证集上的 **AUC-ROC** 分数。训练结束后，性能最好的模型（以验证集 AUC 为准）将被保存在您指定的 `--save_dir` 目录下的文件中。