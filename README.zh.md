# TRFL

[English](README.md) | [简体中文](README.zh.md)

TRFL（发音为 "truffle"）是一个基于 TensorFlow 构建的强化学习基础模块核心库，提供了一系列用于实现强化学习智能体（Reinforcement Learning Agents）的实用数学构建块与可微损失函数。

## 安装指南 (Installation)

TRFL 可通过 pip 运行以下命令进行安装：

```shell
pip install trfl
```

TRFL 兼容 CPU 与 GPU 版本的 TensorFlow。为了保持灵活性，TRFL 并未在包依赖中强行绑定特定版本的 TensorFlow，因此如果您尚未安装，需要单独安装 TensorFlow 以及 TensorFlow Probability：

```shell
pip install tensorflow tensorflow-probability
```

## 使用示例 (Usage Example)

```python
import tensorflow as tf
import trfl

# 上一时间步与当前时间步的 Q 值，形状为 [batch_size, num_actions]
q_tm1 = tf.get_variable(
    "q_tm1", initializer=[[1., 1., 0.], [1., 2., 0.]], dtype=tf.float32)
q_t = tf.get_variable(
    "q_t", initializer=[[0., 1., 0.], [1., 2., 0.]], dtype=tf.float32)

# 动作索引、折扣因子与即时奖励，形状为 [batch_size]
a_tm1 = tf.constant([0, 1], dtype=tf.int32)
r_t = tf.constant([1, 1], dtype=tf.float32)
pcont_t = tf.constant([0, 1], dtype=tf.float32)  # 折扣因子 / 状态继续概率

# 计算 Q-learning 损失与辅助元数据
loss, q_learning = trfl.qlearning(q_tm1, a_tm1, r_t, pcont_t, q_t)
```

`loss` 是表示损失的张量。对于 Q 学习（Q-learning），其计算为预测 Q 值与时序差分目标（TD Target）之间平方差的一半，张量形状为 `[batch_size]`。辅助信息封装在 `q_learning` 具名元组（namedtuple）中，包括 `q_learning.td_error` 和 `q_learning.target`。

`loss` 张量可直接求导，从而计算出相应的强化学习梯度更新：

```python
reduced_loss = tf.reduce_mean(loss)
optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
train_op = optimizer.minimize(reduced_loss)
```

本工具包中的所有损失函数均遵循上述约定，同时返回损失张量和包含辅助信息的 namedtuple。不同函数可能包含不同的 `extra` 字段，详情请参阅各函数的具体文档说明。

## 核心算法构建块一览

TRFL 提供了广泛的强化学习核心数学组件与损失函数：

* **状态价值学习 (State Value Learning)**：TD 学习 (`td_learning`)、广义 $\lambda$ 回报 (`generalized_lambda_returns`)、TD($\lambda$)、QV-max
* **离散动作价值学习 (Discrete-action Value Learning)**：Q 学习 (`qlearning`)、Double Q 学习 (`double_qlearning`)、Persistent Q-learning、SARSA、SARSE、Q($\lambda$)、QV 学习
* **分布式价值学习 (Distributional Value Learning)**：Categorical Distributional Q 学习、Categorical Distributional Double Q 学习、Categorical Distributional TD 学习
* **策略梯度 (Policy Gradient)**：连续动作策略梯度 (`policy_gradient`)、策略熵损失 (`policy_entropy_loss`)、序列 A2C 损失 (`sequence_a2c_loss`)、确定性策略梯度 (DPG)
* **像素控制与辅助任务 (Pixel Control & Auxiliary Tasks)**：Pixel Control 辅助损失函数，加速特征表征学习

## 完整文档 (Documentation)

请访问[官方完整文档页面](https://github.com/deepmind/trfl/blob/master/docs/index.md)获取所有 API 与算法模块的详尽说明。

---

> **说明**：本文档为基于英文原文的中文翻译版本。如翻译内容与英文原文存在不一致之处，请以 [英文原版 (README.md)](README.md) 为准。
