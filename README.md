# TRFL

TRFL (pronounced "truffle") is a library built on top of TensorFlow that exposes
several useful building blocks for implementing Reinforcement Learning agents.


## Installation

TRFL can be installed from pip with the following command:
`pip install trfl`

TRFL will work with both the CPU and GPU version of tensorflow, but to allow
for that it does not list Tensorflow as a requirement, so you need to install
Tensorflow and Tensorflow-probability separately if you haven't already done so.

## Usage Example

```python
import tensorflow as tf
import trfl

# Q-values for the previous and next timesteps, shape [batch_size, num_actions].
q_tm1 = tf.get_variable(
    "q_tm1", initializer=[[1., 1., 0.], [1., 2., 0.]], dtype=tf.float32)
q_t = tf.get_variable(
    "q_t", initializer=[[0., 1., 0.], [1., 2., 0.]], dtype=tf.float32)

# Action indices, discounts and rewards, shape [batch_size].
a_tm1 = tf.constant([0, 1], dtype=tf.int32)
r_t = tf.constant([1, 1], dtype=tf.float32)
pcont_t = tf.constant([0, 1], dtype=tf.float32)  # the discount factor

# Q-learning loss, and auxiliary data.
loss, q_learning = trfl.qlearning(q_tm1, a_tm1, r_t, pcont_t, q_t)
```

`loss` is the tensor representing the loss. For Q-learning, it is half the
squared difference between the predicted Q-values and the TD targets, shape
`[batch_size]`. Extra information is in the `q_learning` namedtuple, including
`q_learning.td_error` and `q_learning.target`.

The `loss` tensor can be differentiated to derive the corresponding RL update.

```python
reduced_loss = tf.reduce_mean(loss)
optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
train_op = optimizer.minimize(reduced_loss)
```

All loss functions in the package return both a loss tensor and a namedtuple
with extra information, using the above convention, but different functions
may have different `extra` fields. Check the documentation of each function
below for more information.

# Documentation

Check out the full documentation page
[here](https://github.com/deepmind/trfl/blob/master/docs/index.md).
