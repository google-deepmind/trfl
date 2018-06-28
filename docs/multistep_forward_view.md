### Multistep forward view

The
[multistep_forward_view](https://github.com/deepmind/trfl/blob/master/trflsequence_ops.py?q=multistep_forward_view)
function computes mixed multistep returns in terms of the instantaneous
`rewards`, discount factors `pcontinues`, `state_value` estimates, and mixing
weights `lambda_`. In the math that follows we will replace these by
$$r_{0:T-1}$$, $$\gamma_{0:T-1}$$, $$V_{1:T}$$, and $$\lambda_{0:T-1}$$,
respectively. Note that in the implementation, the `state_values` array is
offset in time by 1 relative to the other arrays.

The mixed returns $$M_t$$ are computed by the following recurrence, using a
backwards scan:

$$
M_t = r_t + \gamma_t (\lambda_t M_{t+1} + (1-\lambda_t) V_{t+1}) \label{eq:recurrence} \tag{1}\\
M_{T-1} = r_{T-1} + \gamma_{T-1} V_T
$$

Here we can see why $$M_t$$ is a valid estimate of the return at time $$t$$. The
recurrence is applying the Bellman Equation, using the $$\lambda_t$$-weighted
mixture of $$M_{t+1}$$ and $$V_{t+1}$$, both of which are valid estimates of the
expected return at time $$t+1$$.

What's left is to show that we are computing the right mixture of multistep
returns. Let $$R(t, k)$$ be the $$\gamma_{t:k}$$-discounted return from time
$$t$$ to $$k$$:

$$
R(t, k) = r_t + \gamma_t r_{t+1} + \cdots + (\gamma_t \gamma_{t+1} \cdots \gamma_{k-1}) r_k + (\gamma_t \gamma_{t+1} \cdots \gamma_k) V_{k+1}\\
= \sum_{i=t}^k \left(\prod_{j=t}^{i-1} \gamma_j \right) r_i + \left(\prod_{i=t}^k \gamma_i \right) V_{k+1}
$$

We should mention that $$R(t, k)$$ would correspond to the $$k-t$$ step return
$$G_t^{k-t}$$ in Sutton and Barto's notation. Note that $$R$$ satisfies a
Bellman-style recurrence:

$$
R(t-1, k) = r_{t-1} + \gamma_{t-1} R(t, k)\\
R(t, t) = r_t + \gamma_t V_{t+1}
$$

The desired[^1] $$\lambda$$-weighted mixture is given by:

$$
L(t, k) = (1-\lambda_t) R(t, t) + (1-\lambda_{t+1}) \lambda_t R(t, t+1) + \cdots + ((1-\lambda_k) \lambda_{k-1} \cdots \lambda_t) R(t, k) + \lambda_k\cdots\lambda_t R(t, k)\\
= \sum_{i=t}^k \left((1-\lambda_i) \prod_{j=t}^{i-1} \lambda_j \right) R(t, i) + \left(\prod_{i=t}^k \lambda_i \right) R(t, k) \\
$$

We have that

$$ L(t-1, k) = \sum_{i=t-1}^k \left((1-\lambda_i) \prod_{j=t-1}^{i-1} \lambda_j
\right) R(t-1, i) + \left(\prod_{i=t-1}^k \lambda_i \right) R(t-1, k) \\
= (1-\lambda_{t-1})R(t-1, t-1) + \lambda_{t-1}\sum_{i=t}^k \left((1-\lambda_i) \prod_{j=t}^{i-1} \lambda_j \right) (r_{t-1} + \gamma_{t-1} R(t, i)) + \lambda_{t-1}\left(\prod_{i=t}^k \lambda_i \right) \left(r_{t-1} + \gamma_{t-1} R(t, k)\right) \\
= r_{t-1} + (1-\lambda_{t-1})\gamma_{t-1} V_t + \lambda_{t-1}\gamma_{t-1} L(t, k)
$$

Thus, $$L(t, k)$$ also satisfies recurrence $$\eqref{eq:recurrence}$$, as
desired.

[^1]: [Sutton and Barto](http://incompleteideas.net/book/ebook/node74.html),
    equation 7.3
