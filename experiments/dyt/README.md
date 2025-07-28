## Transformers without Normalization 

[Transformers without Normalization](https://arxiv.org/abs/2503.10622) replaces explicit torch normalization layers such as LN and RMSNorm with the element-wise operation: $\tanh(\alpha x)$, which the authors name Dynamic Tanh. Note that this is a normalizing operation, it roughly translates to $l^{\infty}$ normalization, constraining $x$ to the $n$-dimensional unit hypercube.

