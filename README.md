<div align="center">
<h1> Evidential Image Matching: Predicting Transformation Sequences to Derive one Image from Another </h1>

[Daniil Dorin](https://github.com/DorinDaniil)<sup>1 :email:</sup>, [Kseniia Varlamova](https://github.com/varyxi)<sup>1</sup>, [Andrey Grabovoy](https://github.com/andriygav)<sup>1</sup>

<sup>1</sup> Antiplagiat Company, Moscow, Russia

<sup>:email:</sup> Corresponding author

<img width="568" height="172" alt="figure_1" src="https://github.com/user-attachments/assets/69765ade-f55a-40aa-bd6d-9fd756a18d93" />

</div>

### Poblem Statement
Given the dataset:

$$
D = \{(\mathbf{I}^1_i, \mathbf{I}^2_i, t_{i} | i=1, \ldots, N)\}, \quad \mathbf{I}^1_i, \mathbf{I}^2_i \in \mathbb{R}_{+}^{H \times W \times C}, \quad \mathbf{t}_i \in \mathcal{T}^*,
$$

where $\mathcal{T}^*$ is the set of all finite sequences of elements from $\mathcal{T}$. $\mathcal{T}$ is a finite set of transformations. The goal is to construct a mapping:

$$
\mathbf{f}: \quad R_{+}^{H \times W \times C} \times \mathbb{R}_{+}^{H \times W \times C} \rightarrow \mathcal{T}^*.
$$