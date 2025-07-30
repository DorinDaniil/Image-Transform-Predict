# Methods of Evidential Image Matching: Predicting Transformation Sequences to Derive One Image from Another

Given the dataset:

$$
D = \{(\mathbf{I}^1_i, \mathbf{I}^2_i, t_{i} | i=1, \ldots, N)\}, \quad \mathbf{I}^1_i, \mathbf{I}^2_i \in \mathbb{R}_{+}^{H \times W \times C}, \quad \mathbf{t}_i \in \mathcal{T}^*,
$$

where $\mathcal{T}^*$ is the set of all finite sequences of elements from $\mathcal{T}$. $\mathcal{T}$ is a finite set of transformations. The goal is to construct a mapping:

$$
\mathbf{f}: \quad R_{+}^{H \times W \times C} \times \mathbb{R}_{+}^{H \times W \times C} \rightarrow \mathcal{T}^*.
$$
