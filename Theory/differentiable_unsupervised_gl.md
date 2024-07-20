---
layout: page
title: Differentiable Unsupervised Feature Selection based on a Gated Laplacian
subtitle: O. Lindenbaum, U. Shaham, J. Svirsky, E. Peterfreund, Y. Kluger, NeurIPS 2021
permalink: /Theory/differentiable_unsupervised_gl/
menubar: theory_menu
---

# Differentiable Unsupervised Feature Selection based on a Gated Laplacian


[Paper](https://arxiv.org/pdf/2007.04728.pdf)

This article presents DUFS (Differentiable Unsupervised Feature Selection), a novel approach that combines Stochastic Gates (STG) with the Laplacian Score (LS) for unsupervised feature selection. DUFS addresses the challenge of selecting relevant features in the presence of nuisance features. The method integrates two key components:

- Stochastic Gates (STG)- DUFS employs trainable stochastic gates for each feature, which allow the method to explore different subsets of features during the training process. This gating mechanism enables the re-evaluation of feature importance based on various feature combinations.
- Laplacian Score (LS)- The Laplacian Score is computed on the gated features, encouraging the selection of features that preserve the main structures in the data. This component helps in identifying and retaining features that contribute to the underlying data structure.

A significant contribution of this work is the integration of these components into a fully differentiable framework. This allows for end-to-end training using traditional optimization methods such as stochastic gradient descent.
The key innovation lies in computing the Laplacian Score on the gated inputs rather than on the full feature set. This approach is particularly crucial when dealing with datasets containing a large number of nuisance features. By evaluating the Laplacian Score on different subsets of features during training, DUFS can more effectively identify and retain the truly informative features while discarding those that may obscure the underlying data structure.
