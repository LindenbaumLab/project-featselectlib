---
layout: page
title: Deep Unsupervised Feature Selection by Discarding Nuisance and Correlated Features
subtitle: U. Shaham, O. Lindenbaum, J. Svirsky, Y. Kluger, Elsevier Neural Networks, 2022
permalink: /Theory/discarding_nuisance_and_correlated_features/
menubar: theory_menu
---

# Deep Unsupervised Feature Selection by Discarding Nuisance and Correlated Features


[Paper](https://arxiv.org/abs/2110.05306)

This article introduces LS-CAE (Laplacian Score-regularized Concrete Autoencoder), a novel approach to unsupervised feature selection. LS-CAE addresses two common challenges in feature selection: correlated features and nuisance features.
The method combines a Concrete Autoencoder (CAE) with a Laplacian Score term:
- Concrete Autoencoder (CAE)-The CAE component is designed to select a subset of features that can reconstruct the entire feature set, which effectively handles correlated features. This is achieved through a concrete layer that allows for differentiable feature selection.
- Laplacian Score- The Laplacian Score term is computed on the selected features, encouraging the selection of features that preserve the main structures in the data. This component helps in discarding nuisance features that don't contribute to the underlying data structure.

A key insight of this work is the importance of computing the Laplacian on the subset of selected features rather than on the full feature set. This approach is shown to be crucial when dealing with datasets containing many nuisance features, as it prevents the Laplacian from being corrupted by irrelevant information.
