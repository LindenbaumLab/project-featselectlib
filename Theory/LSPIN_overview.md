---
layout: page
title: Locally Sparse Neural Networks for Tabular Biomedical Data
subtitle: by J. Yang, O. Lindenbaum, Y. Kluger, ICML 2022.
permalink: /Theory/LSPIN_overview/
menubar: theory_menu
---

## Locally SParse Interpretable Network

[Paper](https://arxiv.org/abs/2106.06468)

LSPIN is a neural network designed specifically for tabular biomedical data that often involves low sample sizes and high dimensionality. 
The key innovation is its ability to perform sample-specific feature selection through a gating network that learns which features are most relevant for each individual sample. 
By training this gating network alongside a prediction network, LSPIN identifies the most informative subset of features for each data point, leading to both better interpretability and reduced overfitting. 
This local sparsity approach is particularly effective when different subgroups in the data require different features for prediction, a common scenario in biomedical applications.
