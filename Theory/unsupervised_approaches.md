---
layout: page
title: Key Approaches and Algorithms
permalink: /Theory/unsupervised_approaches/
menubar: theory_menu
---

## Key Approaches and Algorithms

### The Laplacian Score

The Laplacian Score is an unsupervised method to evaluate the importance of features based on the structure of the data itself. This method relies on the concept of graph Laplacians to measure feature relevance.

#### Graph Laplacians

Graph Laplacians are matrix representations that capture the similarity between data points in a dataset. To define a graph Laplacian, we begin with a kernel matrix K, which is constructed using the Gaussian kernel.

where $K_{i,j}$ is the similarity between data points $x_i$ and $x_j$, and $\sigma_b$ is a bandwidth parameter that can be chosen based on the dataset.
The unnormalized graph Laplacian $L_{\text{un}}$ is then defined as:
[L_{\text{un}} = D - K]
where $D$ is a diagonal matrix with entries $D_{i,i} = \sum_{j=1}^{n} K_{i,j}$, representing the degree of each node in the graph.

![Graph_Laplacian](/docs/assets/image/graph_laplacian.png)

#### Laplacian Score

The Laplacian Score (LS) is an unsupervised feature selection metric that leverages the eigenvectors of the graph Laplacian matrix. Specifically, the score for a feature #f$ is calculated by the quadratic form:

$$f^T L_{\text{un}} f,$$

where $f$ is the feature vector. The score is minimized when $f$ aligns well with the subspace formed by the smallest eigenvectors of $ L_{\text{un}}$, indicating that the feature preserves the data's local structure.

Eigenvectors corresponding to smaller eigenvalues of the graph Laplacian are associated with slowly varying features across the graph, which are considered to capture the intrinsic structure of the data, such as clusters.

In summary, features with the smallest Laplacian Scores are selected as they are most indicative of the underlying structure of the data.

![LP-Score](/docs/assets/image/ls_fig4.png)