---
layout: page
title: Key Approaches and Algorithms
permalink: /Theory/supervised_approaches/
menubar: theory_menu
---

# Supervised Key Approaches and Algorithms
There are numerous feature selection methods available, each with its own strengths and applications. Here are some well-known approaches:

## Filter Methods:

- **ANOVA F-test**-
 The ANOVA (Analysis of Variance) F- test is used for feature selection in classification problems. It assesses the relationship between each feature and the target variable independently. The F-statistic measures the ratio of variance between classes to variance within classes. Features with higher F-scores are considered more relevant for classification.
- **Chi-square test**-
 The Chi-square test is used for feature selection in classification, particularly with categorical features. It measures the dependence between a feature and the target variable. Higher chi-square scores indicate a stronger relationship, suggesting the feature is more relevant for prediction.

## Wrapper Methods:

- **Recursive Feature Elimination (RFE)**-
  RFE is an iterative process that starts with all features and progressively eliminates the least important ones. It trains a model, ranks features by importance, removes the least important feature, and repeats until a desired number of features is reached. RFE can be computationally intensive but often yields good results.
- **Forward Selection**-
  Forward Selection starts with no features and iteratively adds the most beneficial ones. In each iteration, it tests the addition of each unused feature, selects the one that improves the model's performance the most, and adds it to the selected features set. This process continues until a stopping criterion is met, such as no significant improvement in performance.   


## Embedded Method:
- **Lasso (Least Absolute Shrinkage and Selection Operator)**-  is a popular regularization technique used in supervised   learning for feature selection and preventing overfitting. It adds a penalty term to the loss function, which is the sum of the absolute values of the model coefficients multiplied by a tuning parameter. This L1 regularization encourages sparsity in the model by shrinking some feature coefficients to exactly zero, effectively eliminating less important features. Lasso is particularly useful for linear problems, when dealing with high-dimensional data or when there's a need to identify the most relevant features for prediction.



