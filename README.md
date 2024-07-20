# Feature Selection With DNN Learning Hub 🧠

Welcome to the Feature Selection Learning Hub! This platform is dedicated to providing comprehensive resources and tutorials on feature selection techniques, based on the research articles from Dr. Ofir Lindenbaum's [research lab](https://www.eng.biu.ac.il/lindeno/). Whether you're new to feature selection or looking to deepen your understanding, this hub is designed to support your learning journey.

[Project Page](https://yuvalaza.github.io/project-featselectlib/)

## :bulb: Introduction

Feature selection is a critical aspect of machine learning and data analysis, enabling the identification of relevant features that contribute most to model performance. The research articles produced by Dr. Ofir Lindenbaum's lab offer innovative approaches and methodologies for feature selection, spanning both supervised and unsupervised learning domains. This learning hub serves as a central repository for exploring and implementing these techniques, as well as examining known approaches and algorithms, enriching the understanding of the feature selection field.

## :clipboard: Table of Content

- [Feature Selection With DNN Learning Hub 🧠](#feature-selection-with-dnn-learning-hub-)
  - [:bulb: Introduction](#bulb-introduction)
  - [:clipboard: Table of Content](#clipboard-table-of-content)
  - [:chart\_with\_upwards\_trend: Interactive Examples and Notebooks](#chart_with_upwards_trend-interactive-examples-and-notebooks)
  - [:electric\_plug: Requirements:](#electric_plug-requirements)
  - [:hammer: Usage](#hammer-usage)
  - [:mag\_right: Acknowledgements and References](#mag_right-acknowledgements-and-references)

## :chart_with_upwards_trend: Interactive Examples and Notebooks

- Jupyter Notebooks with Interactive Feature Selection Demonstrations

## :electric_plug: Requirements:

* torch >= 1.9
* scikit-learn >= 0.24
* omegaconf >= 2.0.6
* scipy >= 1.6.0
* matplotlib
* numpy
  
  
## :hammer: Usage

Install the package from pypi:
`pip install featselectlib`

Here is a brief example demonstrating how to use the featselectlib package for feature selection:

```python
import featselectlib
import torch
from omegaconf import OmegaConf

# Define Model Using STG

feature_selection = True
model = featselectlib.STG(task_type='classification',input_dim=X_train.shape[1], output_dim=2, hidden_dims=[60, 20], activation='tanh',
    optimizer='SGD', learning_rate=0.1, batch_size=X_train.shape[0], feature_selection=feature_selection, sigma=0.5, lam=0.5, random_state=1, device='cpu') 

# Run feature selection
   
mu_prob,gates_prob=model.fit(X_train, y_train, nr_epochs=5000, valid_X=X_valid, valid_y=y_valid, print_interval=1000)

# define you cfg parameters for lscae/cae/ls/gl models
cfg = OmegaConf.create({"input_dim": 100})

# define you dataset (Torch based)

dataset = torch.utils.data.Dataset(...)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
lscae_model, lscae_cfg = setup_model(X.shape[1], 'lscae')
lscae.Lscae(kwargs=cfg).select_features(dataloader)
```
For more detailed examples, please refer to the example notebooks [here](https://github.com/yuvalaza/project-featselectlib/tree/master/notebooks)

## :mag_right: Acknowledgements and References
If you use this code, please cite the publication:

```
@incollection{icml2020_5085,
 author = {Yamada, Yutaro and Lindenbaum, Ofir and Negahban, Sahand and Kluger, Yuval},
 booktitle = {Proceedings of Machine Learning and Systems 2020},
 pages = {8952--8963},
 title = {Feature Selection using Stochastic Gates},
 year = {2020}
}
```
```
@article{shaham2022deep,
  title={Deep unsupervised feature selection by discarding nuisance and correlated features},
  author={Shaham, Uri and Lindenbaum, Ofir and Svirsky, Jonathan and Kluger, Yuval},
  journal={Neural Networks},
  year={2022},
  publisher={Elsevier}
}
```
```
@incollection{(NeurIPS2021),
  title={Differentiable Unsupervised Feature Selection based on a Gated Laplacian},
  author={Lindenbaum, Ofir and Shaham, Uri and Peterfreund, Erez and Svirsky, Jonathan and Nicolas, Casey and Kluger, Yuval},
  year={2020}
}
```

[Back to Top](#clipboard-table-of-content)
