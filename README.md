# Fast Approximation of Shapley Values with Limited Data
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-red.svg)](#python)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

This repository is an implementation of ["Fast Approximation of Shapley Values with Limited Data"](https://amrmalkhatib.github.io/assets/files/ff-shap.pdf)
![](./figures/architecture.pdf)
 

## Abstract
Shapley values have multiple desired and theoretically proven properties for explaining black-box model predictions. However, the exact computation of Shapley values can be computationally very expensive, precluding their use when timely explanations are required. FastSHAP is an approach for fast approximation of Shapley values using a trained neural network (the explainer). A novel approach, called FF-SHAP, is proposed, which incorporates three modifications to FastSHAP:
i) the explainer is trained on ground-truth explanations rather than a weighted least squares characterization of the Shapley
values, ii) cosine similarity is used as a loss function instead of mean-squared error, and iii) the actual prediction of the underlying model is given as input to the explainer. 
An empirical investigation is presented showing that FF-SHAP significantly outperforms FastSHAP with respect to fidelity, measured using Spearman’s rank-order correlation. The investigation further shows that FF-SHAP even outperforms FastSHAP when using substantially smaller amounts of data to train the explainer, and
more importantly, FF-SHAP still maintains the performance level of FastSHAP even when trained with as little as 15% of training data.

## Requirements

Install Dependencies: Ensure that you have the necessary dependencies installed. You can install them using

```
pip install -r requirements.txt
```
## Usage

We provide the implementation of the proposed method FF-SHAP + a Jupyter notebook example. 
