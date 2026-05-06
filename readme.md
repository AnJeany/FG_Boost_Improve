# Feat-XGBoost: Enhancing Sampling Performance in XGBoost by Ensemble Feature Engineering

[![Paper](https://img.shields.io/badge/Paper-Pattern%20Recognition%202026-blue)](https://doi.org/10.1016/j.patcog.2026.113169)
[![DOI](https://img.shields.io/badge/DOI-10.1016/j.patcog.2026.113169-green)](https://doi.org/10.1016/j.patcog.2026.113169)
[![Python](https://img.shields.io/badge/Python-3.7%2B-orange)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.5%2B-red)](https://xgboost.readthedocs.io/)

This repository contains the official implementation of **Feat-XGBoost**, an enhanced classifier that integrates ensemble feature engineering techniques within the boosting steps of the XGBoost algorithm with adapted gradient-based one-sided sampling.

## 📄 Paper Abstract

Feature engineering is crucial in enhancing model performance, yet effectively combining multiple feature transformations to maximize their benefits remains a key challenge. In this study, we propose an innovative approach that integrates various feature engineering techniques within the boosting steps of the XGBoost algorithm and adapts the gradient-based one-sided sampling, forming an enhanced classifier named **Feat-XGBoost**.

## 📖 Citation

If you use this code or find our work helpful in your research, please cite:

```bibtex
@article{kong2026enhancing,
  title={Enhancing Sampling Performance in XGBoost by Ensemble Feature Engineering},
  author={Kong, Lingping and Suganthan, Ponnuthurai Nagaratnam and Sn{\'a}{\v{s}}el, V{\'a}clav and Ojha, Varun and Pan, Jeng-Shyang},
  journal={Pattern Recognition},
  pages={113169},
  year={2026},
  publisher={Elsevier}
}
```

## 🔗 Links

- **Paper**: [https://doi.org/10.1016/j.patcog.2026.113169](https://doi.org/10.1016/j.patcog.2026.113169)
- **ScienceDirect**: [https://www.sciencedirect.com/science/article/pii/S0031320326001342](https://www.sciencedirect.com/science/article/pii/S0031320326001342)

## ✨ Key Features

- Integration of multiple feature engineering techniques within XGBoost boosting steps
- Adapted gradient-based one-sided sampling for improved efficiency
- Ensemble approach to maximize benefits of feature transformations
- Compatible with UCI benchmark datasets

## 📊 Datasets

The experiments were conducted using datasets from the **UCI Machine Learning Repository**. For detailed information about the datasets used in this study, please refer to the paper.

## 🙏 Acknowledgements and Resources

We would like to express our gratitude to the following resources and repositories that inspired and contributed to this work:

### Blog Posts & Tutorials
- **Gregory Gundersen's Blog** - [Random Fourier Features](http://gregorygundersen.com/blog/2019/12/23/random-fourier-features/)
- **YouTube Tutorial** - [Machine Learning Course by MeanXai](https://youtu.be/APZyWo9hIj0)
- **GitHub Repository** - [MeanXai Machine Learning Examples](https://github.com/meanxai/machine_learning)

### Code References
- **Random Fourier Features Implementation** - [rffgpr.py](https://github.com/gwgundersen/random-fourier-features/blob/master/rffgpr.py)
- **Scikit-learn MDS Implementation** - [\_mds.py](https://github.com/scikit-learn/scikit-learn/blob/70fdc843a4b8182d97a3508c1a426acc5e87e980/sklearn/manifold/_mds.py)
- **Types of Transformations** - [Krishnaik06 Repository](https://github.com/krishnaik06/Types-Of-Trnasformation)
- **AutoFeat Library** - [AutoFeat by cod3licious](https://github.com/cod3licious/autofeat)

### Research Papers
- **t-SNE Paper** - [Visualizing Data using t-SNE](https://cseweb.ucsd.edu/~lvdmaaten/workshops/nips2010/papers/vandermaaten.pdf) by Laurens van der Maaten



## 📧 Contact

For questions, comments, or inquiries about this work, please contact the author:

**Lingping **  
Email: [lingping.kong@vsb.cz]



## ⭐ Citation Request

If you use Feat-XGBoost in your research, please cite our paper as shown in the citation section above.
