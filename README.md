## Introduction

This repo holds the code for a final project during the course "Hot Topics in Computational Biology" by Prof. Zeng at Tsinghua University, Beijing.

The project aims to improve performance of the DeepSynergy framework for synergy prediction of drug combinations for cancer treatment proposed by Preuer et al..

For more information see: 

[DeepSynergy: Predicting anti-cancer drug synergy with Deep Learning](https://guides.github.com/features/mastering-markdown/)



## Instructions

1. Download the dataset from [here](http://www.bioinf.jku.at/software/DeepSynergy/X.p.gz)
2. Install the environment from environment.yml using conda 
3. Run data_preprocessing.py to split the data into test and training data for 5 folds
4. Run svd.py to compute the SVD for the training data in the inner folds
5. Run svd_results.py to see the cumulative variance covered by the eigenvectors
6. Run hyperparameter_search.py to perform Bayesian Optimization on the hyperparameters for the inner cross validation of test fold 0
7. Run evaluate_bayes_optimization.py to find the best configuration
8. Run train_model.py to train an ensemble of models on each of the test folds using the best configuration for final evaluation
9. Run test_model_performance.py for final evaluation 

