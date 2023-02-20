# Regularized Blackbox Regression
This repository contains a Python implementation of a regularized blackbox regression algorithm. The implementation is based on the scikit-learn library.

## Description
The regularize_blackbox() function automatically tunes a blackbox regression model by regularizing it with one of three regularization methods: Dropout, NoiseAddition, and Robust. The function uses cross-validation to select the amount of regularization that optimizes the specified criterion (MSE or MAD).

## Dependencies
The code requires the following libraries to run:

numpy
pandas
scikit-learn

## Usage
The regularize_blackbox() function takes the following parameters:

model: a function that takes as input a matrix X in R<sup>n x p</sup> and a vector of responses Y in R<sup>n</sup> and returns a function that maps inputs to outputs.
x_train: an array-like of shape (n,p) that contains the training data.
y_train: an array-like of shape (n,) that contains the response values to x_train.
m: a positive integer that specifies the number of Monte Carlo replicates to be used if the regularization specified is Dropout or NoiseAddition.
k: a positive integer indicating the number of CV-folds to be used to tune the amount of regularization.
c: an array-like of shape (p,) that contains a vector of column bounds to be used if the method specified is Robust.
reg_method: a regularization method that belongs to the set {Dropout, NoiseAddition, Robust}.
eval_method: a criterion to be used to evaluate the regularization that belongs to the set {MSE, MAD} where MSE encodes mean square error and MAD encodes mean absolute deviation.
The function returns a predictive model that optimizes the specified criterion using the specified regularization method.
