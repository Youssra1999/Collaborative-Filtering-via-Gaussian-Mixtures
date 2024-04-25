
# Collaborative Filtering via Gaussian Mixtures on Netflix Data

## Introduction

This project aims to build a mixture model for collaborative filtering using Gaussian mixtures. The dataset used consists of movie ratings provided by users, extracted from a subset of the Netflix database. Since users have rated only a small fraction of the movies, the dataset is partially filled. The objective is to predict the missing entries in the matrix.

## Approach

We employ mixtures of Gaussians to tackle this problem. The underlying assumption is that each user's rating profile is a sample from a mixture model. In other words, there are K potential types of users, and within the context of each user, we need to sample a user type and then the rating profile from the Gaussian distribution associated with that type. To estimate this mixture from the partially observed rating matrix, we utilize the Expectation Maximization (EM) algorithm.

The EM algorithm iteratively assigns users to types (E-step) and then re-estimates the Gaussians associated with each type (M-step). Once we have the mixture model, we can use it to predict values for all the missing entries in the data matrix.

## Implementation

The implementation involves the following steps:
1. Initialization of parameters.
2. E-step: Assign users to types.
3. M-step: Re-estimate Gaussians associated with each type.
4. Predict missing entries.

## Usage

To use the collaborative filtering model:
1. Prepare the movie rating dataset.
2. Initialize the model parameters.
3. Run the EM algorithm to estimate the mixture model.
4. Predict the missing entries using the trained model.

## Requirements

- Python 3.x
- NumPy
- Scikit-learn

## References

- T. Hofmann, "Collaborative Filtering via Gaussian Probabilistic Latent Semantic Analysis," SIGIR 2004.
