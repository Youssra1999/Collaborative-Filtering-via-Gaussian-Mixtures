# Collaborative Filtering via Gaussian Mixtures on Netflix Data

## Introduction

<img src="https://github.com/Youssra1999/Collaborative-Filtering-via-Gaussian-Mixtures/blob/main/Pink%20Black%20Photocentric%20Neon%20Tech%20Talk%20Podcast%20Instagram%20Post%20(2).png" alt="Image" width="500" align="right">

This project aims to build a mixture model for collaborative filtering using Gaussian mixtures. The dataset used consists of movie ratings provided by users, extracted from a subset of the Netflix database. Since users have rated only a small fraction of the movies, the dataset is partially filled. The objective is to predict the missing entries in the matrix.


# Approach

We employ mixtures of Gaussians to tackle this problem. The underlying assumption is that each user's rating profile is a sample from a mixture model. In other words, there are K potential types of users, and within the context of each user, we need to sample a user type and then the rating profile from the Gaussian distribution associated with that type. To estimate this mixture from the partially observed rating matrix, we utilize the Expectation Maximization (EM) algorithm.

The EM algorithm iteratively assigns users to types (E-step) and then re-estimates the Gaussians associated with each type (M-step). Once we have the mixture model, we can use it to predict values for all the missing entries in the data matrix.

## Implementation

### K-means Comparison

In this part of the project, we compare clustering obtained via K-means to the (soft) clustering induced by EM. Our K-means algorithm is modified to return additional information. Specifically, the resulting clusters of points are used to estimate a Gaussian model for each cluster. Thus, our K-means algorithm returns a mixture model where the means of the component Gaussians are the centroids computed by the K-means algorithm. This allows us to directly plot and compare solutions returned by the two algorithms as if they were both estimating mixtures.

The steps involved in this comparison are as follows:

1. **Loading the Toy Dataset**: We load a 2D toy dataset using NumPy's `loadtxt` function.

2. **Running the K-means Algorithm**: We run the K-means algorithm on this data using the implementation provided in `kmeans.py`. The `init` function from the `common` module is used to initialize K-means with different values of K and seeds. We select the seed that minimizes the total cost and save the associated plots for each K value.

3. **Reporting the Lowest Cost**: We report the lowest cost for each K value based on the results obtained from K-means clustering.

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
