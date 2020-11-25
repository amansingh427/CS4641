from matplotlib import pyplot as plt
import numpy as np


from imgcompression import ImgCompression



class Dimensionality(object):
    def __init__(self):
        pass

    def pca(self, X):
        """
        Decompose dataset into principal components. 
        You may use your SVD function from the previous part in your implementation.

        Args: 
            X: N x D array corresponding to a dataset, in which N is the number of points and D is the number of features
        Return:
            U: N x N 
            S: min(N, D) x 1 
            V: D x D
        """
        return ImgCompression.svd(self, X)


    def intrinsic_dimension(self, S, recovered_variance=.99):
        """
        Find the number of principal components necessary to recover given proportion of variance

        Args: 
            S: 1-d array corresponding to the singular values of a dataset

            recovered_varaiance: float in [0,1].  Minimum amount of variance 
                to recover from given principal components
        Return:
            dim: int, the number of principal components necessary to recover 
                the given proportion of the variance
        """
        for dim in range(0, len(S)):
            if ImgCompression.recovered_variance_proportion(self, S, dim) >= recovered_variance:
                return dim


    def num_linearly_ind_features(self, S, eps=1e-11):
        """
        Find the number of linearly independent features in dataset

        Args: 
            S: 1-d array corresponding to the singular values of a dataset
        Return:
            dim: int, the number of linearly independent dimensions in our data
        """
        return len(S[S >= eps])

