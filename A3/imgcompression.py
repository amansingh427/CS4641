from matplotlib import pyplot as plt
import numpy as np

class ImgCompression(object):
    def __init__(self):
        pass

    def svd(self, X):
        """
        Do SVD. You could use numpy SVD.
        Your function should be able to handle black and white
        images (N*D arrays) as well as color images (N*D*3 arrays)
        In the image compression, we assume that each colum of the image is a feature. Image is the matrix X.
        Args:
            X: N * D array corresponding to an image (N*D*3 if color image)
        Return:
            U: N*N (*3 for color images)
            S: min(N, D)*1 (*3 for color images)
            V: D*D (*3 for color images)
        """
        if len(X.shape) < 3:
            return np.linalg.svd(X)
        U, S, V = np.linalg.svd(np.transpose(X, (2, 0, 1)))
        return np.transpose(U, (1, 2, 0)), np.transpose(S, (1, 0)), np.transpose(V, (1, 2, 0))



    def rebuild_svd(self, U, S, V, k):
        """
        Rebuild SVD by k componments.
        Args:
            U: N*N (*3 for color images)
            S: min(N, D)*1 (*3 for color images)
            V: D*D (*3 for color images)
            k: int corresponding to number of components
        Return:
            Xrebuild: N*D array of reconstructed image (N*D*3 if color image)

        Hint: numpy.matmul may be helpful for reconstructing color images
        """
        if len(U.shape) < 3:
            return np.dot(U[:, :k], np.dot(np.diag(S[:k]), V[:k]))
        S_new = np.zeros((3, k, k))
        diag = np.arange(k)
        S_new[:, diag, diag] = np.transpose(S, (1, 0))[:, :k]
        return np.transpose(np.transpose(U, (2, 0, 1))[:, :, :k] @ S_new @ np.transpose(V, (2, 0, 1))[:, :k], (1, 2, 0))




    def compression_ratio(self, X, k):
        """
        Compute compression of an image: (num stored values in original)/(num stored values in compressed)
        Args:
            X: N * D array corresponding to an image (N * D * 3 if color image)
            k: int corresponding to number of components
        Return:
            compression_ratio: float of proportion of storage used by compressed image
        """
        N, D = X.shape[0], X.shape[1]
        return (N * k + k + k * D) / N / D


    def recovered_variance_proportion(self, S, k):
        """
        Compute the proportion of the variance in the original matrix recovered by a rank-k approximation

        Args:
           S: min(N, D)*1 (*3 for color images) of singular values for the image
           k: int, rank of approximation
        Return:
           recovered_var: int (array of 3 ints for color image) corresponding to proportion of recovered variance
        """
        return sum(S[:k]**2) / sum(S**2)
