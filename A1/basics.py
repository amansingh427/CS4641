import numpy as np

class Basics:
    def __init__(self):
        pass

    # =========================================
    # 6.1 Numpy array broadcasting
    # =========================================

    def center_n_scale(X):
        """
        Input:
            X: N x M numpy array, in which the rows correspond to data points and the
            columns correspond to features
        Output:
            Y: N x M numpy array, corresponding to a mean centered and scaled A
            array.
        """
        row, col = X.shape
        mean = X.mean(a, axis=1)
        Y = X - mean
        return Y


    # =========================================
    # 6.2 Elementwise and matrix multiplication
    # =========================================

    def element_vs_matrix(X, Y, v):
        """
        Input:
            X: N x M numpy array
            Y: M x P numpy array
            v: M dimensional numpy array
        Output:
            Z: N x P numpy array, corresponding to the matrix multiplication XY
            W: N x M numpy array, resulting of the element wise multiplication
            between X and v using array broadcasting
        """
        Z = np.multiply(X,Y)
        W = np.dot(X,v)
        return Z,W


    # =========================================
    # 6.3 Numpy indexing
    # =========================================

    def larger_than(X,t):
        """
        Input:
            X: N x M numpy array, containing random integer values
            t: integer, threshold value
        Output:
            Y: N x M numpy array, corresponding to array X in which values larger
            than the threshold were set to 100.
        """
        Y = X
        b = Y > t
        Y[b] = 100
        return Y
        


    # =========================================
    # 6.4 Numpy.where
    # =========================================

    def where(X,t):
        """
        Input:
            X: N x M numpy array, containing random integer values
            t: integer, threshold value
        Output:
            I: numpy array, corresponding to indices in the first row of X that
            are smaller than the threshold t.
            Y: N x M numpy array, corresponding to array X in which values smaller
            than the threshold were set to -500.
        """
        raise NotImplementedError
