import numpy as np

class Regression(object):
    
    def __init__(self):
        pass
    
    def rmse(self, pred, label):
    
        return np.sqrt(np.mean(np.square(pred - label)))
    
    
    def construct_polynomial_feats(self, x, degree):
    
        return np.power(x.reshape(len(x), 1), np.arange(degree+1))


    def predict(self, xtest, weight):
    
        return np.dot(xtest, weight)


    def ridge_fit_closed(self, xtrain, ytrain, c_lambda):
    
        lambda_matrix = c_lambda * np.identity(xtrain.shape[1])
        lambda_matrix[:,0] = 0
        return np.linalg.pinv(xtrain.T @ xtrain + lambda_matrix) @ xtrain.T @ ytrain

        
    def ridge_fit_GD(self, xtrain, ytrain, c_lambda, epochs=500, learning_rate=1e-7):
        
        N, D = xtrain.shape
        weight = np.zeros((D, 1))
        for i in range(epochs):
            weight += (learning_rate / N) * (xtrain.T @ (ytrain - xtrain @ weight) - c_lambda * weight)
        return weight
        

    def ridge_fit_SGD(self, xtrain, ytrain, c_lambda, epochs=100, learning_rate=0.001):
    
        N, D = xtrain.shape
        weight = np.zeros((D, 1))
        for i in range(epochs):
            for j in range(N):
                weight += (learning_rate / N) * (xtrain.T @ (ytrain - xtrain @ weight) - c_lambda * weight)
        return weight