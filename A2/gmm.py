import numpy as np

class GMM(object):
    def __init__(self): # No need to implement
        pass
    
    def softmax(self,logits):
    
        exp_stable_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        prob = exp_stable_logits / np.sum(exp_stable_logits, axis=1, keepdims=True)
        return prob

    def logsumexp(self,logits):

        max_in_row = np.amax(logits, axis = 1, keepdims = True)
        s = np.log(np.sum(np.exp(logits - max_in_row), axis = 1, keepdims = True)) + max_in_row
        return s

    def _init_components(self, points, K, **kwargs):

        #obtain the shape
        D = points.shape[1]
        # obtain min/max values
        
        # Initialize mixing coefficients pi
        pi = 1.0/K
        # Initialize center for each gaussian
        centers = points[np.random.choice(points.shape[0], K, replace=False)]
        # random initialization for mu from dataset
        cluster_idx = np.argmin(np.sqrt(np.sum(np.square(centers)[:,np.newaxis,:], axis=2) - 2 * centers.dot(np.transpose(points)) + np.sum(np.square(points), axis=1)), axis=0)
        mu = np.empty(centers.shape)
        for x in range(K):
            mu[x] = np.mean(points[cluster_idx == x], axis=0)
        # Initialize covariance
        sigma = np.array([np.cov(np.transpose(points[cluster_idx == i])) for i in range(K)]).reshape(K, D, D) 
        return pi,mu,sigma

    def _ll_joint(self, points, pi, mu, sigma, **kwargs):

        p_minus_mu = points[:,np.newaxis,:]-mu[np.newaxis,:,:]
        N, K, D = p_minus_mu.shape
        div = D * np.log(2 * np.pi) + np.linalg.slogdet(sigma)[1]
        l1 = - 0.5 * (div + np.sum(p_minus_mu*np.linalg.solve(sigma, p_minus_mu.reshape(*p_minus_mu.shape, 1)).reshape(N, K, -1), axis = 2))
        return l1

    def _E_step(self, points, pi, mu, sigma, **kwargs):

        shapes = self._ll_joint(points, pi, mu, sigma)
        gamma = self.softmax(shapes + np.log(pi))
        return gamma

    def _M_step(self, points, gamma, **kwargs):

        N, K = gamma.shape
        _, D = points.shape
        N_par = np.sum(gamma, axis = 0)
        pi = N_par / N
        mu = np.transpose(gamma).dot(points) / N_par[:, np.newaxis]
        p_minus_mu = points[:,np.newaxis,:]-mu[np.newaxis,:,:]
        sigma = np.zeros([K, D, D])
        for n in range(N):
            for k in range(K):
                sigma[k] += gamma[n, k] * np.outer(p_minus_mu[n,k], p_minus_mu[n,k])
        return pi, mu, sigma

    def __call__(self, points, K, max_iters=100, abs_tol=1e-16, rel_tol=1e-16, **kwargs):

        pi, mu, sigma = self._init_components(points, K, **kwargs)
        for it in range(0,max_iters):
            # E-step
            gamma = self._E_step(points, pi, mu, sigma)
            
            # M-step
            pi, mu, sigma = self._M_step(points, gamma)
            
            # calculate the negative log-likelihood of observation
            joint_ll = self._ll_joint(points, pi, mu, sigma)
            loss = -np.sum(self.logsumexp(joint_ll))
            if it:
                diff = np.abs(prev_loss - loss)
                if diff < abs_tol and diff / prev_loss < rel_tol:
                    break
            prev_loss = loss
            if it % 10 == 0:  print('iter %d, loss: %.4f' % (it, loss))
        return gamma, (pi, mu, sigma)
