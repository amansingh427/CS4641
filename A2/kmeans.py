import numpy as np
import matplotlib.pyplot as plt

class KMeans(object):

    def __init__(self): #No need to implement
        pass
    
    def pairwise_dist(self, x, y):
    
        return np.sqrt(np.sum(np.square(x)[:,np.newaxis,:], axis=2) - 2 * x.dot(np.transpose(y)) + np.sum(np.square(y), axis=1))

    def _init_centers(self, points, K, **kwargs):
    
        return points[np.random.choice(points.shape[0], K, replace=False)]

    def _update_assignment(self, centers, points):

        return np.argmin(self.pairwise_dist(centers, points), axis=0)

    def _update_centers(self, old_centers, cluster_idx, points):

        K = old_centers.shape[0]
        centers = np.empty(old_centers.shape)
        for x in range(K):
            centers[x] = np.mean(points[cluster_idx == x], axis=0)
        return centers
    
    def _get_loss(self, centers, cluster_idx, points):

        return np.sum(np.square(self.pairwise_dist(centers, points)[cluster_idx, np.arange(len(cluster_idx))]))
        
    def __call__(self, points, K, max_iters=100, abs_tol=1e-16, rel_tol=1e-16, verbose=False, **kwargs):

        centers = self._init_centers(points, K, **kwargs)
        for it in range(max_iters):
            assignments = self._update_assignment(centers, points)
            centers = self._update_centers(centers, assignments, points)
            loss = self._get_loss(centers, assignments, points)
            K = centers.shape[0]
            if it:
                diff = np.abs(prev_loss - loss)
                if diff < abs_tol and diff / prev_loss < rel_tol:
                    break
            prev_loss = loss
            if verbose:
                print('iter %d, loss: %.4f' % (it, loss))
        return assignments, centers, loss
    
    def find_optimal_num_clusters(self, data, max_K=15):

        losses = np.empty(max_K)
        _, img = data.shape
        flat_img = np.reshape(data, [-1, img]).astype(np.float32)
        for i in range(max_K):
            _, _, losses[i] = KMeans()(flat_img, i+1)
        plt.plot(np.arange(max_K) + 1, losses)
        plt.show()
        return losses
