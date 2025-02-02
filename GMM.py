import numpy as np
from scipy.stats import multivariate_normal
from sklearn.datasets import make_spd_matrix

"""
Possible optimizations?:
    - Use cupy
    - Stochastic variant of EM?
"""

class GMM:
    def __init__(self, n_clusters = 3, print_freq = 20, max_iter = 200, tol=1e-3):
        self.k = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.covs = None    # sigma (k x D x D)
        self.means = None   # mu (k x D)
        self.probs = None   # pi (k x 1)
        self.print_freq = print_freq

    def log_likelihood(self, X):
        n = X.shape[0]
        result = 0.0
        for i in range(n):
            ln_arg = [self.probs[k] * multivariate_normal.pdf(x=X[i], mean=self.means[k], cov=self.covs[k]) for k in range(self.k)]
            result += np.log( np.sum(ln_arg) )
        return result

    def fit(self, X, means=None):
        # dimensions
        n = X.shape[0]
        d = X.shape[1]

        # randomly initialize parameters: sigmas, mus, pis
        if means is not None:
            self.means = means
        else:
            self.means = np.random.normal(size=(self.k, d))
        self.covs = np.random.normal(size=(self.k, d, d))
        for i in range(self.k):
            self.covs[i] = make_spd_matrix(n_dim=d)         # symmetric positive definite :)
        self.probs = [1.0 / self.k for _ in range(self.k)]  # all equally probable
        posteriors = np.zeros(shape=(n, self.k))

        prev_ll = 0

        for it in range(self.max_iter):

            # EXPECTATION -------------------------------- ||
            # compute posteriors p(z|x)
            for i in range(n):
                # elements of gaussian mixture (posterior denominator)
                gm_n = [self.probs[j] * multivariate_normal.pdf(x=X[i], mean=self.means[j], cov=self.covs[j]) for j in range(self.k)]
                for k in range(self.k):
                    posterior_n_k = gm_n[k] / np.sum(gm_n)
                    posteriors[i][k] = posterior_n_k

            # MAXIMIZATION ------------------------------- ||
            # update mus, sigmas, pis

            # "number of elements assigned to each cluster"
            cluster_counts = np.sum(posteriors, axis=0) # k x 1

            # means (mu)
            self.means = posteriors.T @ X
            for k in range(self.k):
                self.means[k] /= cluster_counts[k]

            # covariances (sigma)
            self.covs = np.zeros((self.k, d, d))
            for k in range(self.k):
                x_centered = X - self.means[k]
                for i in range(n):
                    x_centered_t= x_centered[i].reshape(-1, 1)
                    self.covs[k] += posteriors[i][k] * (x_centered_t @ x_centered_t.T)
                self.covs[k] /= cluster_counts[k]

            # probabilities (weights??) (pi)
            self.probs = cluster_counts / np.sum(cluster_counts)

            # check likelihood convergence
            log_likelihood = self.log_likelihood(X=X)
            if it % self.print_freq == 0:
                print(f'log likelihood at iteration {it} : {log_likelihood}')
            if np.abs(log_likelihood - prev_ll) < self.tol:
                print("Converged at ", it, "iterations")
                break
            prev_ll = log_likelihood
            
    
    def predict(self, X):
        predictions = np.array([self.probs[k] * multivariate_normal.pdf(x=X, mean=self.means[k], cov=self.covs[k]) for k in range(self.k)])
        return np.argmax(predictions, axis=0)
                    
    def fit_predict(self, X, means=None):
        self.fit(X, means)
        return self.predict(X)