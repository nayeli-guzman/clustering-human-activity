import numpy as np
from scipy.stats import multivariate_normal
from sklearn.datasets import make_spd_matrix
from time import time

class GMM:
    def __init__(self, n_clusters = 3, print_freq = 20, max_iter = 200, tol=1e-3):
        self.k = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.covs = None    # sigma (k x D x D)
        self.means = None   # mu (k x D)
        self.probs = None   # pi (k x 1)
        self.print_freq = print_freq
        self.log_ll = 0.0

    def log_likelihood(self, X):
        ll = [self.probs[k] * multivariate_normal.pdf(x=X, mean=self.means[k], cov=self.covs[k]) for k in range(self.k)]
        return np.sum(np.log(np.sum(ll, axis=0)))
        # return result

    def fit(self, X, centroids=None):
        # dimensions
        n = X.shape[0]
        d = X.shape[1]

        # randomly initialize parameters: sigmas, mus, pis
        if centroids is not None:
            self.means = centroids
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
            posteriors = np.array([self.probs[k] * multivariate_normal.pdf(x=X, mean=self.means[k], cov=self.covs[k]) for k in range(self.k)])
            posteriors /= np.sum(posteriors, axis=0)
            posteriors = posteriors.T

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
        self.log_ll = prev_ll
            
    
    def predict(self, X):
        predictions = np.array([self.probs[k] * multivariate_normal.pdf(x=X, mean=self.means[k], cov=self.covs[k]) for k in range(self.k)])
        return np.argmax(predictions, axis=0)
                    
    def fit_predict(self, X, means=None):
        self.fit(X, means)
        return self.predict(X)