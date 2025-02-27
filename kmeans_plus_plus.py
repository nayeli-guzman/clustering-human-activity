import numpy as np

class KMeans:
    def __init__(self, n_clusters, max_iter=300, random_state=None, tol=1e-5):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.centroids = None
        self.tol = tol
        self.clusters = None
        self.num_iter = 0

    def fit_predict(self, X):
        self.fit(X)  
        return self.predict(X) 
    
    def fit(self, X):

        centroids = self._init_centroids(X)

        self.num_iter=0

        while True:

            clusters = self._predict_clusters(X, centroids)
            last_centroids = centroids.copy()
            centroids = self._get_new_centroids(X, clusters)

            centroid_shift = np.linalg.norm(centroids - last_centroids)

            if centroid_shift <= self.tol:
                break
            
            self.num_iter+=1

        self.centroids = centroids

    def predict(self, X):
        if self.centroids is None:
            raise ValueError("El modelo no ha sido ajustado. Llama a fit primero.")
        
        return self._predict_clusters(X, self.centroids)
    
    def _init_centroids(self, X):
        if self.n_clusters > len(X):
            raise ValueError("amount of clusters is greater than amount of data. It cannot be possible")
        
        np.random.seed(self.random_state)
        centroids = [X[np.random.randint(len(X))]]
        
        for _ in range(1, self.n_clusters):
            dist_sq = np.min([np.linalg.norm(X - c, axis=1)**2 for c in centroids], axis=0)
            prob = dist_sq / dist_sq.sum()
            cumulative_prob = np.cumsum(prob)
            r = np.random.rand()
            for j, p in enumerate(cumulative_prob):
                if r < p:
                    centroids.append(X[j])
                    break
        
        return np.array(centroids)
        
    def _predict_clusters(self, X, centroids_init):
        
        size_data = len(X)

        clusters = []

        for i in range(size_data):
            distances = np.linalg.norm(X[i] - centroids_init, axis=1)
            cluster = np.argmin(distances)
            clusters.append(cluster)

        clusters = np.array(clusters)

        return clusters
                
    def _get_euclidean_distance(x1, x2):
        return np.linalg.norm(x1 - x2)

    def _get_new_centroids(self, X, clusters):

        centroids = []

        for k in range (self.n_clusters):
            data_k = X[clusters == k]

            if len(data_k) <= 0:
                centroids.append(X[np.random.randint(len(X))])
                pass

            centroid = np.mean(data_k, axis=0)
            centroids.append(centroid)


        centroids = np.array(centroids)

        return centroids

    def get_centroids(self):
            return self.centroids