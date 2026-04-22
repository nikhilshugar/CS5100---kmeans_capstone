import numpy as np
import time

class IncrementalKMeans:
    def __init__(self, k, random_state=None):
        self.k = k
        self.rng = np.random.RandomState(random_state)
        self.means = None
        self.weights = None
        self.labels_ = None
        self.fit_time_ = 0.0

    def _nearest_mean(self, point):
        dists = np.sum((self.means - point) ** 2, axis=1)
        return int(np.argmin(dists))

    def fit(self, X):
        start = time.time()
        n, d = X.shape
        assert n >= self.k, "Need at least k data points"
        # assign initial k points as intial means
        self.means = X[:self.k].copy().astype(float)
        self.weights = np.ones(self.k, dtype=int)
        self.labels_ = np.full(n, -1, dtype=int)
        # loop through each remaining data point
        for i in range(self.k):
            self.labels_[i] = i
        for idx in range(self.k, n):
            point = X[idx]
            # finding the nearest closest mean using the squared euclidean distance
            nearest = self._nearest_mean(point)
            self.labels_[idx] = nearest
            w = self.weights[nearest]
            # update the mean right away
            self.means[nearest] = (self.means[nearest] * w + point) / (w + 1)
            # increasing the weight as the data point is added.
            self.weights[nearest] += 1
        self.fit_time_ = time.time() - start
        return self

    def predict(self, X):
        return np.array([self._nearest_mean(x) for x in X])

    def within_class_variation(self, X):
        total = 0.0
        for i in range(len(X)):
            c = self.labels_[i]
            total += np.sum((X[i] - self.means[c]) ** 2)
        return total / len(X)

    def cluster_diagnostics(self, X):
        diags = []
        for c in range(self.k):
            mask = self.labels_ == c
            pts = X[mask]
            size = len(pts)
            mean = self.means[c]
            wcv = np.mean(np.sum((pts - mean) ** 2, axis=1)) if size > 0 else 0.0
            diags.append({
                "cluster": c,
                "size": size,
                "mean": mean.copy(),
                "within_cluster_variance": wcv,
            })
        return diags