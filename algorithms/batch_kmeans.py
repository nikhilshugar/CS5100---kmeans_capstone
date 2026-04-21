import numpy as np
import time

class BatchKMeans:

    def __init__(self, k, max_iter=300, tol=1e-6, random_state=None):
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.rng = np.random.RandomState(random_state)
        self.means = None
        self.labels_ = None
        self.n_iter_ = 0
        self.fit_time_ = 0.0

    def _assign(self, X):
        diffs = X[:, np.newaxis, :] - self.means[np.newaxis, :, :]
        dists = np.sum(diffs ** 2, axis=2)
        return np.argmin(dists, axis=1)

    def fit(self, X):
        start = time.time()
        n, d = X.shape
        self.means = X[:self.k].copy().astype(float)
        self.labels_ = np.full(n, -1, dtype=int)
        for iteration in range(self.max_iter):
            new_labels = self._assign(X)
            new_means = np.zeros_like(self.means)
            for c in range(self.k):
                mask = new_labels == c
                if np.any(mask):
                    new_means[c] = X[mask].mean(axis=0)
                else:
                    new_means[c] = self.means[c]
            shift = np.sum((new_means - self.means) ** 2)
            self.means = new_means
            self.labels_ = new_labels
            self.n_iter_ = iteration + 1
            if shift < self.tol:
                break
        self.fit_time_ = time.time() - start
        return self

    def predict(self, X):
        diffs = X[:, np.newaxis, :] - self.means[np.newaxis, :, :]
        dists = np.sum(diffs ** 2, axis=2)
        return np.argmin(dists, axis=1)

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