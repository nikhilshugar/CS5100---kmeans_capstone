import numpy as np

class CRIncrementalKMeans:

    def __init__(self, k_init, C, R, random_state=None):
        self.k_init = k_init
        self.C = C
        self.R = R
        self.rng = np.random.RandomState(random_state)
        self.means = None
        self.weights = None
        self.labels_ = None
        self.k_final = None

    def _nearest_mean_idx(self, point):
        dists = np.sum((self.means - point) ** 2, axis=1)
        idx = int(np.argmin(dists))
        return idx, np.sqrt(dists[idx])

    def _merge_close_means(self):
        while len(self.means) > 1:
            min_dist = np.inf
            merge_i, merge_j = -1, -1
            n_means = len(self.means)
            for i in range(n_means):
                for j in range(i + 1, n_means):
                    d = np.sqrt(np.sum((self.means[i] - self.means[j]) ** 2))
                    if d < min_dist:
                        min_dist = d
                        merge_i, merge_j = i, j
            if min_dist < self.C:
                wi, wj = self.weights[merge_i], self.weights[merge_j]
                self.means[merge_i] = (
                    self.means[merge_i] * wi + self.means[merge_j] * wj
                ) / (wi + wj)
                self.weights[merge_i] = wi + wj
                self.means = np.delete(self.means, merge_j, axis=0)
                self.weights = np.delete(self.weights, merge_j)
            else:
                break

    def fit(self, X):
        n, d = X.shape
        k = min(self.k_init, n)
        self.means = X[:k].copy().astype(float)
        self.weights = np.ones(k, dtype=int)
        self._merge_close_means()
        for idx in range(k, n):
            point = X[idx]
            nearest, dist = self._nearest_mean_idx(point)
            if dist > self.R:
                self.means = np.vstack([self.means, point.astype(float)])
                self.weights = np.append(self.weights, 1)
            else:
                w = self.weights[nearest]
                self.means[nearest] = (self.means[nearest] * w + point) / (w + 1)
                self.weights[nearest] += 1
            self._merge_close_means()
        self.labels_ = np.array([self._nearest_mean_idx(x)[0] for x in X])
        self.k_final = len(self.means)
        return self

    def within_class_variation(self, X):
        total = 0.0
        for i in range(len(X)):
            c = self.labels_[i]
            total += np.sum((X[i] - self.means[c]) ** 2)
        return total / len(X)