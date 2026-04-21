import numpy as np

def generate_2d_blobs(n=300, k=4, spread=1.0, seed=42):
    rng = np.random.RandomState(seed)
    centers = rng.uniform(-10, 10, size=(k, 2))
    labels_true = rng.randint(0, k, size=n)
    X = np.array([centers[l] + rng.randn(2) * spread for l in labels_true])
    return X, labels_true, centers

def generate_overlapping_blobs(n=400, k=4, spread=3.0, seed=42):
    rng = np.random.RandomState(seed)
    centers = rng.uniform(-5, 5, size=(k, 2))
    labels_true = rng.randint(0, k, size=n)
    X = np.array([centers[l] + rng.randn(2) * spread for l in labels_true])
    return X, labels_true, centers

def generate_ab_dataset(n=250, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.randint(1, 11, size=(n, 4)).astype(float)
    d1, d2 = 0, 1
    labels = []
    for i in range(n):
        high1 = X[i, d1] > 5
        high2 = X[i, d2] > 5
        if high1 == high2:
            labels.append("A")
        else:
            labels.append("B")
    return X, np.array(labels), (d1, d2)