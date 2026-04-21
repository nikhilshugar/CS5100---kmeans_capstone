import numpy as np
import sys
sys.path.insert(0, ".")

from algorithms.incremental_kmeans import IncrementalKMeans

X = np.array([
    [1.0, 1.0],
    [9.0, 9.0],
    [2.0, 1.0],
    [8.0, 8.0],
    [1.0, 2.0],
    [9.0, 8.0],
])

km = IncrementalKMeans(k=2, random_state=42)
km.fit(X)

print("=== RESULTS ===")
print(f"Labels:  {km.labels_}")
print(f"Weights: {km.weights}")
print(f"Mean 0:  {km.means[0]}")
print(f"Mean 1:  {km.means[1]}")
print(f"WCV:     {km.within_class_variation(X):.4f}")

print("\n=== VERIFY BY HAND ===")
print("Expected labels: [0, 1, 0, 1, 0, 1]")
print("Expected mean 0: ~[1.333, 1.333]")
print("Expected mean 1: ~[8.667, 8.333]")

print("\n=== ORDER SENSITIVITY (swap first two points) ===")
X_swapped = X[[1, 0, 2, 3, 4, 5]]
km2 = IncrementalKMeans(k=2, random_state=42)
km2.fit(X_swapped)
print(f"Labels:  {km2.labels_}")
print(f"Mean 0:  {km2.means[0]}")
print(f"Mean 1:  {km2.means[1]}")
print("Notice: cluster indices flipped, but grouping is the same")

print("\n=== TIE BREAKING ===")
X_tie = np.array([
    [0.0, 0.0],
    [10.0, 0.0],
    [5.0, 0.0],
])
km3 = IncrementalKMeans(k=2, random_state=42)
km3.fit(X_tie)
print(f"Point [5,0] assigned to cluster: {km3.labels_[2]}")
print("Lower index (0) wins ties - this is MacQueen's convention")