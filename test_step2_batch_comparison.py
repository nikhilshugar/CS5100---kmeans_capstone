import numpy as np
import sys
sys.path.insert(0, ".")

from algorithms.incremental_kmeans import IncrementalKMeans
from algorithms.batch_kmeans import BatchKMeans
from data.generators import generate_2d_blobs

print("=" * 60)
print("TEST 1: Easy 6-point dataset (should match)")
print("=" * 60)

X_easy = np.array([
    [1.0, 1.0],
    [9.0, 9.0],
    [2.0, 1.0],
    [8.0, 8.0],
    [1.0, 2.0],
    [9.0, 8.0],
])

ikm = IncrementalKMeans(k=2, random_state=42)
ikm.fit(X_easy)

bkm = BatchKMeans(k=2, random_state=42)
bkm.fit(X_easy)

print(f"\nIncremental: labels={ikm.labels_}, WCV={ikm.within_class_variation(X_easy):.4f}")
print(f"      means: {ikm.means}")
print(f"Batch:       labels={bkm.labels_}, WCV={bkm.within_class_variation(X_easy):.4f}")
print(f"      means: {bkm.means}")
print(f"Batch converged in {bkm.n_iter_} iterations")
print(f"\nSame result? {np.allclose(ikm.means, bkm.means)}")


print("\n" + "=" * 60)
print("TEST 2: Borderline points (may differ)")
print("=" * 60)

X_tricky = np.array([
    [0.0, 0.0],
    [10.0, 0.0],
    [4.0, 0.0],
    [6.0, 0.0],
    [4.5, 0.0],
    [5.5, 0.0],
    [3.0, 0.0],
    [7.0, 0.0],
])

ikm2 = IncrementalKMeans(k=2, random_state=42)
ikm2.fit(X_tricky)

bkm2 = BatchKMeans(k=2, random_state=42)
bkm2.fit(X_tricky)

print(f"\nIncremental: labels={ikm2.labels_}")
print(f"      means: {ikm2.means.flatten()}")
print(f"      WCV:   {ikm2.within_class_variation(X_tricky):.4f}")

print(f"\nBatch:       labels={bkm2.labels_}")
print(f"      means: {bkm2.means.flatten()}")
print(f"      WCV:   {bkm2.within_class_variation(X_tricky):.4f}")
print(f"      iters: {bkm2.n_iter_}")

same = np.array_equal(ikm2.labels_, bkm2.labels_)
print(f"\nSame assignments? {same}")
if not same:
    diff_idx = np.where(ikm2.labels_ != bkm2.labels_)[0]
    print(f"Differences at indices: {diff_idx}")
    for i in diff_idx:
        print(f"  Point {X_tricky[i]}: incremental={ikm2.labels_[i]}, batch={bkm2.labels_[i]}")


print("\n" + "=" * 60)
print("TEST 3: 300 points, k=4 - WCV and runtime")
print("=" * 60)

X_large, _, _ = generate_2d_blobs(n=300, k=4, spread=1.0, seed=42)

ikm3 = IncrementalKMeans(k=4, random_state=42)
ikm3.fit(X_large)

bkm3 = BatchKMeans(k=4, random_state=42)
bkm3.fit(X_large)

print(f"\nIncremental: WCV={ikm3.within_class_variation(X_large):.4f}, "
      f"time={ikm3.fit_time_*1000:.1f}ms")
print(f"Batch:       WCV={bkm3.within_class_variation(X_large):.4f}, "
      f"time={bkm3.fit_time_*1000:.1f}ms, iters={bkm3.n_iter_}")

wcv_diff = ikm3.within_class_variation(X_large) - bkm3.within_class_variation(X_large)
print(f"\nWCV difference (inc - batch): {wcv_diff:.4f}")
if wcv_diff > 0:
    print("Batch has lower WCV (expected - it iterates to converge)")
elif wcv_diff < 0:
    print("Incremental has lower WCV (unusual but possible)")
else:
    print("Same WCV")

print("\nCluster sizes:")
for c in range(4):
    inc_size = np.sum(ikm3.labels_ == c)
    bat_size = np.sum(bkm3.labels_ == c)
    print(f"  Cluster {c}: incremental={inc_size}, batch={bat_size}")