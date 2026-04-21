import numpy as np
from algorithms import IncrementalKMeans

def order_sensitivity_experiment(X, k=4, n_runs=10, seed_base=0):
    results = []
    for run in range(n_runs):
        rng = np.random.RandomState(seed_base + run)
        perm = rng.permutation(len(X))
        X_shuffled = X[perm]
        km = IncrementalKMeans(k=k, random_state=seed_base + run)
        km.fit(X_shuffled)
        wcv = km.within_class_variation(X_shuffled)
        sizes = [int(np.sum(km.labels_ == c)) for c in range(k)]
        results.append({
            "run": run,
            "within_class_variation": wcv,
            "fit_time": km.fit_time_,
            "cluster_sizes": sizes,
        })
    return results