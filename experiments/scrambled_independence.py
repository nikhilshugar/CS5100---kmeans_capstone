import numpy as np
from algorithms import IncrementalKMeans

def scramble_dataset(X, rng):
    X_scrambled = X.copy()
    for col in range(X.shape[1]):
        rng.shuffle(X_scrambled[:, col])
    return X_scrambled

def scrambled_independence_test(X, k=12, n_scrambles=19, seed=42):
    rng = np.random.RandomState(seed)
    km_original = IncrementalKMeans(k=k, random_state=seed)
    km_original.fit(X)
    wcv_original = km_original.within_class_variation(X)
    scrambled_wcvs = []
    for i in range(n_scrambles):
        X_s = scramble_dataset(X, rng)
        km_s = IncrementalKMeans(k=k, random_state=seed + i + 1)
        km_s.fit(X_s)
        scrambled_wcvs.append(km_s.within_class_variation(X_s))
    scrambled_wcvs = np.array(scrambled_wcvs)
    mean_scrambled = np.mean(scrambled_wcvs)
    ratio = mean_scrambled / wcv_original if wcv_original > 0 else float('inf')
    rank = 1 + np.sum(scrambled_wcvs <= wcv_original)
    total = n_scrambles + 1
    p_value = rank / total
    return {
        "wcv_original": wcv_original,
        "wcv_scrambled_mean": mean_scrambled,
        "wcv_scrambled_std": np.std(scrambled_wcvs),
        "wcv_scrambled_all": scrambled_wcvs,
        "ratio": ratio,
        "rank": int(rank),
        "total": total,
        "p_value": p_value,
    }

def generate_hollow_square(n=150, inner=60, outer=100, seed=42):
    rng = np.random.RandomState(seed)
    points = []
    while len(points) < n:
        x = rng.uniform(-outer / 2, outer / 2)
        y = rng.uniform(-outer / 2, outer / 2)
        if abs(x) > inner / 2 or abs(y) > inner / 2:
            points.append([x, y])
    return np.array(points)

def paper_hollow_square_experiment(seed=42):
    X = generate_hollow_square(n=150, inner=60, outer=100, seed=seed)
    result = scrambled_independence_test(X, k=12, n_scrambles=19, seed=seed)
    return result, X

def paper_5d_experiment(seed=42):
    from data import generate_ab_dataset
    X_train, y_train, _ = generate_ab_dataset(n=250, seed=seed)
    dim5 = np.where(y_train == "A", 10.0, 0.0)
    X5 = np.column_stack([X_train, dim5])
    results = {}
    for k in [6, 12, 18]:
        results[k] = scrambled_independence_test(X5, k=k, n_scrambles=19, seed=seed)
    return results, X5