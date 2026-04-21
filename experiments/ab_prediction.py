import numpy as np
from algorithms import IncrementalKMeans, BatchKMeans
from data import generate_ab_dataset


def ab_prediction_experiment(seed_train=42, seed_test=99, k=8):
    X_train, y_train, sel_dims = generate_ab_dataset(n=250, seed=seed_train)
    X_test, y_test, _ = generate_ab_dataset(n=250, seed=seed_test)
    X_a = X_train[y_train == "A"]
    X_b = X_train[y_train == "B"]
    km_a = IncrementalKMeans(k=k, random_state=seed_train)
    km_a.fit(X_a)
    km_b = IncrementalKMeans(k=k, random_state=seed_train)
    km_b.fit(X_b)
    all_means = np.vstack([km_a.means, km_b.means])
    all_labels_map = ["A"] * k + ["B"] * k
    correct = 0
    predictions = []
    for i in range(len(X_test)):
        dists = np.sum((all_means - X_test[i]) ** 2, axis=1)
        nearest = int(np.argmin(dists))
        pred = all_labels_map[nearest]
        predictions.append(pred)
        if pred == y_test[i]:
            correct += 1
    accuracy_inc = correct / len(X_test)
    bkm_a = BatchKMeans(k=k, random_state=seed_train)
    bkm_a.fit(X_a)
    bkm_b = BatchKMeans(k=k, random_state=seed_train)
    bkm_b.fit(X_b)
    all_means_batch = np.vstack([bkm_a.means, bkm_b.means])
    correct_batch = 0
    for i in range(len(X_test)):
        dists = np.sum((all_means_batch - X_test[i]) ** 2, axis=1)
        nearest = int(np.argmin(dists))
        pred = all_labels_map[nearest]
        if pred == y_test[i]:
            correct_batch += 1
    accuracy_batch = correct_batch / len(X_test)
    return {
        "incremental_accuracy": accuracy_inc,
        "batch_accuracy": accuracy_batch,
        "n_A_train": len(X_a),
        "n_B_train": len(X_b),
        "n_test": len(X_test),
        "selected_dims": sel_dims,
        "predictions_incremental": np.array(predictions),
        "y_test": y_test,
    }