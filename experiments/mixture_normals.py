import numpy as np
from algorithms import IncrementalKMeans
from data import generate_ab_dataset

def _gaussian_log_likelihood(x, mu, cov_inv, cov_log_det):
    diff = x - mu
    return -0.5 * (cov_log_det + diff @ cov_inv @ diff)

def mixture_normals_prediction(seed_train=42, seed_test=99, k=16):
    X_train, y_train, _ = generate_ab_dataset(n=250, seed=seed_train)
    X_test, y_test, _ = generate_ab_dataset(n=250, seed=seed_test)
    dim5_train = np.where(y_train == "A", 10.0, 0.0)
    X5_train = np.column_stack([X_train, dim5_train])
    dim5_test = np.where(y_test == "A", 10.0, 0.0)
    km = IncrementalKMeans(k=k, random_state=seed_train)
    km.fit(X5_train)
    cluster_means = []
    cluster_covs = []
    cluster_cov_invs = []
    cluster_cov_log_dets = []
    cluster_weights = []
    valid_clusters = []
    for c in range(k):
        mask = km.labels_ == c
        pts = X5_train[mask]
        if len(pts) > 5:
            mu = np.mean(pts, axis=0)
            cov = np.cov(pts, rowvar=False) + 1e-4 * np.eye(5)
            try:
                cov_inv = np.linalg.inv(cov)
                sign, cov_log_det = np.linalg.slogdet(cov)
                if sign <= 0:
                    continue
                cluster_means.append(mu)
                cluster_covs.append(cov)
                cluster_cov_invs.append(cov_inv)
                cluster_cov_log_dets.append(cov_log_det)
                cluster_weights.append(len(pts))
                valid_clusters.append(c)
            except np.linalg.LinAlgError:
                continue
    n_valid = len(valid_clusters)
    total_w = sum(cluster_weights)
    cluster_priors = [w / total_w for w in cluster_weights]
    predictions = []
    for i in range(len(X_test)):
        x4 = X_test[i]
        log_responsibilities = np.zeros(n_valid)
        for c_idx in range(n_valid):
            mu = cluster_means[c_idx]
            cov = cluster_covs[c_idx]
            mu_a = mu[:4]
            cov_aa = cov[:4, :4] + 1e-4 * np.eye(4)
            try:
                cov_aa_inv = np.linalg.inv(cov_aa)
                sign, cov_aa_log_det = np.linalg.slogdet(cov_aa)
                if sign <= 0:
                    log_responsibilities[c_idx] = -np.inf
                    continue
                log_responsibilities[c_idx] = (
                    np.log(cluster_priors[c_idx] + 1e-300)
                    + _gaussian_log_likelihood(x4, mu_a, cov_aa_inv, cov_aa_log_det)
                )
            except np.linalg.LinAlgError:
                log_responsibilities[c_idx] = -np.inf
        max_log = np.max(log_responsibilities)
        if max_log == -np.inf:
            responsibilities = np.ones(n_valid) / n_valid
        else:
            log_responsibilities -= max_log
            responsibilities = np.exp(log_responsibilities)
            resp_sum = np.sum(responsibilities)
            if resp_sum > 0:
                responsibilities /= resp_sum
            else:
                responsibilities = np.ones(n_valid) / n_valid
        pred_val = 0.0
        for c_idx in range(n_valid):
            mu = cluster_means[c_idx]
            cov = cluster_covs[c_idx]
            mu_a = mu[:4]
            mu_b = mu[4]
            cov_aa = cov[:4, :4] + 1e-4 * np.eye(4)
            cov_ba = cov[4, :4]
            try:
                cov_aa_inv = np.linalg.inv(cov_aa)
                cond_mean = mu_b + cov_ba @ cov_aa_inv @ (x4 - mu_a)
            except np.linalg.LinAlgError:
                cond_mean = mu_b
            pred_val += responsibilities[c_idx] * cond_mean
        predictions.append(pred_val)
    predictions = np.array(predictions)
    se = np.sqrt(np.mean((predictions - dim5_test) ** 2))
    pred_labels = np.where(predictions > 5, "A", "B")
    accuracy = np.mean(pred_labels == y_test)
    mean_pred_a = np.mean(predictions[y_test == "A"])
    mean_pred_b = np.mean(predictions[y_test == "B"])
    return {
        "standard_error": se,
        "accuracy": accuracy,
        "mean_pred_A": mean_pred_a,
        "mean_pred_B": mean_pred_b,
        "predictions": predictions,
        "y_test": y_test,
    }