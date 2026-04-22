import numpy as np

from algorithms import IncrementalKMeans, BatchKMeans, CRIncrementalKMeans
from data import generate_2d_blobs, generate_overlapping_blobs, generate_ab_dataset
from experiments import (
    ab_prediction_experiment,
    mixture_normals_prediction,
    order_sensitivity_experiment,
    paper_hollow_square_experiment,
    paper_5d_experiment,
)


def main():
    np.set_printoptions(precision=4, suppress=True)

    print("REPRODUCING MACQUEEN'S (1967) K-MEANS")

    print("\n2D Gaussian Blobs (k=4)")
    X_blob, _, _ = generate_2d_blobs(n=300, k=4, seed=42)

    ikm = IncrementalKMeans(k=4, random_state=42)
    ikm.fit(X_blob)
    print(f"Incremental: WCV={ikm.within_class_variation(X_blob):.4f}, time={ikm.fit_time_:.4f}s")

    bkm = BatchKMeans(k=4, random_state=42)
    bkm.fit(X_blob)
    print(f"Batch: WCV={bkm.within_class_variation(X_blob):.4f}, time={bkm.fit_time_:.4f}s, {bkm.n_iter_} iters")

    print("\nOrder Sensitivity (10 runs)")
    X_train_os, y_train_os, _ = generate_ab_dataset(n=250, seed=42)
    dim5_os = np.where(y_train_os == "A", 10.0, 0.0)
    X5_os = np.column_stack([X_train_os, dim5_os])
    sens = order_sensitivity_experiment(X5_os, k=18, n_runs=10)
    wcvs = [r["within_class_variation"] for r in sens]
    print(f"WCV range: [{min(wcvs):.4f}, {max(wcvs):.4f}]")
    print(f"Mean={np.mean(wcvs):.4f}, Std={np.std(wcvs):.4f}")
    max_var_pct = (max(wcvs) - min(wcvs)) / np.mean(wcvs) * 100
    print(f"Max variation: {max_var_pct:.1f}% (paper says ~7% for 250pts/5D/k=18)")

    print("\nA/B Prediction")
    ab = ab_prediction_experiment(seed_train=42, seed_test=99, k=8)
    print(f"{ab['n_A_train']} A's, {ab['n_B_train']} B's in training")
    print(f"Incremental accuracy: {ab['incremental_accuracy']*100:.1f}% (paper says ~87%)")
    print(f"Batch accuracy: {ab['batch_accuracy']*100:.1f}%")

    print("\nMixture-of-Normals")
    mix = mixture_normals_prediction(seed_train=42, seed_test=99, k=16)
    print(f"SE: {mix['standard_error']:.2f} (paper says 2.8)")
    print(f"Accuracy: {mix['accuracy']*100:.1f}% (paper says 96%)")
    print(f"Mean pred A: {mix['mean_pred_A']:.1f} (paper says 10.3)")
    print(f"Mean pred B: {mix['mean_pred_B']:.1f} (paper says 1.3)")

    print("\nOverlapping Blobs")
    X_ov, _, _ = generate_overlapping_blobs(n=400, k=4, spread=3.0, seed=42)
    ikm2 = IncrementalKMeans(k=4, random_state=42)
    ikm2.fit(X_ov)
    bkm2 = BatchKMeans(k=4, random_state=42)
    bkm2.fit(X_ov)
    print(f"Incremental WCV: {ikm2.within_class_variation(X_ov):.4f}")
    print(f"Batch WCV: {bkm2.within_class_variation(X_ov):.4f}")

    print("\nC/R Coarsening & Refinement")
    X_cr, _, _ = generate_2d_blobs(n=300, k=6, spread=1.5, seed=123)
    cr = CRIncrementalKMeans(k_init=6, C=3.0, R=8.0, random_state=123)
    cr.fit(X_cr)
    print(f"Started with k=6, ended with k={cr.k_final}")
    print(f"WCV: {cr.within_class_variation(X_cr):.4f}")

    print("\nScrambled Independence Test")
    print("Hollow square (150 pts, k=12):")
    hs_result, _ = paper_hollow_square_experiment(seed=42)
    print(f"  Original WCV: {hs_result['wcv_original']:.2f}")
    print(f"  Scrambled WCV: {hs_result['wcv_scrambled_mean']:.2f}")
    print(f"  Ratio: {hs_result['ratio']:.2f}x (paper says 1.6x)")
    print(f"  p-value: {hs_result['p_value']:.3f}")

    print("5D dataset:")
    fived_results, _ = paper_5d_experiment(seed=42)
    paper_ratios = {6: 1.40, 12: 1.55, 18: 1.39}
    for k_val in [6, 12, 18]:
        r = fived_results[k_val]
        print(f"  k={k_val}: {r['ratio']:.2f}x (paper says {paper_ratios[k_val]}x)")

    print("\nCluster Diagnostics (2D Blobs, Incremental)")
    diags = ikm.cluster_diagnostics(X_blob)
    for d in diags:
        print(f"  Cluster {d['cluster']}: {d['size']} pts, "
              f"WCV={d['within_cluster_variance']:.4f}, "
              f"mean={d['mean']}")

if __name__ == "__main__":
    main()