import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from algorithms import IncrementalKMeans, BatchKMeans, CRIncrementalKMeans
from data import generate_2d_blobs, generate_overlapping_blobs, generate_ab_dataset
from experiments import (
    ab_prediction_experiment,
    mixture_normals_prediction,
    order_sensitivity_experiment,
    paper_hollow_square_experiment,
    paper_5d_experiment,
    scrambled_independence_test,
)

st.set_page_config(page_title="K-Means Capstone", layout="wide")
st.title("Reproducing MacQueen's (1967) K-Means")
st.markdown("""
Reproducing the incremental k-means algorithm from *Some Methods for Classification
and Analysis of Multivariate Observations* by J. MacQueen (1967).
""")
st.sidebar.header("Configuration")
experiment = st.sidebar.selectbox("Select Experiment", [
    "1. 2D Blob Clustering",
    "2. Overlapping Blobs",
    "3. A/B Prediction (Section 3.2)",
    "4. Mixture-of-Normals (Section 3.3)",
    "5. Order Sensitivity",
    "6. C/R Coarsening & Refinement",
    "7. Scrambled Independence Test (Section 3.4)",
])

seed = st.sidebar.number_input("Random Seed", value=42, min_value=0, step=1)

def plot_clusters_2d(X, labels, means, title, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.figure
    k = len(means)
    colors = cm.tab10(np.linspace(0, 1, max(k, 10)))
    for c in range(k):
        mask = labels == c
        if np.any(mask):
            ax.scatter(X[mask, 0], X[mask, 1], c=[colors[c % 10]],
                       alpha=0.5, s=20, label=f"Cluster {c}")
    ax.scatter(means[:, 0], means[:, 1], c="black", marker="X",
               s=200, edgecolors="white", linewidths=1.5, zorder=5,
               label="Means")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    return fig
if experiment.startswith("1") or experiment.startswith("2"):
    col1, col2 = st.sidebar.columns(2)
    n_points = col1.number_input("N points", value=300, min_value=50, step=50)
    k = col2.number_input("K clusters", value=4, min_value=2, max_value=20)
    if experiment.startswith("1"):
        spread = st.sidebar.slider("Cluster spread", 0.5, 5.0, 1.0, 0.5)
        X, y_true, centers = generate_2d_blobs(n=n_points, k=k, spread=spread, seed=seed)
        st.header("Experiment 1: 2D Gaussian Blobs")
    else:
        spread = st.sidebar.slider("Cluster spread", 0.5, 8.0, 3.0, 0.5)
        X, y_true, centers = generate_overlapping_blobs(n=n_points, k=k, spread=spread, seed=seed)
        st.header("Experiment 2: Overlapping Blobs")
    ikm = IncrementalKMeans(k=k, random_state=seed)
    ikm.fit(X)
    bkm = BatchKMeans(k=k, random_state=seed)
    bkm.fit(X)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    plot_clusters_2d(X, ikm.labels_, ikm.means,
                     f"Incremental K-Means (WCV={ikm.within_class_variation(X):.3f})", ax1)
    plot_clusters_2d(X, bkm.labels_, bkm.means,
                     f"Batch K-Means (WCV={bkm.within_class_variation(X):.3f})", ax2)
    plt.tight_layout()
    st.pyplot(fig)
    st.subheader("Comparison Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Incremental WCV", f"{ikm.within_class_variation(X):.4f}")
    col2.metric("Batch WCV", f"{bkm.within_class_variation(X):.4f}")
    col3.metric("WCV Difference",
                f"{abs(ikm.within_class_variation(X) - bkm.within_class_variation(X)):.4f}")
    col1, col2, col3 = st.columns(3)
    col1.metric("Incremental Time", f"{ikm.fit_time_:.4f}s")
    col2.metric("Batch Time", f"{bkm.fit_time_:.4f}s")
    col3.metric("Batch Iterations", bkm.n_iter_)
    st.subheader("Cluster Diagnostics (Incremental)")
    diags = ikm.cluster_diagnostics(X)
    for d in diags:
        st.write(f"**Cluster {d['cluster']}**: size={d['size']}, "
                 f"WCV={d['within_cluster_variance']:.4f}, "
                 f"mean=({d['mean'][0]:.2f}, {d['mean'][1]:.2f})")
elif experiment.startswith("3"):
    st.header("Experiment 3: A/B Prediction (Paper Section 3.2)")
    st.markdown("""
    **Paper procedure**: 250 training 4D vectors (uniform integers 1-10), labeled A/B
    by two selected dimensions. Cluster A's and B's separately (k=8 each).
    Predict 250 new test points by nearest mean. **Paper reports ~87% accuracy.**
    """)
    k_ab = st.sidebar.number_input("K per class", value=8, min_value=2, max_value=20)
    seed_test = st.sidebar.number_input("Test seed", value=99, min_value=0, step=1)
    result = ab_prediction_experiment(seed_train=seed, seed_test=seed_test, k=k_ab)
    col1, col2 = st.columns(2)
    col1.metric("Incremental Accuracy",
                f"{result['incremental_accuracy']*100:.1f}%",
                delta=f"{(result['incremental_accuracy']-0.87)*100:+.1f}% vs paper's 87%")
    col2.metric("Batch Accuracy",
                f"{result['batch_accuracy']*100:.1f}%")
    st.write(f"Training split: {result['n_A_train']} A's, "
             f"{result['n_B_train']} B's")
    preds = result["predictions_incremental"]
    y_test = result["y_test"]
    tp_a = np.sum((preds == "A") & (y_test == "A"))
    tp_b = np.sum((preds == "B") & (y_test == "B"))
    fp_a = np.sum((preds == "A") & (y_test == "B"))
    fp_b = np.sum((preds == "B") & (y_test == "A"))
    st.subheader("Prediction Breakdown (Incremental)")
    st.write(f"True A predicted A: {tp_a}, True B predicted B: {tp_b}")
    st.write(f"True B predicted A: {fp_a}, True A predicted B: {fp_b}")
elif experiment.startswith("4"):
    st.header("Experiment 4: Mixture-of-Normals Prediction (Paper Section 3.3)")
    st.markdown("""
    **Paper procedure**: Add 5th dimension (A=10, B=0). Cluster combined 5D data (k=16).
    Fit normal per cluster, predict 5th dim via conditional regression.
    **Paper reports**: SE=2.8, 96% accuracy, mean(A)=10.3, mean(B)=1.3.
    """)
    k_mix = st.sidebar.number_input("K clusters", value=16, min_value=4, max_value=30)
    result = mixture_normals_prediction(seed_train=seed, seed_test=99, k=k_mix)
    col1, col2 = st.columns(2)
    col1.metric("Standard Error", f"{result['standard_error']:.2f}",
                delta=f"{result['standard_error']-2.8:+.2f} vs paper's 2.8")
    col2.metric("Classification Accuracy", f"{result['accuracy']*100:.1f}%",
                delta=f"{(result['accuracy']-0.96)*100:+.1f}% vs paper's 96%")
    col1, col2 = st.columns(2)
    col1.metric("Mean Prediction (A)", f"{result['mean_pred_A']:.1f}",
                delta=f"{result['mean_pred_A']-10.3:+.1f} vs paper's 10.3")
    col2.metric("Mean Prediction (B)", f"{result['mean_pred_B']:.1f}",
                delta=f"{result['mean_pred_B']-1.3:+.1f} vs paper's 1.3")
    fig, ax = plt.subplots(figsize=(8, 4))
    preds_a = result["predictions"][result["y_test"] == "A"]
    preds_b = result["predictions"][result["y_test"] == "B"]
    ax.hist(preds_a, bins=20, alpha=0.6, label="True A", color="blue")
    ax.hist(preds_b, bins=20, alpha=0.6, label="True B", color="red")
    ax.axvline(5, color="black", linestyle="--", label="Threshold=5")
    ax.set_xlabel("Predicted 5th Dimension")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Predictions by True Class")
    ax.legend()
    st.pyplot(fig)
elif experiment.startswith("5"):
    st.header("Experiment 5: Order Sensitivity")
    st.markdown("""
    Run incremental k-means multiple times with different random orderings of the
    same data. The paper notes that within-class variation changed by at most ~7%
    across 3 runs (250 pts, 5D, k=18).
    """)
    n_runs = st.sidebar.slider("Number of runs", 3, 30, 10)
    k_os = st.sidebar.number_input("K clusters", value=4, min_value=2, max_value=20)
    n_pts = st.sidebar.number_input("N points", value=250, min_value=50, step=50)
    X_os, _, _ = generate_2d_blobs(n=n_pts, k=k_os, spread=1.5, seed=seed)
    results = order_sensitivity_experiment(X_os, k=k_os, n_runs=n_runs, seed_base=seed)
    wcvs = [r["within_class_variation"] for r in results]
    times = [r["fit_time"] for r in results]
    col1, col2, col3 = st.columns(3)
    col1.metric("Mean WCV", f"{np.mean(wcvs):.4f}")
    col2.metric("Std WCV", f"{np.std(wcvs):.4f}")
    variation_pct = (max(wcvs) - min(wcvs)) / np.mean(wcvs) * 100
    col3.metric("Max Variation %", f"{variation_pct:.1f}%")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.bar(range(n_runs), wcvs, color="steelblue")
    ax1.axhline(np.mean(wcvs), color="red", linestyle="--", label="Mean")
    ax1.set_xlabel("Run")
    ax1.set_ylabel("Within-Class Variation")
    ax1.set_title("WCV Across Random Orderings")
    ax1.legend()
    ax2.bar(range(n_runs), [t * 1000 for t in times], color="coral")
    ax2.set_xlabel("Run")
    ax2.set_ylabel("Time (ms)")
    ax2.set_title("Runtime Across Runs")
    plt.tight_layout()
    st.pyplot(fig)
    st.subheader("Cluster Size Distribution per Run")
    for r in results:
        st.write(f"Run {r['run']}: sizes={r['cluster_sizes']}, "
                 f"WCV={r['within_class_variation']:.4f}")
elif experiment.startswith("6"):
    st.header("Experiment 6: Coarsening (C) & Refinement (R)")
    st.markdown("""
    The paper describes a modified k-means with two parameters:
    - **C (Coarsening)**: Merge means closer than C.
    - **R (Refinement)**: If a new point is farther than R from all means, seed a new cluster.
    Ordinarily C < R.
    """)
    k_init = st.sidebar.number_input("Initial K", value=6, min_value=2, max_value=20)
    C_val = st.sidebar.slider("C (coarsening threshold)", 0.5, 10.0, 3.0, 0.5)
    R_val = st.sidebar.slider("R (refinement threshold)", 1.0, 15.0, 8.0, 0.5)
    n_pts = st.sidebar.number_input("N points", value=300, min_value=50, step=50)
    X_cr, _, _ = generate_2d_blobs(n=n_pts, k=k_init, spread=1.5, seed=seed)
    cr = CRIncrementalKMeans(k_init=k_init, C=C_val, R=R_val, random_state=seed)
    cr.fit(X_cr)
    ikm_cr = IncrementalKMeans(k=k_init, random_state=seed)
    ikm_cr.fit(X_cr)
    col1, col2 = st.columns(2)
    col1.metric("C/R Final K", cr.k_final)
    col1.metric("C/R WCV", f"{cr.within_class_variation(X_cr):.4f}")
    col2.metric("Plain Incremental K", k_init)
    col2.metric("Plain WCV", f"{ikm_cr.within_class_variation(X_cr):.4f}")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    plot_clusters_2d(X_cr, cr.labels_, cr.means,
                     f"C/R K-Means (final k={cr.k_final})", ax1)
    plot_clusters_2d(X_cr, ikm_cr.labels_, ikm_cr.means,
                     f"Plain Incremental (k={k_init})", ax2)
    plt.tight_layout()
    st.pyplot(fig)
elif experiment.startswith("7"):
    st.header("Experiment 7: Scrambled Dimension Test (Paper Section 3.4)")
    st.markdown("""
    Tests whether variables in a dataset are related. Procedure:
    1. Run k-means on original data, record WCV.
    2. Scramble each column independently (destroys relationships).
    3. Run k-means on scrambled data, record WCV.
    4. If original WCV is much lower than scrambled, variables are related.
    **Paper reports**: 1.6x ratio for hollow square, 1.40-1.55x for 5D data.
    """)
    test_type = st.sidebar.selectbox("Test Dataset", [
        "Hollow Square (paper's 2D test)",
        "5D A/B Dataset (paper's 5D test)",
    ])
    if test_type.startswith("Hollow"):
        k_sc = st.sidebar.number_input("K clusters", value=12, min_value=2, max_value=30)
        n_scrambles = st.sidebar.slider("Number of scrambles", 5, 50, 19)
        from experiments.scrambled_independence import generate_hollow_square
        X_hs = generate_hollow_square(n=150, inner=60, outer=100, seed=seed)
        result = scrambled_independence_test(X_hs, k=k_sc, n_scrambles=n_scrambles, seed=seed)
        col1, col2, col3 = st.columns(3)
        col1.metric("Original WCV", f"{result['wcv_original']:.2f}")
        col2.metric("Scrambled WCV (mean)", f"{result['wcv_scrambled_mean']:.2f}")
        col3.metric("Ratio", f"{result['ratio']:.2f}x",
                     delta=f"{result['ratio']-1.6:+.2f}x vs paper's 1.6x")
        col1, col2 = st.columns(2)
        col1.metric("Rank", f"{result['rank']}/{result['total']}")
        col2.metric("p-value", f"{result['p_value']:.3f}")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.scatter(X_hs[:, 0], X_hs[:, 1], s=15, alpha=0.6, color="steelblue")
        ax1.set_title("Hollow Square Dataset (original)")
        ax1.set_aspect("equal")
        ax1.grid(True, alpha=0.3)
        ax2.hist(result['wcv_scrambled_all'], bins=10, color="coral", alpha=0.7,
                 edgecolor="black", label="Scrambled WCVs")
        ax2.axvline(result['wcv_original'], color="steelblue", linewidth=2,
                     linestyle="--", label=f"Original WCV = {result['wcv_original']:.2f}")
        ax2.set_xlabel("Within-Class Variation")
        ax2.set_ylabel("Count")
        ax2.set_title("Original vs Scrambled WCV Distribution")
        ax2.legend()
        plt.tight_layout()
        st.pyplot(fig)
    else:
        n_scrambles = st.sidebar.slider("Number of scrambles", 5, 50, 19)
        results_5d, X5 = paper_5d_experiment(seed=seed)
        for k_val in [6, 12, 18]:
            r = results_5d[k_val]
            paper_vals = {6: 1.40, 12: 1.55, 18: 1.39}
            st.subheader(f"k = {k_val}")
            col1, col2, col3 = st.columns(3)
            col1.metric("Original WCV", f"{r['wcv_original']:.2f}")
            col2.metric("Scrambled WCV (mean)", f"{r['wcv_scrambled_mean']:.2f}")
            col3.metric("Ratio", f"{r['ratio']:.2f}x",
                         delta=f"{r['ratio']-paper_vals[k_val]:+.2f}x vs paper's {paper_vals[k_val]}x")
st.markdown("---")
st.markdown("""
**Reference**: MacQueen, J. (1967). *Some Methods for Classification and Analysis
of Multivariate Observations.* Proceedings of the Fifth Berkeley Symposium on
Mathematical Statistics and Probability, pp. 281-297.
""")