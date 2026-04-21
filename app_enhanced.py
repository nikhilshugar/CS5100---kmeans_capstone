import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Rectangle
import time

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
from experiments.scrambled_independence import generate_hollow_square

st.set_page_config(page_title="K-Means Capstone", layout="wide")

st.title("Reproducing MacQueen's (1967) K-Means")

st.sidebar.header("Configuration")

category = st.sidebar.radio("Category", [
    "Home",
    "Paper Experiments",
    "Visual Analysis",
    "Dashboard",
])

if category == "Paper Experiments":
    experiment = st.sidebar.selectbox("Experiment", [
        "2D Blob Clustering",
        "Overlapping Blobs",
        "A/B Prediction (Section 3.2)",
        "Mixture-of-Normals (Section 3.3)",
        "Order Sensitivity (Section 3.1)",
        "C/R Coarsening & Refinement (Section 3.1)",
        "Scrambled Independence Test (Section 3.4)",
    ])
elif category == "Visual Analysis":
    experiment = st.sidebar.selectbox("View", [
        "Click-to-Cluster",
        "Step-by-Step Incremental",
        "Batch Iteration-by-Iteration",
        "Elbow Curve (Find Best K)",
        "Convergence Plot",
        "Cluster Size Histogram",
        "A/B Confusion Matrix",
        "Scrambled Before/After",
    ])
elif category == "Dashboard":
    experiment = "Dashboard"
else:
    experiment = "Home"

seed = st.sidebar.number_input("Random Seed", value=42, min_value=0, step=1)


def get_colors(k):
    return cm.tab10(np.linspace(0, 1, max(k, 10)))


def plot_clusters_2d(X, labels, means, title, ax):
    k = len(means)
    colors = get_colors(k)
    for c in range(k):
        mask = labels == c
        if np.any(mask):
            ax.scatter(X[mask, 0], X[mask, 1], c=[colors[c % 10]],
                       alpha=0.5, s=20, label=f"Cluster {c}")
    ax.scatter(means[:, 0], means[:, 1], c="black", marker="X",
               s=200, edgecolors="white", linewidths=1.5, zorder=5)
    ax.set_title(title)
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)

if experiment == "Home":
    st.header("About This Project")

    st.markdown("""
    This project is a reproduction of key algorithms and experiments from:

    > **MacQueen, J. (1967).** *Some Methods for Classification and Analysis of Multivariate Observations.*
    > Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability, pp. 281-297.

    This paper introduced k-means clustering. The idea is simple: given n data points and
    a number k, partition the data into k groups so that points in each group are close to
    their group's center. The metric used is within-class variation (WCV) — the average
    squared distance from each point to its cluster mean.
    """)

    st.subheader("What the Paper Covers")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Algorithms**
        - Incremental K-Means (Section 2.1) — processes points one at a time, updates means right away
        - C/R Similarity Grouping (Section 3.1) — clusters can merge or split so k adapts to the data
        - Batch Two-Step (Section 3.6) — assigns all points, recomputes means, repeats
        """)

    with col2:
        st.markdown("""
        **Experiments**
        - A/B Prediction (Section 3.2) — classifies 4D points as A or B, paper got ~87% accuracy
        - Mixture-of-Normals (Section 3.3) — predicts a continuous value using Gaussian regression, paper got SE=2.8 and 96% accuracy
        - Scrambled Test (Section 3.4) — detects variable relationships by scrambling columns, paper got 1.6x ratio
        """)

    st.subheader("Implementation Status")

    st.markdown("""
    | Component | Paper Section | Status |
    |---|---|---|
    | Incremental K-Means | 2.1 | Implemented |
    | Batch K-Means (comparison) | 3.6 | Implemented |
    | C/R Coarsening & Refinement | 3.1 | Implemented |
    | A/B Prediction Experiment | 3.2 | Reproduced |
    | Mixture-of-Normals Regression | 3.3 | Reproduced |
    | Scrambled Independence Test | 3.4 | Reproduced |
    | Order Sensitivity Analysis | 3.1 | Reproduced |
    """)

    st.subheader("Incremental vs Batch")

    st.markdown("""
    | | Incremental (MacQueen) | Batch (Lloyd) |
    |---|---|---|
    | Updates means | After every point | After all points assigned |
    | Passes through data | One pass | Multiple iterations |
    | Order dependent | Yes | No |
    | WCV | Depends on order | Usually lower |
    """)

    st.subheader("Navigation")

    st.markdown("""
    - **Paper Experiments** — run each experiment from the paper, tweak parameters
    - **Visual Analysis** — step-by-step animations, elbow curve, confusion matrix, click-to-cluster
    - **Dashboard** — all results on one page
    """)


elif experiment in ("2D Blob Clustering", "Overlapping Blobs"):
    col1, col2 = st.sidebar.columns(2)
    n_points = col1.number_input("N points", value=300, min_value=50, step=50)
    k = col2.number_input("K clusters", value=4, min_value=2, max_value=20)

    if experiment == "2D Blob Clustering":
        spread = st.sidebar.slider("Cluster spread", 0.5, 5.0, 1.0, 0.5)
        X, _, _ = generate_2d_blobs(n=n_points, k=k, spread=spread, seed=seed)
        st.header("2D Gaussian Blobs")
    else:
        spread = st.sidebar.slider("Cluster spread", 0.5, 8.0, 3.0, 0.5)
        X, _, _ = generate_overlapping_blobs(n=n_points, k=k, spread=spread, seed=seed)
        st.header("Overlapping Blobs")

    ikm = IncrementalKMeans(k=k, random_state=seed)
    ikm.fit(X)
    bkm = BatchKMeans(k=k, random_state=seed)
    bkm.fit(X)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    plot_clusters_2d(X, ikm.labels_, ikm.means,
                     f"Incremental (WCV={ikm.within_class_variation(X):.3f})", ax1)
    plot_clusters_2d(X, bkm.labels_, bkm.means,
                     f"Batch (WCV={bkm.within_class_variation(X):.3f})", ax2)
    plt.tight_layout()
    st.pyplot(fig)

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

elif experiment == "A/B Prediction (Section 3.2)":
    st.header("A/B Prediction (Paper Section 3.2)")
    st.markdown("250 training points in 4D, labeled A or B by two dimensions. "
                "Cluster each class with k=8, predict test points by nearest mean. Paper got ~87%.")

    k_ab = st.sidebar.number_input("K per class", value=8, min_value=2, max_value=20)
    seed_test = st.sidebar.number_input("Test seed", value=99, min_value=0, step=1)

    result = ab_prediction_experiment(seed_train=seed, seed_test=seed_test, k=k_ab)

    col1, col2 = st.columns(2)
    col1.metric("Incremental Accuracy",
                f"{result['incremental_accuracy']*100:.1f}%",
                delta=f"{(result['incremental_accuracy']-0.87)*100:+.1f}% vs paper's 87%")
    col2.metric("Batch Accuracy", f"{result['batch_accuracy']*100:.1f}%")

    st.write(f"Training split: {result['n_A_train']} A's, {result['n_B_train']} B's")

    preds = result["predictions_incremental"]
    y_test = result["y_test"]
    tp_a = np.sum((preds == "A") & (y_test == "A"))
    tp_b = np.sum((preds == "B") & (y_test == "B"))
    fp_a = np.sum((preds == "A") & (y_test == "B"))
    fp_b = np.sum((preds == "B") & (y_test == "A"))

    st.subheader("Prediction Breakdown")
    st.write(f"True A predicted A: {tp_a}, True B predicted B: {tp_b}")
    st.write(f"True B predicted A: {fp_a}, True A predicted B: {fp_b}")

elif experiment == "Mixture-of-Normals (Section 3.3)":
    st.header("Mixture-of-Normals (Paper Section 3.3)")
    st.markdown("Same A/B data but with a 5th dimension added (A=10, B=0). "
                "Cluster the 5D data with k=16, fit a Gaussian per cluster, "
                "predict the 5th dimension for new points. Paper got SE=2.8 and 96% accuracy.")

    k_mix = st.sidebar.number_input("K clusters", value=16, min_value=4, max_value=30)
    result = mixture_normals_prediction(seed_train=seed, seed_test=99, k=k_mix)

    col1, col2 = st.columns(2)
    col1.metric("Standard Error", f"{result['standard_error']:.2f}",
                delta=f"{result['standard_error']-2.8:+.2f} vs paper's 2.8")
    col2.metric("Accuracy", f"{result['accuracy']*100:.1f}%",
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

elif experiment == "Order Sensitivity (Section 3.1)":
    st.header("Order Sensitivity")
    st.markdown("Since incremental k-means depends on point order, this runs the same data "
                "multiple times with different shuffles and checks how much WCV changes. "
                "Paper says ~7% for 250pts in 5D with k=18.")

    n_runs = st.sidebar.slider("Number of runs", 3, 30, 10)
    k_os = st.sidebar.number_input("K clusters", value=4, min_value=2, max_value=20)
    n_pts = st.sidebar.number_input("N points", value=250, min_value=50, step=50)

    X_os, _, _ = generate_2d_blobs(n=n_pts, k=k_os, spread=1.5, seed=seed)
    results = order_sensitivity_experiment(X_os, k=k_os, n_runs=n_runs, seed_base=seed)

    wcvs = [r["within_class_variation"] for r in results]

    col1, col2, col3 = st.columns(3)
    col1.metric("Mean WCV", f"{np.mean(wcvs):.4f}")
    col2.metric("Std WCV", f"{np.std(wcvs):.4f}")
    variation_pct = (max(wcvs) - min(wcvs)) / np.mean(wcvs) * 100
    col3.metric("Max Variation %", f"{variation_pct:.1f}%")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.bar(range(n_runs), wcvs, color="steelblue")
    ax1.axhline(np.mean(wcvs), color="red", linestyle="--", label="Mean")
    ax1.set_xlabel("Run")
    ax1.set_ylabel("WCV")
    ax1.set_title("WCV Across Random Orderings")
    ax1.legend()

    ax2.bar(range(n_runs), [r["fit_time"] * 1000 for r in results], color="coral")
    ax2.set_xlabel("Run")
    ax2.set_ylabel("Time (ms)")
    ax2.set_title("Runtime Across Runs")
    plt.tight_layout()
    st.pyplot(fig)

elif experiment == "C/R Coarsening & Refinement (Section 3.1)":
    st.header("C/R Coarsening & Refinement")

    k_init = st.sidebar.number_input("Initial K", value=6, min_value=2, max_value=20)
    C_val = st.sidebar.slider("C (coarsening)", 0.5, 10.0, 3.0, 0.5)
    R_val = st.sidebar.slider("R (refinement)", 1.0, 15.0, 8.0, 0.5)
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
    plot_clusters_2d(X_cr, cr.labels_, cr.means, f"C/R (final k={cr.k_final})", ax1)
    plot_clusters_2d(X_cr, ikm_cr.labels_, ikm_cr.means, f"Plain (k={k_init})", ax2)
    plt.tight_layout()
    st.pyplot(fig)

elif experiment == "Scrambled Independence Test (Section 3.4)":
    st.header("Scrambled Independence Test (Section 3.4)")
    st.markdown("Run k-means on the original data. Then scramble each column independently "
                "(destroys variable relationships but keeps the same values). Run k-means again. "
                "If the original WCV is lower, the variables were related.")

    test_type = st.sidebar.selectbox("Dataset", ["Hollow Square", "5D A/B Dataset"])
    n_scrambles = st.sidebar.slider("Number of scrambles", 5, 50, 19)

    if test_type == "Hollow Square":
        k_sc = st.sidebar.number_input("K clusters", value=12, min_value=2, max_value=30)
        X_hs = generate_hollow_square(n=150, seed=seed)
        result = scrambled_independence_test(X_hs, k=k_sc, n_scrambles=n_scrambles, seed=seed)

        col1, col2, col3 = st.columns(3)
        col1.metric("Original WCV", f"{result['wcv_original']:.2f}")
        col2.metric("Scrambled WCV", f"{result['wcv_scrambled_mean']:.2f}")
        col3.metric("Ratio", f"{result['ratio']:.2f}x",
                     delta=f"{result['ratio']-1.6:+.2f}x vs paper's 1.6x")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.scatter(X_hs[:, 0], X_hs[:, 1], s=15, alpha=0.6)
        ax1.set_title("Hollow Square")
        ax1.set_aspect("equal")
        ax1.grid(True, alpha=0.3)

        ax2.hist(result['wcv_scrambled_all'], bins=10, color="coral", alpha=0.7, edgecolor="black")
        ax2.axvline(result['wcv_original'], color="steelblue", linewidth=2, linestyle="--",
                     label=f"Original = {result['wcv_original']:.2f}")
        ax2.set_xlabel("WCV")
        ax2.set_title("Original vs Scrambled")
        ax2.legend()
        plt.tight_layout()
        st.pyplot(fig)
    else:
        results_5d, _ = paper_5d_experiment(seed=seed)
        for k_val in [6, 12, 18]:
            r = results_5d[k_val]
            paper_vals = {6: 1.40, 12: 1.55, 18: 1.39}
            st.subheader(f"k = {k_val}")
            col1, col2, col3 = st.columns(3)
            col1.metric("Original WCV", f"{r['wcv_original']:.2f}")
            col2.metric("Scrambled WCV", f"{r['wcv_scrambled_mean']:.2f}")
            col3.metric("Ratio", f"{r['ratio']:.2f}x",
                         delta=f"{r['ratio']-paper_vals[k_val]:+.2f}x vs paper's {paper_vals[k_val]}x")

elif experiment == "Click-to-Cluster":
    st.header("Click-to-Cluster")
    st.markdown("Add points using the controls below, then k-means runs on them.")
    k = st.sidebar.number_input("K clusters", value=2, min_value=2, max_value=10)
    algo_choice = st.sidebar.radio("Algorithm", ["Incremental", "Batch", "Both"])

    if "custom_points" not in st.session_state:
        st.session_state.custom_points = []

    col_add, col_presets = st.columns(2)

    with col_add:
        st.markdown("**Add a point:**")
        c1, c2, c3 = st.columns([2, 2, 1])
        new_x = c1.number_input("x", value=0.0, min_value=-10.0, max_value=10.0, step=0.5, key="nx")
        new_y = c2.number_input("y", value=0.0, min_value=-10.0, max_value=10.0, step=0.5, key="ny")
        if c3.button("Add"):
            st.session_state.custom_points.append([new_x, new_y])
            st.rerun()
    with col_presets:
        st.markdown("**Quick presets:**")
        if st.button("6-point hand trace"):
            st.session_state.custom_points = [
                [1, 1], [9, 9], [2, 1], [8, 8], [1, 2], [9, 8]
            ]
            st.rerun()
        if st.button("Borderline test (8 pts)"):
            st.session_state.custom_points = [
                [0, 0], [10, 0], [4, 0], [6, 0], [4.5, 0], [5.5, 0], [3, 0], [7, 0]
            ]
            st.rerun()
        if st.button("Three groups"):
            st.session_state.custom_points = [
                [1, 1], [2, 1], [1, 2], [2, 2],
                [8, 1], [9, 1], [8, 2], [9, 2],
                [5, 8], [6, 8], [5, 9], [6, 9],
            ]
            st.rerun()
        if st.button("Clear all"):
            st.session_state.custom_points = []
            st.rerun()
    points = st.session_state.custom_points
    n_points = len(points)

    if n_points > 0:
        st.write(f"**{n_points} points:** {[[round(p[0],1), round(p[1],1)] for p in points]}")

    if n_points >= k:
        X_user = np.array(points)
        if algo_choice == "Both":
            ikm = IncrementalKMeans(k=k, random_state=seed)
            ikm.fit(X_user)
            bkm = BatchKMeans(k=k, random_state=seed)
            bkm.fit(X_user)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            plot_clusters_2d(X_user, ikm.labels_, ikm.means,
                             f"Incremental (WCV={ikm.within_class_variation(X_user):.3f})", ax1)
            plot_clusters_2d(X_user, bkm.labels_, bkm.means,
                             f"Batch (WCV={bkm.within_class_variation(X_user):.3f})", ax2)
            plt.tight_layout()
            st.pyplot(fig)
            col1, col2 = st.columns(2)
            col1.metric("Incremental WCV", f"{ikm.within_class_variation(X_user):.4f}")
            col2.metric("Batch WCV", f"{bkm.within_class_variation(X_user):.4f}")
        elif algo_choice == "Incremental":
            ikm = IncrementalKMeans(k=k, random_state=seed)
            ikm.fit(X_user)
            fig, ax = plt.subplots(figsize=(7, 5))
            plot_clusters_2d(X_user, ikm.labels_, ikm.means,
                             f"Incremental (WCV={ikm.within_class_variation(X_user):.3f})", ax)
            st.pyplot(fig)
            st.metric("WCV", f"{ikm.within_class_variation(X_user):.4f}")
        else:
            bkm = BatchKMeans(k=k, random_state=seed)
            bkm.fit(X_user)
            fig, ax = plt.subplots(figsize=(7, 5))
            plot_clusters_2d(X_user, bkm.labels_, bkm.means,
                             f"Batch (WCV={bkm.within_class_variation(X_user):.3f})", ax)
            st.pyplot(fig)
            st.metric("WCV", f"{bkm.within_class_variation(X_user):.4f}")
    elif n_points > 0:
        fig, ax = plt.subplots(figsize=(7, 5))
        X_temp = np.array(points)
        ax.scatter(X_temp[:, 0], X_temp[:, 1], c="steelblue", s=60)
        ax.set_xlim(-11, 11)
        ax.set_ylim(-11, 11)
        ax.set_title(f"Add {k - n_points} more points to cluster")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    else:
        st.write("Add some points or use a preset to get started.")
elif experiment == "Step-by-Step Incremental":
    st.header("Step-by-Step Incremental K-Means")
    st.markdown("Use the slider to step through points one at a time. "
                "Gray dots are unassigned, red dot is the current point being processed.")
    n_pts = st.sidebar.number_input("N points", value=30, min_value=6, max_value=100)
    k = st.sidebar.number_input("K clusters", value=3, min_value=2, max_value=6)
    spread = st.sidebar.slider("Spread", 0.5, 3.0, 1.0, 0.5)
    X, _, _ = generate_2d_blobs(n=n_pts, k=k, spread=spread, seed=seed)
    step = st.slider("Step (point being processed)", 0, n_pts - 1, n_pts - 1)
    means = X[:k].copy().astype(float)
    weights = np.ones(k, dtype=int)
    labels = np.full(n_pts, -1, dtype=int)
    for i in range(k):
        labels[i] = i
    for idx in range(k, min(step + 1, n_pts)):
        point = X[idx]
        dists = np.sum((means - point) ** 2, axis=1)
        nearest = int(np.argmin(dists))
        labels[idx] = nearest
        w = weights[nearest]
        means[nearest] = (means[nearest] * w + point) / (w + 1)
        weights[nearest] += 1
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = get_colors(k)
    unassigned = labels == -1
    if np.any(unassigned):
        ax.scatter(X[unassigned, 0], X[unassigned, 1], c="lightgray",
                   s=20, alpha=0.4, label="Unassigned")
    for c in range(k):
        mask = labels == c
        if np.any(mask):
            ax.scatter(X[mask, 0], X[mask, 1], c=[colors[c % 10]],
                       s=30, alpha=0.6, label=f"Cluster {c}")
    ax.scatter(means[:, 0], means[:, 1], c="black", marker="X",
               s=250, edgecolors="white", linewidths=2, zorder=5)
    if step >= k:
        ax.scatter(X[step, 0], X[step, 1], c="red", s=150,
                   edgecolors="black", linewidths=2, zorder=6, label="Current point")
    assigned_count = np.sum(labels != -1)
    ax.set_title(f"Step {step}: {assigned_count}/{n_pts} points assigned")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    st.write(f"**Current means:** {means.round(3).tolist()}")
    st.write(f"**Weights:** {weights.tolist()}")
    if step >= k:
        nearest_cluster = labels[step]
        dist_to_nearest = np.sqrt(np.sum((X[step] - means[nearest_cluster]) ** 2))
        st.write(f"**Point {step}** = {X[step].round(3)} → Cluster {nearest_cluster} "
                 f"(distance = {dist_to_nearest:.3f})")
elif experiment == "Batch Iteration-by-Iteration":
    st.header("Batch K-Means: Iteration by Iteration")
    st.markdown("Slide through iterations to see how assignments and means change. "
                "Red arrows show mean movement between iterations.")
    n_pts = st.sidebar.number_input("N points", value=100, min_value=20, max_value=500)
    k = st.sidebar.number_input("K clusters", value=4, min_value=2, max_value=8)
    spread = st.sidebar.slider("Spread", 0.5, 3.0, 1.5, 0.5)
    X, _, _ = generate_2d_blobs(n=n_pts, k=k, spread=spread, seed=seed)
    history_means = []
    history_labels = []
    history_wcv = []
    means = X[:k].copy().astype(float)
    history_means.append(means.copy())
    for iteration in range(50):
        diffs = X[:, np.newaxis, :] - means[np.newaxis, :, :]
        dists = np.sum(diffs ** 2, axis=2)
        labels = np.argmin(dists, axis=1)
        history_labels.append(labels.copy())
        wcv = 0.0
        for i in range(len(X)):
            wcv += np.sum((X[i] - means[labels[i]]) ** 2)
        wcv /= len(X)
        history_wcv.append(wcv)
        new_means = np.zeros_like(means)
        for c in range(k):
            mask = labels == c
            if np.any(mask):
                new_means[c] = X[mask].mean(axis=0)
            else:
                new_means[c] = means[c]
        shift = np.sum((new_means - means) ** 2)
        means = new_means
        history_means.append(means.copy())
        if shift < 1e-6:
            break
    total_iters = len(history_labels)
    iter_num = st.slider("Iteration", 0, total_iters - 1, total_iters - 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = get_colors(k)
    cur_labels = history_labels[iter_num]
    cur_means = history_means[iter_num + 1]
    for c in range(k):
        mask = cur_labels == c
        if np.any(mask):
            ax1.scatter(X[mask, 0], X[mask, 1], c=[colors[c % 10]], s=20, alpha=0.5)
    ax1.scatter(cur_means[:, 0], cur_means[:, 1], c="black", marker="X",
                s=200, edgecolors="white", linewidths=1.5, zorder=5)
    if iter_num > 0:
        prev_means = history_means[iter_num]
        for c in range(k):
            ax1.annotate("", xy=cur_means[c], xytext=prev_means[c],
                         arrowprops=dict(arrowstyle="->", color="red", lw=2))
    ax1.set_title(f"Iteration {iter_num + 1} (WCV = {history_wcv[iter_num]:.4f})")
    ax1.grid(True, alpha=0.3)
    ax2.plot(range(1, total_iters + 1), history_wcv, "o-", color="steelblue", linewidth=2)
    ax2.axvline(iter_num + 1, color="red", linestyle="--", alpha=0.7)
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("WCV")
    ax2.set_title("WCV Convergence")
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    st.write(f"Converged in **{total_iters}** iterations")
elif experiment == "Dashboard":
    st.header("All Experiments Dashboard")
    st.subheader("A/B Prediction")
    ab = ab_prediction_experiment(seed_train=seed, seed_test=99, k=8)
    col1, col2, col3 = st.columns(3)
    col1.metric("Incremental", f"{ab['incremental_accuracy']*100:.1f}%")
    col2.metric("Batch", f"{ab['batch_accuracy']*100:.1f}%")
    col3.metric("Paper", "~87%")
    st.subheader("Mixture-of-Normals")
    mix = mixture_normals_prediction(seed_train=seed, seed_test=99, k=16)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("SE", f"{mix['standard_error']:.2f}", delta=f"{mix['standard_error']-2.8:+.2f} vs 2.8")
    col2.metric("Accuracy", f"{mix['accuracy']*100:.1f}%", delta=f"{(mix['accuracy']-0.96)*100:+.1f}% vs 96%")
    col3.metric("Mean(A)", f"{mix['mean_pred_A']:.1f}", delta=f"{mix['mean_pred_A']-10.3:+.1f} vs 10.3")
    col4.metric("Mean(B)", f"{mix['mean_pred_B']:.1f}", delta=f"{mix['mean_pred_B']-1.3:+.1f} vs 1.3")
    st.subheader("Scrambled Independence Test")
    hs, _ = paper_hollow_square_experiment(seed=seed)
    col1, col2 = st.columns(2)
    col1.metric("Hollow Sq Ratio", f"{hs['ratio']:.2f}x", delta=f"{hs['ratio']-1.6:+.2f}x vs 1.6x")
    fived, _ = paper_5d_experiment(seed=seed)
    paper_vals = {6: 1.40, 12: 1.55, 18: 1.39}
    cols = st.columns(3)
    for i, k_val in enumerate([6, 12, 18]):
        cols[i].metric(f"5D k={k_val}", f"{fived[k_val]['ratio']:.2f}x",
                       delta=f"{fived[k_val]['ratio']-paper_vals[k_val]:+.2f}x vs {paper_vals[k_val]}x")
    st.subheader("2D Blobs: Incremental vs Batch")
    X_blob, _, _ = generate_2d_blobs(n=300, k=4, seed=seed)
    ikm = IncrementalKMeans(k=4, random_state=seed)
    ikm.fit(X_blob)
    bkm = BatchKMeans(k=4, random_state=seed)
    bkm.fit(X_blob)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    plot_clusters_2d(X_blob, ikm.labels_, ikm.means,
                     f"Incremental (WCV={ikm.within_class_variation(X_blob):.3f})", ax1)
    plot_clusters_2d(X_blob, bkm.labels_, bkm.means,
                     f"Batch (WCV={bkm.within_class_variation(X_blob):.3f})", ax2)
    plt.tight_layout()
    st.pyplot(fig)
elif experiment == "Elbow Curve (Find Best K)":
    st.header("Elbow Curve: Find the Best K")
    st.markdown("Run k-means for k=2 through k=20 and plot WCV vs k. Look for the 'elbow'.")
    n_pts = st.sidebar.number_input("N points", value=300, min_value=50, step=50)
    k_true = st.sidebar.number_input("True K (for data generation)", value=5, min_value=2, max_value=10)
    spread = st.sidebar.slider("Spread", 0.5, 3.0, 1.0, 0.5)
    max_k = st.sidebar.slider("Max K to test", 5, 30, 20)
    X, _, _ = generate_2d_blobs(n=n_pts, k=k_true, spread=spread, seed=seed)
    k_range = range(2, max_k + 1)
    inc_wcvs = []
    bat_wcvs = []
    for k_test in k_range:
        ikm = IncrementalKMeans(k=k_test, random_state=seed)
        ikm.fit(X)
        inc_wcvs.append(ikm.within_class_variation(X))

        bkm = BatchKMeans(k=k_test, random_state=seed)
        bkm.fit(X)
        bat_wcvs.append(bkm.within_class_variation(X))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(list(k_range), inc_wcvs, "o-", label="Incremental", color="steelblue", linewidth=2)
    ax.plot(list(k_range), bat_wcvs, "s-", label="Batch", color="coral", linewidth=2)
    ax.axvline(k_true, color="green", linestyle="--", linewidth=2,
               label=f"True K = {k_true}")
    ax.set_xlabel("K (number of clusters)")
    ax.set_ylabel("Within-Class Variation")
    ax.set_title("Elbow Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    st.markdown(f"Data generated with **{k_true}** true clusters. "
                f"The elbow (where the curve bends) suggests the right k.")
elif experiment == "Convergence Plot":
    st.header("Batch K-Means: WCV Convergence per Iteration")
    n_pts = st.sidebar.number_input("N points", value=300, min_value=50, step=50)
    k = st.sidebar.number_input("K clusters", value=4, min_value=2, max_value=20)
    spread = st.sidebar.slider("Spread", 0.5, 3.0, 1.5, 0.5)
    X, _, _ = generate_2d_blobs(n=n_pts, k=k, spread=spread, seed=seed)
    means = X[:k].copy().astype(float)
    wcv_history = []
    for iteration in range(100):
        diffs = X[:, np.newaxis, :] - means[np.newaxis, :, :]
        dists = np.sum(diffs ** 2, axis=2)
        labels = np.argmin(dists, axis=1)
        wcv = 0.0
        for i in range(len(X)):
            wcv += np.sum((X[i] - means[labels[i]]) ** 2)
        wcv /= len(X)
        wcv_history.append(wcv)
        new_means = np.zeros_like(means)
        for c in range(k):
            mask = labels == c
            if np.any(mask):
                new_means[c] = X[mask].mean(axis=0)
            else:
                new_means[c] = means[c]
        shift = np.sum((new_means - means) ** 2)
        means = new_means
        if shift < 1e-6:
            break
    ikm = IncrementalKMeans(k=k, random_state=seed)
    ikm.fit(X)
    inc_wcv = ikm.within_class_variation(X)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, len(wcv_history) + 1), wcv_history, "o-",
            color="coral", linewidth=2, markersize=8, label="Batch WCV per iteration")
    ax.axhline(inc_wcv, color="steelblue", linestyle="--", linewidth=2,
               label=f"Incremental WCV = {inc_wcv:.4f}")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Within-Class Variation")
    ax.set_title(f"Batch converged in {len(wcv_history)} iterations")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    st.write(f"Batch final WCV: **{wcv_history[-1]:.4f}**")
    st.write(f"Incremental WCV: **{inc_wcv:.4f}**")
elif experiment == "Cluster Size Histogram":
    st.header("Cluster Size Distribution: Incremental vs Batch")
    n_pts = st.sidebar.number_input("N points", value=300, min_value=50, step=50)
    k = st.sidebar.number_input("K clusters", value=6, min_value=2, max_value=20)
    spread = st.sidebar.slider("Spread", 0.5, 3.0, 1.5, 0.5)
    X, _, _ = generate_2d_blobs(n=n_pts, k=k, spread=spread, seed=seed)
    ikm = IncrementalKMeans(k=k, random_state=seed)
    ikm.fit(X)
    bkm = BatchKMeans(k=k, random_state=seed)
    bkm.fit(X)
    inc_sizes = [np.sum(ikm.labels_ == c) for c in range(k)]
    bat_sizes = [np.sum(bkm.labels_ == c) for c in range(k)]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    x_pos = np.arange(k)
    width = 0.35
    ax1.bar(x_pos - width/2, inc_sizes, width, label="Incremental", color="steelblue")
    ax1.bar(x_pos + width/2, bat_sizes, width, label="Batch", color="coral")
    ax1.set_xlabel("Cluster")
    ax1.set_ylabel("Number of Points")
    ax1.set_title("Points per Cluster")
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")
    inc_diags = ikm.cluster_diagnostics(X)
    bat_diags = bkm.cluster_diagnostics(X)
    inc_wcvs = [d["within_cluster_variance"] for d in inc_diags]
    bat_wcvs = [d["within_cluster_variance"] for d in bat_diags]
    ax2.bar(x_pos - width/2, inc_wcvs, width, label="Incremental", color="steelblue")
    ax2.bar(x_pos + width/2, bat_wcvs, width, label="Batch", color="coral")
    ax2.set_xlabel("Cluster")
    ax2.set_ylabel("Within-Cluster Variance")
    ax2.set_title("Variance per Cluster")
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    st.pyplot(fig)
    st.write(f"**Incremental**: sizes={inc_sizes}, total WCV={ikm.within_class_variation(X):.4f}")
    st.write(f"**Batch**: sizes={bat_sizes}, total WCV={bkm.within_class_variation(X):.4f}")
elif experiment == "A/B Confusion Matrix":
    st.header("A/B Prediction: Confusion Matrix")
    k_ab = st.sidebar.number_input("K per class", value=8, min_value=2, max_value=20)
    seed_test = st.sidebar.number_input("Test seed", value=99, min_value=0, step=1)
    result = ab_prediction_experiment(seed_train=seed, seed_test=seed_test, k=k_ab)
    preds = result["predictions_incremental"]
    y_test = result["y_test"]
    tp = np.sum((preds == "A") & (y_test == "A"))
    fn = np.sum((preds == "B") & (y_test == "A"))
    fp = np.sum((preds == "A") & (y_test == "B"))
    tn = np.sum((preds == "B") & (y_test == "B"))
    conf_matrix = np.array([[tp, fn], [fp, tn]])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    im = ax1.imshow(conf_matrix, cmap="Blues", aspect="auto")
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(["Predicted A", "Predicted B"])
    ax1.set_yticklabels(["Actual A", "Actual B"])
    ax1.set_title(f"Incremental (Accuracy: {result['incremental_accuracy']*100:.1f}%)")
    for i in range(2):
        for j in range(2):
            ax1.text(j, i, str(conf_matrix[i, j]), ha="center", va="center",
                     fontsize=24, fontweight="bold",
                     color="white" if conf_matrix[i, j] > conf_matrix.max()/2 else "black")
    X_test, y_test_data, (d1, d2) = generate_ab_dataset(n=250, seed=seed_test)
    correct = preds == y_test
    ax2.scatter(X_test[correct, d1], X_test[correct, d2], c="green", s=20, alpha=0.5, label="Correct")
    ax2.scatter(X_test[~correct, d1], X_test[~correct, d2], c="red", s=40, alpha=0.8,
                marker="x", label="Wrong")
    ax2.axhline(5.5, color="gray", linestyle="--", alpha=0.5)
    ax2.axvline(5.5, color="gray", linestyle="--", alpha=0.5)
    ax2.set_xlabel(f"Dimension {d1}")
    ax2.set_ylabel(f"Dimension {d2}")
    ax2.set_title("Correct vs Wrong Predictions (dims 0 & 1)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    col1, col2 = st.columns(2)
    col1.metric("Precision (A)", f"{tp/(tp+fp)*100:.1f}%" if (tp+fp) > 0 else "N/A")
    col2.metric("Recall (A)", f"{tp/(tp+fn)*100:.1f}%" if (tp+fn) > 0 else "N/A")
    col1.metric("Precision (B)", f"{tn/(tn+fn)*100:.1f}%" if (tn+fn) > 0 else "N/A")
    col2.metric("Recall (B)", f"{tn/(tn+fp)*100:.1f}%" if (tn+fp) > 0 else "N/A")
elif experiment == "Scrambled Before/After":
    st.header("Scrambled Independence Test: Before & After")
    dataset = st.sidebar.selectbox("Dataset", ["Hollow Square", "Random 2D Blobs"])
    if dataset == "Hollow Square":
        X_orig = generate_hollow_square(n=150, seed=seed)
    else:
        n_pts = st.sidebar.number_input("N points", value=200, min_value=50, step=50)
        spread = st.sidebar.slider("Spread", 0.5, 3.0, 1.5, 0.5)
        X_orig, _, _ = generate_2d_blobs(n=n_pts, k=4, spread=spread, seed=seed)
    k_sc = st.sidebar.number_input("K clusters", value=12, min_value=2, max_value=30)
    rng = np.random.RandomState(seed)
    X_scrambled = X_orig.copy()
    for col in range(X_scrambled.shape[1]):
        rng.shuffle(X_scrambled[:, col])
    km_orig = IncrementalKMeans(k=k_sc, random_state=seed)
    km_orig.fit(X_orig)
    km_scram = IncrementalKMeans(k=k_sc, random_state=seed)
    km_scram.fit(X_scrambled)
    wcv_orig = km_orig.within_class_variation(X_orig)
    wcv_scram = km_scram.within_class_variation(X_scrambled)
    col1, col2, col3 = st.columns(3)
    col1.metric("Original WCV", f"{wcv_orig:.2f}")
    col2.metric("Scrambled WCV", f"{wcv_scram:.2f}")
    col3.metric("Ratio", f"{wcv_scram/wcv_orig:.2f}x")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].scatter(X_orig[:, 0], X_orig[:, 1], s=15, alpha=0.6, c="steelblue")
    axes[0, 0].set_title("Original Data")
    axes[0, 0].set_aspect("equal")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 1].scatter(X_scrambled[:, 0], X_scrambled[:, 1], s=15, alpha=0.6, c="coral")
    axes[0, 1].set_title("Scrambled Data")
    axes[0, 1].set_aspect("equal")
    axes[0, 1].grid(True, alpha=0.3)
    colors = get_colors(k_sc)
    for c in range(k_sc):
        mask = km_orig.labels_ == c
        if np.any(mask):
            axes[1, 0].scatter(X_orig[mask, 0], X_orig[mask, 1],
                               c=[colors[c % 10]], s=15, alpha=0.5)
    axes[1, 0].scatter(km_orig.means[:, 0], km_orig.means[:, 1],
                       c="black", marker="X", s=100, edgecolors="white", zorder=5)
    axes[1, 0].set_title(f"Original Clustered (WCV={wcv_orig:.2f})")
    axes[1, 0].set_aspect("equal")
    axes[1, 0].grid(True, alpha=0.3)
    for c in range(k_sc):
        mask = km_scram.labels_ == c
        if np.any(mask):
            axes[1, 1].scatter(X_scrambled[mask, 0], X_scrambled[mask, 1],
                               c=[colors[c % 10]], s=15, alpha=0.5)
    axes[1, 1].scatter(km_scram.means[:, 0], km_scram.means[:, 1],
                       c="black", marker="X", s=100, edgecolors="white", zorder=5)
    axes[1, 1].set_title(f"Scrambled Clustered (WCV={wcv_scram:.2f})")
    axes[1, 1].set_aspect("equal")
    axes[1, 1].grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    st.markdown(f"Scrambling increased WCV by **{wcv_scram/wcv_orig:.2f}x** — "
                f"higher ratio means the original variables were more related.")
st.markdown("---")
st.markdown("""
**Reference**: MacQueen, J. (1967). *Some Methods for Classification and Analysis
of Multivariate Observations.* Proceedings of the Fifth Berkeley Symposium, pp. 281-297.
""")