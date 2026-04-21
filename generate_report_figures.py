import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from algorithms import IncrementalKMeans, BatchKMeans, CRIncrementalKMeans
from data import generate_2d_blobs, generate_overlapping_blobs
from experiments import (
    ab_prediction_experiment,
    mixture_normals_prediction,
    order_sensitivity_experiment,
    paper_hollow_square_experiment,
    paper_5d_experiment,
)

os.makedirs("figures", exist_ok=True)

SEED = 42


def plot_2d(X, labels, means, title, filename):
    fig, ax = plt.subplots(figsize=(7, 5))
    k = len(means)
    colors = cm.tab10(np.linspace(0, 1, max(k, 10)))
    for c in range(k):
        mask = labels == c
        if np.any(mask):
            ax.scatter(X[mask, 0], X[mask, 1], c=[colors[c % 10]],
                       alpha=0.5, s=20, label=f"Cluster {c}")
    ax.scatter(means[:, 0], means[:, 1], c="black", marker="X",
               s=200, edgecolors="white", linewidths=1.5, zorder=5)
    ax.set_title(title)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(f"figures/{filename}", dpi=150)
    plt.close(fig)
    print(f"  Saved figures/{filename}")


print("Figure 1: 2D Blobs - Incremental vs Batch")
X, _, _ = generate_2d_blobs(n=300, k=4, seed=SEED)

ikm = IncrementalKMeans(k=4, random_state=SEED)
ikm.fit(X)
bkm = BatchKMeans(k=4, random_state=SEED)
bkm.fit(X)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
colors = cm.tab10(np.linspace(0, 1, 10))
for c in range(4):
    m1 = ikm.labels_ == c
    m2 = bkm.labels_ == c
    ax1.scatter(X[m1, 0], X[m1, 1], c=[colors[c]], alpha=0.5, s=20)
    ax2.scatter(X[m2, 0], X[m2, 1], c=[colors[c]], alpha=0.5, s=20)
ax1.scatter(ikm.means[:, 0], ikm.means[:, 1], c="black", marker="X", s=200,
            edgecolors="white", linewidths=1.5, zorder=5)
ax2.scatter(bkm.means[:, 0], bkm.means[:, 1], c="black", marker="X", s=200,
            edgecolors="white", linewidths=1.5, zorder=5)
ax1.set_title(f"Incremental K-Means (WCV={ikm.within_class_variation(X):.3f})")
ax2.set_title(f"Batch K-Means (WCV={bkm.within_class_variation(X):.3f})")
for ax in (ax1, ax2):
    ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig("figures/fig1_blobs_comparison.png", dpi=150)
plt.close()
print("  Saved figures/fig1_blobs_comparison.png")


print("Figure 2: Overlapping Blobs")
X_ov, _, _ = generate_overlapping_blobs(n=400, k=4, spread=3.0, seed=SEED)
ikm2 = IncrementalKMeans(k=4, random_state=SEED)
ikm2.fit(X_ov)
plot_2d(X_ov, ikm2.labels_, ikm2.means,
        f"Overlapping Blobs - Incremental (WCV={ikm2.within_class_variation(X_ov):.3f})",
        "fig2_overlapping.png")


print("Figure 3: Order Sensitivity")
from data import generate_ab_dataset
X_train_os, y_train_os, _ = generate_ab_dataset(n=250, seed=SEED)
dim5_os = np.where(y_train_os == "A", 10.0, 0.0)
X_os = np.column_stack([X_train_os, dim5_os])
sens = order_sensitivity_experiment(X_os, k=18, n_runs=10, seed_base=SEED)
wcvs = [r["within_class_variation"] for r in sens]

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(range(10), wcvs, color="steelblue", edgecolor="navy", alpha=0.8)
ax.axhline(np.mean(wcvs), color="red", linestyle="--", linewidth=1.5,
           label=f"Mean={np.mean(wcvs):.4f}")
ax.set_xlabel("Run (different random ordering)")
ax.set_ylabel("Within-Class Variation")
variation_pct = (max(wcvs) - min(wcvs)) / np.mean(wcvs) * 100
ax.set_title(f"Order Sensitivity (250pts, 5D, k=18): Max variation = {variation_pct:.1f}%")
ax.legend()
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
fig.savefig("figures/fig3_order_sensitivity.png", dpi=150)
plt.close()
print("  Saved figures/fig3_order_sensitivity.png")


print("Figure 4: A/B Prediction over multiple seeds")
inc_accs, bat_accs = [], []
seeds_test = list(range(50, 70))
for st_seed in seeds_test:
    r = ab_prediction_experiment(seed_train=SEED, seed_test=st_seed, k=8)
    inc_accs.append(r["incremental_accuracy"])
    bat_accs.append(r["batch_accuracy"])

fig, ax = plt.subplots(figsize=(9, 4))
x_pos = np.arange(len(seeds_test))
w = 0.35
ax.bar(x_pos - w/2, [a*100 for a in inc_accs], w, label="Incremental", color="steelblue")
ax.bar(x_pos + w/2, [a*100 for a in bat_accs], w, label="Batch", color="coral")
ax.axhline(87, color="green", linestyle="--", label="Paper's 87%")
ax.set_xlabel("Test Seed Index")
ax.set_ylabel("Accuracy (%)")
ax.set_title("A/B Prediction Accuracy Across Test Seeds")
ax.legend()
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
fig.savefig("figures/fig4_ab_accuracy.png", dpi=150)
plt.close()
print("  Saved figures/fig4_ab_accuracy.png")
print(f"  Incremental: mean={np.mean(inc_accs)*100:.1f}%, "
      f"std={np.std(inc_accs)*100:.1f}%")
print(f"  Batch: mean={np.mean(bat_accs)*100:.1f}%, "
      f"std={np.std(bat_accs)*100:.1f}%")


print("Figure 5: Mixture-of-Normals Prediction")
mix = mixture_normals_prediction(seed_train=SEED, seed_test=99, k=16)

fig, ax = plt.subplots(figsize=(8, 4))
preds_a = mix["predictions"][mix["y_test"] == "A"]
preds_b = mix["predictions"][mix["y_test"] == "B"]
ax.hist(preds_a, bins=20, alpha=0.6, label=f"True A (mean pred={np.mean(preds_a):.1f})",
        color="blue")
ax.hist(preds_b, bins=20, alpha=0.6, label=f"True B (mean pred={np.mean(preds_b):.1f})",
        color="red")
ax.axvline(5, color="black", linestyle="--", linewidth=2, label="Threshold=5")
ax.set_xlabel("Predicted 5th Dimension Value")
ax.set_ylabel("Count")
ax.set_title(f"Mixture-of-Normals: SE={mix['standard_error']:.2f}, "
             f"Accuracy={mix['accuracy']*100:.1f}%")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig("figures/fig5_mixture_normals.png", dpi=150)
plt.close()
print("  Saved figures/fig5_mixture_normals.png")
print(f"  SE={mix['standard_error']:.2f} (paper: 2.8)")
print(f"  Accuracy={mix['accuracy']*100:.1f}% (paper: 96%)")


print("Figure 6: C/R Similarity Grouping")
X_cr, _, _ = generate_2d_blobs(n=300, k=6, spread=1.5, seed=123)

cr = CRIncrementalKMeans(k_init=6, C=3.0, R=8.0, random_state=123)
cr.fit(X_cr)
ikm_plain = IncrementalKMeans(k=6, random_state=123)
ikm_plain.fit(X_cr)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
colors_cr = cm.tab10(np.linspace(0, 1, max(cr.k_final, 10)))
for c in range(cr.k_final):
    m = cr.labels_ == c
    if np.any(m):
        ax1.scatter(X_cr[m, 0], X_cr[m, 1], c=[colors_cr[c % 10]], alpha=0.5, s=20)
ax1.scatter(cr.means[:, 0], cr.means[:, 1], c="black", marker="X", s=200,
            edgecolors="white", linewidths=1.5, zorder=5)
ax1.set_title(f"C/R K-Means (C={3.0}, R={8.0}, final k={cr.k_final}, "
              f"WCV={cr.within_class_variation(X_cr):.3f})")
ax1.grid(True, alpha=0.3)

colors_p = cm.tab10(np.linspace(0, 1, 10))
for c in range(6):
    m = ikm_plain.labels_ == c
    if np.any(m):
        ax2.scatter(X_cr[m, 0], X_cr[m, 1], c=[colors_p[c % 10]], alpha=0.5, s=20)
ax2.scatter(ikm_plain.means[:, 0], ikm_plain.means[:, 1], c="black", marker="X",
            s=200, edgecolors="white", linewidths=1.5, zorder=5)
ax2.set_title(f"Plain Incremental (k=6, "
              f"WCV={ikm_plain.within_class_variation(X_cr):.3f})")
ax2.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig("figures/fig6_cr_grouping.png", dpi=150)
plt.close()
print("  Saved figures/fig6_cr_grouping.png")


print("Figure 7: Runtime Scaling")
sizes = [100, 250, 500, 1000, 2000, 5000]
inc_times, bat_times = [], []
for n in sizes:
    X_t, _, _ = generate_2d_blobs(n=n, k=8, seed=SEED)
    ikm_t = IncrementalKMeans(k=8, random_state=SEED)
    ikm_t.fit(X_t)
    inc_times.append(ikm_t.fit_time_)
    bkm_t = BatchKMeans(k=8, random_state=SEED)
    bkm_t.fit(X_t)
    bat_times.append(bkm_t.fit_time_)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(sizes, [t*1000 for t in inc_times], "o-", label="Incremental", color="steelblue")
ax.plot(sizes, [t*1000 for t in bat_times], "s-", label="Batch", color="coral")
ax.set_xlabel("Number of Points")
ax.set_ylabel("Runtime (ms)")
ax.set_title("Runtime Scaling: Incremental vs Batch K-Means")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig("figures/fig7_runtime.png", dpi=150)
plt.close()
print("  Saved figures/fig7_runtime.png")


print("Figure 8: Scrambled Independence Test")
hs_result, X_hs = paper_hollow_square_experiment(seed=SEED)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

ax1.scatter(X_hs[:, 0], X_hs[:, 1], s=15, alpha=0.6, color="steelblue")
ax1.set_title("Hollow Square Dataset (150 points)")
ax1.set_aspect("equal")
ax1.grid(True, alpha=0.3)

ax2.hist(hs_result['wcv_scrambled_all'], bins=10, color="coral", alpha=0.7,
         edgecolor="black", label="Scrambled WCVs")
ax2.axvline(hs_result['wcv_original'], color="steelblue", linewidth=2,
             linestyle="--", label=f"Original WCV = {hs_result['wcv_original']:.2f}")
ax2.set_xlabel("Within-Class Variation")
ax2.set_ylabel("Count")
ax2.set_title(f"Scrambled Test: ratio = {hs_result['ratio']:.2f}x (paper: 1.6x)")
ax2.legend()

plt.tight_layout()
fig.savefig("figures/fig8_scrambled_test.png", dpi=150)
plt.close()
print("  Saved figures/fig8_scrambled_test.png")
print(f"  Hollow square ratio: {hs_result['ratio']:.2f}x (paper: 1.6x)")

fived_results, _ = paper_5d_experiment(seed=SEED)
for k_val in [6, 12, 18]:
    r = fived_results[k_val]
    paper_vals = {6: 1.40, 12: 1.55, 18: 1.39}
    print(f"  5D k={k_val}: ratio = {r['ratio']:.2f}x (paper: {paper_vals[k_val]}x)")


print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"A/B Accuracy (incremental): {np.mean(inc_accs)*100:.1f}% (paper ~87%)")
print(f"A/B Accuracy (batch): {np.mean(bat_accs)*100:.1f}%")
print(f"Mixture SE: {mix['standard_error']:.2f} (paper 2.8)")
print(f"Mixture Accuracy: {mix['accuracy']*100:.1f}% (paper 96%)")
print(f"Mixture Mean Pred A: {mix['mean_pred_A']:.1f} (paper 10.3)")
print(f"Mixture Mean Pred B: {mix['mean_pred_B']:.1f} (paper 1.3)")
print(f"Order Sensitivity: {variation_pct:.1f}% (paper ~7%)")
print(f"Scrambled ratio (hollow sq): {hs_result['ratio']:.2f}x (paper 1.6x)")
print(f"Scrambled ratio (5D k=6): {fived_results[6]['ratio']:.2f}x (paper 1.40x)")
print(f"Scrambled ratio (5D k=12): {fived_results[12]['ratio']:.2f}x (paper 1.55x)")
print(f"Scrambled ratio (5D k=18): {fived_results[18]['ratio']:.2f}x (paper 1.39x)")
print("=" * 60)
print("Figures saved to ./figures/")