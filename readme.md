# Reproducing MacQueen's (1967) K-Means Clustering

This project reproduces the key algorithms and experiments from:

> MacQueen, J. (1967). *Some Methods for Classification and Analysis of Multivariate Observations.*
> Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability, pp. 281–297.

All clustering algorithms are implemented from scratch using NumPy. No external clustering libraries are used.

---

## What's Included

**Algorithms**
- Incremental K-Means — MacQueen's original online algorithm (Section 2.1)
- Batch K-Means — Lloyd's algorithm for comparison (Section 3.6)
- C/R Coarsening & Refinement — adaptive variant where k changes during processing (Section 3.1)

**Experiments**
- A/B Prediction — 4D classification experiment, paper reports ~87% accuracy (Section 3.2)
- Mixture-of-Normals — regression via cluster-based Gaussian fitting, paper reports SE=2.8 (Section 3.3)
- Scrambled Independence Test — nonparametric test for variable dependence (Section 3.4)
- Order Sensitivity — measures WCV stability across random orderings (Section 3.1)

---

## Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- A terminal / command line

---

## Setup

**1. Clone the repository**

```bash
git clone https://github.com/nikhilshugar/CS5100---kmeans_capstone.git
cd CS5100---kmeans_capstone
```

**2. Create a virtual environment (recommended)**

On macOS / Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows:
```bash
python -m venv .venv
.venv\Scripts\activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

This installs NumPy, Matplotlib, and Streamlit.

---

## Running the Project

### Run all experiments from the terminal

```bash
python3 main.py
```

This runs every experiment and prints results including accuracy, standard error, WCV comparisons, and a summary of pass/fail checks against the paper's reported values.

### Generate report figures

```bash
python3 generate_report_figures.py
```

This creates a `figures/` folder containing PNG plots used in the report:
- `fig1_blobs_comparison.png` — Incremental vs Batch on 2D blobs
- `fig3_order_sensitivity.png` — WCV across random orderings
- `fig4_ab_accuracy.png` — A/B prediction accuracy across test seeds
- `fig5_mixture_normals.png` — Mixture-of-normals prediction distribution
- `fig6_cr_grouping.png` — C/R coarsening and refinement
- `fig7_runtime.png` — Runtime scaling comparison
- `fig8_scrambled_test.png` — Scrambled independence test

### Launch the interactive UI

```bash
streamlit run app.py
```

This opens a browser-based interface where you can:
- Select any experiment from a dropdown
- Adjust parameters (k, spread, seed, C, R)
- View side-by-side comparisons of incremental and batch k-means
- Inspect cluster diagnostics, prediction breakdowns, and WCV metrics

An extended version with additional visualizations (step-by-step animation, elbow curve, convergence plots, confusion matrix) is also available:

```bash
streamlit run app_enhanced.py
```

### Run individual test scripts

```bash
python3 test_step1_incremental.py
python3 test_step2_batch_comparison.py
```

These verify algorithm correctness against hand-traced examples.

---

## Project Structure

```
├── algorithms/
│   ├── __init__.py
│   ├── incremental_kmeans.py      # MacQueen's incremental algorithm
│   ├── batch_kmeans.py            # Lloyd's batch algorithm
│   └── cr_kmeans.py               # C/R coarsening & refinement variant
│
├── data/
│   ├── __init__.py
│   └── generators.py              # Synthetic dataset generators
│
├── experiments/
│   ├── __init__.py
│   ├── ab_prediction.py           # A/B prediction (Section 3.2)
│   ├── mixture_normals.py         # Mixture-of-normals regression (Section 3.3)
│   ├── order_sensitivity.py       # Order sensitivity analysis (Section 3.1)
│   └── scrambled_independence.py  # Scrambled independence test (Section 3.4)
│
├── main.py                        # Runs all experiments
├── generate_report_figures.py     # Generates all report figures
├── app.py                         # Streamlit interactive UI
├── app_enhanced.py                # Extended UI with additional visualizations
├── test_step1_incremental.py      # Hand-trace verification
├── test_step2_batch_comparison.py # Incremental vs batch comparison
├── requirements.txt
└── README.md
```

---

## Key Results

| Metric | Paper | Reproduced |
|---|---|---|
| A/B accuracy (incremental) | ~87% | 88.0% |
| A/B accuracy (batch) | — | 92.4% |
| Mixture-of-normals SE | 2.8 | 2.74 |
| Mixture-of-normals accuracy | 96% | 90.8% |
| Order sensitivity (5D, k=18) | ~7% | ~20% |
| Scrambled ratio (5D, k=18) | 1.39x | 1.39x |

---

## Reference

MacQueen, J. (1967). Some Methods for Classification and Analysis of Multivariate Observations. *Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability*, 1, 281–297.