"""
Wineinformatics: predicting whether a wine achieves a 90+ rating from its
sensory-review keywords (Computational Wine Wheel attributes).

Produces everything needed for Section IV (Results) of the paper:
  A) Naive Bayes      - results + figures
  B) Decision Tree    - depth 5..9 results + figures
  C) SVM              - results + figures
  D) Comparison       - comparison table + graphs
  E) Knowledge gained - common keywords / 90+ keywords

Data: Full Wine Data.xlsx (1010 wines), stratified 80/20 train/test split,
      5-fold cross-validation on all wines.

Run:  .venv/bin/python wine_classification.py
Outputs go to ./results/ (figures/, tables/, predictions/, svm_light/).
Dashboard data goes to ./dashboard/data.js - open dashboard/index.html.
"""
import json
import os
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import confusion_matrix, roc_curve, auc

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
DATA_FILE = "Full Wine Data.xlsx"
OUT = "results"
SEED = 42
TEST_SIZE = 0.20          # 80% training / 20% testing
N_FOLDS = 5               # 5-fold cross validation (Chapter 6 evaluation slides)
DT_DEPTHS = [5, 6, 7, 8, 9]
SVM_C = 1.0               # linear kernel, same default model as SVM-light

FIG = os.path.join(OUT, "figures")
TAB = os.path.join(OUT, "tables")
PRED = os.path.join(OUT, "predictions")
SVML = os.path.join(OUT, "svm_light")
for d in (FIG, TAB, PRED, SVML):
    os.makedirs(d, exist_ok=True)

plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 10})
COLORS = {"Naive Bayes": "#4C72B0", "Decision Tree": "#55A868", "SVM": "#C44E52"}


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def load_data(path):
    df = pd.read_excel(path)
    names = df.iloc[:, 0].astype(str)
    y = df.iloc[:, 1].astype(int).values          # 1 = 90+, 0 = below 90
    X = df.iloc[:, 2:].astype(int)
    return names, X, y


# ----------------------------------------------------------------------------
# A) Naive Bayes - implemented from scratch (Bernoulli NB + Laplace smoothing)
# ----------------------------------------------------------------------------
class NaiveBayes:
    """P(C|X) ~ P(C) * prod_k P(x_k|C), with Laplace smoothing so no
    attribute probability is ever zero (Chapter 6 part 2 slides)."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        X = np.asarray(X)
        self.classes_ = np.unique(y)
        self.prior_ = np.array([(y == c).mean() for c in self.classes_])
        # P(attribute = 1 | class) with Laplace smoothing (2 possible values)
        self.p1_ = np.array([(X[y == c].sum(0) + self.alpha) /
                             ((y == c).sum() + 2 * self.alpha)
                             for c in self.classes_])
        return self

    def predict_log_proba(self, X):
        X = np.asarray(X)
        logp1, logp0 = np.log(self.p1_), np.log(1 - self.p1_)
        return np.log(self.prior_) + X @ logp1.T + (1 - X) @ logp0.T

    def predict_proba(self, X):
        lp = self.predict_log_proba(X)
        lp -= lp.max(1, keepdims=True)
        p = np.exp(lp)
        return p / p.sum(1, keepdims=True)

    def predict(self, X):
        return self.classes_[self.predict_log_proba(X).argmax(1)]


# ----------------------------------------------------------------------------
# Evaluation helpers (accuracy, precision, recall/sensitivity, specificity, F1)
# ----------------------------------------------------------------------------
def metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    acc = (tp + tn) / (tp + tn + fp + fn)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"Accuracy": acc, "Precision": prec, "Recall (Sensitivity)": rec,
            "Specificity": spec, "F1-score": f1,
            "TP": tp, "FP": fp, "FN": fn, "TN": tn}


def cross_validate(make_model, X, y):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    accs = []
    for tr, te in skf.split(X, y):
        m = make_model().fit(X[tr], y[tr])
        accs.append((m.predict(X[te]) == y[te]).mean())
    return np.array(accs)


def save_predictions(name, y_true, y_pred):
    """Format requested in the project description: Real grade / Predicted grade."""
    pd.DataFrame({"Real grade": y_true, "Predicted grade": y_pred}).to_csv(
        os.path.join(PRED, f"{name}_predictions.txt"), sep="\t", index=False)


def plot_confusion(cm, title, path, cmap="Blues"):
    fig, ax = plt.subplots(figsize=(4, 3.6))
    ax.imshow(cm, cmap=cmap)
    labels = ["90- (0)", "90+ (1)"]
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title(title)
    ax.spines[:].set_visible(False)
    fig.savefig(path)
    plt.close(fig)


def plot_cv_folds(accs, title, path, color):
    fig, ax = plt.subplots(figsize=(5, 3.2))
    folds = [f"Fold {i + 1}" for i in range(len(accs))]
    bars = ax.bar(folds, accs * 100, color=color)
    ax.axhline(accs.mean() * 100, ls="--", color="black", lw=1,
               label=f"Mean = {accs.mean() * 100:.2f}%")
    ax.bar_label(bars, fmt="%.1f%%", fontsize=8)
    ax.set_ylim(min(50, accs.min() * 100 - 5), 100)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title)
    ax.legend(loc="lower right", frameon=False)
    fig.savefig(path)
    plt.close(fig)


def write_svm_light(path, X, y):
    """Write data in SVM-light format (label idx:val ...) so the original
    svm_learn.exe / svm_classify.exe can be run on Windows if required."""
    with open(path, "w") as f:
        for row, label in zip(X, y):
            feats = " ".join(f"{i + 1}:{v}" for i, v in enumerate(row) if v)
            f.write(f"{1 if label == 1 else -1} {feats}\n")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    names, Xdf, y = load_data(DATA_FILE)
    features = np.array(Xdf.columns)
    X = Xdf.values
    X_tr, X_te, y_tr, y_te, n_tr, n_te = train_test_split(
        X, y, names, test_size=TEST_SIZE, stratify=y, random_state=SEED)
    X_cv, y_cv = X, y                    # 5-fold CV on all wines
    split_desc = (f"{DATA_FILE}: {int((1 - TEST_SIZE) * 100)}/"
                  f"{int(TEST_SIZE * 100)} stratified split (seed {SEED}); "
                  f"{N_FOLDS}-fold cross-validation on all {len(y)} wines.")
    print(f"Dataset: {X.shape[0]} wines, {X.shape[1]} attributes | "
          f"90+ = {(y == 1).sum()}, 90- = {(y == 0).sum()}")
    print(f"Train: {len(y_tr)}  Test: {len(y_te)}\n")

    # Dataset tables/figures (Section II support)
    pd.DataFrame({
        "Set": ["All", "Training", "Testing"],
        "Wines": [len(y), len(y_tr), len(y_te)],
        "90+ (1)": [(y == 1).sum(), (y_tr == 1).sum(), (y_te == 1).sum()],
        "90- (0)": [(y == 0).sum(), (y_tr == 0).sum(), (y_te == 0).sum()],
    }).to_csv(os.path.join(TAB, "dataset_summary.csv"), index=False)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    counts = [(y == 0).sum(), (y == 1).sum()]
    b = axes[0].bar(["90- (0)", "90+ (1)"], counts, color=["#8C8C8C", "#7B2D43"])
    axes[0].bar_label(b)
    axes[0].set_title("Class distribution")
    axes[0].set_ylabel("Number of wines")
    kw = X.sum(1)
    axes[1].hist([kw[y == 0], kw[y == 1]], bins=range(0, kw.max() + 2),
                 label=["90-", "90+"], color=["#8C8C8C", "#7B2D43"])
    axes[1].set_title("Keywords per review")
    axes[1].set_xlabel("Number of attributes present")
    axes[1].legend(frameon=False)
    fig.savefig(os.path.join(FIG, "00_dataset_overview.png"))
    plt.close(fig)

    summary = {}      # model -> test metrics
    cv_scores = {}    # model -> fold accuracies
    times = {}
    probas = {}

    # ---------------------------------------------------------------- A) NB
    print("=== A) Naive Bayes ===")
    t = time.time()
    nb = NaiveBayes(alpha=1.0).fit(X_tr, y_tr)
    times["Naive Bayes"] = time.time() - t
    y_nb = nb.predict(X_te)
    probas["Naive Bayes"] = nb.predict_proba(X_te)[:, 1]
    summary["Naive Bayes"] = metrics(y_te, y_nb)
    cv_scores["Naive Bayes"] = cross_validate(lambda: NaiveBayes(1.0), X_cv, y_cv)
    save_predictions("naive_bayes", y_te, y_nb)

    # sanity check: our implementation vs scikit-learn's BernoulliNB
    sk_acc = (BernoulliNB(alpha=1.0).fit(X_tr, y_tr).predict(X_te) == y_te).mean()
    print(f"Test accuracy = {summary['Naive Bayes']['Accuracy']:.4f} "
          f"(sklearn BernoulliNB check: {sk_acc:.4f})")
    print(f"5-fold CV = {cv_scores['Naive Bayes'].mean():.4f} "
          f"± {cv_scores['Naive Bayes'].std():.4f}")

    plot_confusion(confusion_matrix(y_te, y_nb), "Naive Bayes – Confusion Matrix",
                   os.path.join(FIG, "A1_nb_confusion_matrix.png"), "Blues")
    plot_cv_folds(cv_scores["Naive Bayes"], "Naive Bayes – 5-fold CV accuracy",
                  os.path.join(FIG, "A2_nb_cv_folds.png"), COLORS["Naive Bayes"])

    # Laplace smoothing sensitivity
    alphas = [0.01, 0.1, 0.5, 1, 2, 5, 10]
    a_acc = [cross_validate(lambda a=a: NaiveBayes(a), X_cv, y_cv).mean() for a in alphas]
    pd.DataFrame({"alpha": alphas, "CV accuracy": a_acc}).to_csv(
        os.path.join(TAB, "nb_laplace_alpha.csv"), index=False)
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot([str(a) for a in alphas], np.array(a_acc) * 100, "o-",
            color=COLORS["Naive Bayes"])
    ax.set_xlabel("Laplace smoothing α")
    ax.set_ylabel("5-fold CV accuracy (%)")
    ax.set_title("Naive Bayes – effect of Laplace smoothing")
    fig.savefig(os.path.join(FIG, "A3_nb_laplace_alpha.png"))
    plt.close(fig)

    # most discriminative words according to NB (log-likelihood ratio)
    llr = np.log(nb.p1_[1] / nb.p1_[0])
    order = np.argsort(llr)
    top = np.r_[order[:10], order[-10:]]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(features[top], llr[top],
            color=["#8C8C8C"] * 10 + ["#7B2D43"] * 10)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("log P(word | 90+) / P(word | 90-)")
    ax.set_title("Naive Bayes – most indicative keywords")
    fig.savefig(os.path.join(FIG, "A4_nb_top_keywords.png"))
    plt.close(fig)

    # ---------------------------------------------------------------- B) DT
    print("\n=== B) Decision Tree (depth 5-9) ===")
    dt_rows = []
    dt_models = {}
    for d in DT_DEPTHS:
        mk = lambda d=d: DecisionTreeClassifier(criterion="entropy", max_depth=d,
                                                random_state=SEED)
        t = time.time()
        m = mk().fit(X_tr, y_tr)
        tt = time.time() - t
        dt_models[d] = m
        yp = m.predict(X_te)
        save_predictions(f"decision_tree_depth{d}", y_te, yp)
        cv = cross_validate(mk, X_cv, y_cv)
        r = metrics(y_te, yp)
        r.update({"Depth": d, "Train accuracy": m.score(X_tr, y_tr),
                  "CV mean": cv.mean(), "CV std": cv.std(),
                  "Leaves": m.get_n_leaves(), "Train time (s)": tt})
        dt_rows.append(r)
        print(f"depth {d}: train={r['Train accuracy']:.4f} test={r['Accuracy']:.4f} "
              f"CV={cv.mean():.4f}±{cv.std():.4f} leaves={r['Leaves']}")
    dt_df = pd.DataFrame(dt_rows).set_index("Depth")
    dt_df.round(4).to_csv(os.path.join(TAB, "decision_tree_depths.csv"))

    best_d = int(dt_df["CV mean"].idxmax())
    best_dt = dt_models[best_d]
    print(f"Best depth by CV: {best_d}")
    y_dt = best_dt.predict(X_te)
    summary["Decision Tree"] = metrics(y_te, y_dt)
    cv_scores["Decision Tree"] = cross_validate(
        lambda: DecisionTreeClassifier(criterion="entropy", max_depth=best_d,
                                       random_state=SEED), X_cv, y_cv)
    times["Decision Tree"] = dt_df.loc[best_d, "Train time (s)"]
    probas["Decision Tree"] = best_dt.predict_proba(X_te)[:, 1]

    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(DT_DEPTHS, dt_df["Train accuracy"] * 100, "o-", label="Training",
            color="#999999")
    ax.plot(DT_DEPTHS, dt_df["Accuracy"] * 100, "s-", label="Testing",
            color=COLORS["Decision Tree"])
    ax.errorbar(DT_DEPTHS, dt_df["CV mean"] * 100, yerr=dt_df["CV std"] * 100,
                fmt="^--", capsize=3, label="5-fold CV", color="#2F4B7C")
    ax.set_xticks(DT_DEPTHS)
    ax.set_xlabel("Maximum tree depth")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Decision Tree – accuracy vs depth")
    ax.legend(frameon=False)
    fig.savefig(os.path.join(FIG, "B1_dt_accuracy_vs_depth.png"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.6))
    w = 0.2
    for i, col in enumerate(["Accuracy", "Precision", "Recall (Sensitivity)", "F1-score"]):
        ax.bar(np.array(DT_DEPTHS) + (i - 1.5) * w, dt_df[col] * 100, w, label=col)
    ax.set_xticks(DT_DEPTHS)
    ax.set_ylim(50, 100)
    ax.set_xlabel("Maximum tree depth")
    ax.set_ylabel("Score (%)")
    ax.set_title("Decision Tree – test metrics by depth")
    ax.legend(frameon=False, ncol=4, fontsize=8, loc="upper center",
              bbox_to_anchor=(0.5, -0.18))
    fig.savefig(os.path.join(FIG, "B2_dt_metrics_by_depth.png"))
    plt.close(fig)

    plot_confusion(confusion_matrix(y_te, y_dt),
                   f"Decision Tree (depth {best_d}) – Confusion Matrix",
                   os.path.join(FIG, "B3_dt_confusion_matrix.png"), "Greens")

    fig, ax = plt.subplots(figsize=(22, 9))
    plot_tree(best_dt, max_depth=3, feature_names=features,
              class_names=["90-", "90+"], filled=True, rounded=True,
              impurity=True, fontsize=9, ax=ax)
    ax.set_title(f"Decision Tree (depth {best_d}) – top 4 levels")
    fig.savefig(os.path.join(FIG, "B4_dt_tree_structure.png"))
    plt.close(fig)

    imp = pd.Series(best_dt.feature_importances_, index=features)
    imp = imp[imp > 0].sort_values().tail(15)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.barh(imp.index, imp.values, color=COLORS["Decision Tree"])
    ax.set_xlabel("Information-gain importance")
    ax.set_title(f"Decision Tree (depth {best_d}) – top 15 attributes")
    fig.savefig(os.path.join(FIG, "B5_dt_feature_importance.png"))
    plt.close(fig)

    # --------------------------------------------------------------- C) SVM
    print("\n=== C) SVM (linear kernel) ===")
    write_svm_light(os.path.join(SVML, "wine_train.txt"), X_tr, y_tr)
    write_svm_light(os.path.join(SVML, "wine_test.txt"), X_te, y_te)
    mk_svm = lambda: SVC(kernel="linear", C=SVM_C)
    t = time.time()
    svm = mk_svm().fit(X_tr, y_tr)
    times["SVM"] = time.time() - t
    y_svm = svm.predict(X_te)
    summary["SVM"] = metrics(y_te, y_svm)
    cv_scores["SVM"] = cross_validate(mk_svm, X_cv, y_cv)
    probas["SVM"] = svm.decision_function(X_te)
    save_predictions("svm", y_te, y_svm)
    print(f"Test accuracy = {summary['SVM']['Accuracy']:.4f} | support vectors = "
          f"{svm.n_support_.sum()}")
    print(f"5-fold CV = {cv_scores['SVM'].mean():.4f} ± {cv_scores['SVM'].std():.4f}")

    plot_confusion(confusion_matrix(y_te, y_svm), "SVM – Confusion Matrix",
                   os.path.join(FIG, "C1_svm_confusion_matrix.png"), "Reds")
    plot_cv_folds(cv_scores["SVM"], "SVM – 5-fold CV accuracy",
                  os.path.join(FIG, "C2_svm_cv_folds.png"), COLORS["SVM"])

    # kernel / C comparison
    svm_rows = []
    for kernel in ["linear", "rbf", "poly"]:
        for C in [0.01, 0.1, 1, 10]:
            cv = cross_validate(lambda k=kernel, c=C: SVC(kernel=k, C=c), X_cv, y_cv)
            svm_rows.append({"Kernel": kernel, "C": C, "CV mean": cv.mean(),
                             "CV std": cv.std()})
    svm_df = pd.DataFrame(svm_rows)
    svm_df.round(4).to_csv(os.path.join(TAB, "svm_kernel_C.csv"), index=False)
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    for kernel, g in svm_df.groupby("Kernel"):
        ax.plot(g["C"].astype(str), g["CV mean"] * 100, "o-", label=kernel)
    ax.set_xlabel("C (penalty parameter)")
    ax.set_ylabel("5-fold CV accuracy (%)")
    ax.set_title("SVM – kernel and C comparison")
    ax.legend(title="Kernel", frameon=False)
    fig.savefig(os.path.join(FIG, "C3_svm_kernel_C.png"))
    plt.close(fig)

    coef = pd.Series(svm.coef_.ravel(), index=features).sort_values()
    top = pd.concat([coef.head(10), coef.tail(10)])
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(top.index, top.values,
            color=["#8C8C8C"] * 10 + [COLORS["SVM"]] * 10)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Linear SVM weight (− = 90-, + = 90+)")
    ax.set_title("SVM – most influential keywords")
    fig.savefig(os.path.join(FIG, "C4_svm_top_weights.png"))
    plt.close(fig)

    # -------------------------------------------------------- D) Comparison
    print("\n=== D) Comparison ===")
    models = ["Naive Bayes", "Decision Tree", "SVM"]
    comp = pd.DataFrame(summary).T.loc[models]
    comp["5-fold CV mean"] = [cv_scores[m].mean() for m in models]
    comp["5-fold CV std"] = [cv_scores[m].std() for m in models]
    comp["AUC"] = [auc(*roc_curve(y_te, probas[m])[:2]) for m in models]
    comp["Train time (s)"] = [times[m] for m in models]
    comp.index = ["Naive Bayes", f"Decision Tree (depth {best_d})", "SVM (linear)"]
    comp.index.name = "Model"
    comp.round(4).to_csv(os.path.join(TAB, "model_comparison.csv"))
    print(comp[["Accuracy", "Precision", "Recall (Sensitivity)", "Specificity",
                "F1-score", "AUC", "5-fold CV mean"]].round(4).to_string())

    # grouped bar chart of metrics
    cols = ["Accuracy", "Precision", "Recall (Sensitivity)", "Specificity", "F1-score"]
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(cols))
    w = 0.26
    for i, m in enumerate(models):
        bars = ax.bar(x + (i - 1) * w, comp.iloc[i][cols].astype(float) * 100, w,
                      label=comp.index[i], color=COLORS[m])
        ax.bar_label(bars, fmt="%.1f", fontsize=7)
    ax.set_xticks(x, cols)
    ax.set_ylim(50, 100)
    ax.set_ylabel("Score (%)")
    ax.set_title("Model comparison on the testing dataset")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    fig.savefig(os.path.join(FIG, "D1_metric_comparison.png"))
    plt.close(fig)

    # CV accuracy boxplot
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    bp = ax.boxplot([cv_scores[m] * 100 for m in models], patch_artist=True,
                    tick_labels=comp.index)
    for patch, m in zip(bp["boxes"], models):
        patch.set_facecolor(COLORS[m])
        patch.set_alpha(0.7)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("5-fold cross-validation accuracy")
    fig.savefig(os.path.join(FIG, "D2_cv_boxplot.png"))
    plt.close(fig)

    # ROC curves
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    for i, m in enumerate(models):
        fpr, tpr, _ = roc_curve(y_te, probas[m])
        ax.plot(fpr, tpr, color=COLORS[m],
                label=f"{comp.index[i]} (AUC={comp['AUC'].iloc[i]:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves – testing dataset")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.savefig(os.path.join(FIG, "D3_roc_curves.png"))
    plt.close(fig)

    # side-by-side confusion matrices
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, m, yp, cmap in zip(axes, comp.index, [y_nb, y_dt, y_svm],
                               ["Blues", "Greens", "Reds"]):
        cm = confusion_matrix(y_te, yp)
        ax.imshow(cm, cmap=cmap)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=13,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks([0, 1], ["90-", "90+"])
        ax.set_yticks([0, 1], ["90-", "90+"])
        ax.set_xlabel("Predicted")
        ax.set_title(m, fontsize=10)
        ax.spines[:].set_visible(False)
    axes[0].set_ylabel("True")
    fig.savefig(os.path.join(FIG, "D4_confusion_matrices_side_by_side.png"))
    plt.close(fig)

    # confusion-matrix table (TP/FP/FN/TN)
    comp[["TP", "FP", "FN", "TN"]].astype(int).to_csv(
        os.path.join(TAB, "confusion_counts.csv"))

    # ------------------------------------------------ E) Knowledge gained
    print("\n=== E) Knowledge gained ===")
    freq_all = Xdf.sum().sort_values(ascending=False)
    freq_pos = Xdf[y == 1].mean()
    freq_neg = Xdf[y == 0].mean()
    kw = pd.DataFrame({
        "Count (all)": Xdf.sum(),
        "Count 90+": Xdf[y == 1].sum(),
        "Count 90-": Xdf[y == 0].sum(),
        "% of 90+ wines": freq_pos * 100,
        "% of 90- wines": freq_neg * 100,
        "P(90+ | keyword)": Xdf[y == 1].sum() / Xdf.sum().replace(0, np.nan),
        "NB log-ratio": llr,
        "SVM weight": svm.coef_.ravel(),
    })
    kw.round(4).sort_values("Count (all)", ascending=False).to_csv(
        os.path.join(TAB, "keyword_statistics.csv"))

    common = kw.sort_values("Count (all)", ascending=False).head(20)
    common.round(3).to_csv(os.path.join(TAB, "top20_common_keywords.csv"))
    # 90+ keywords: appear in >= MIN_KW wines and are mostly in 90+ wines
    MIN_KW = 10
    pos_kw = kw[kw["Count (all)"] >= MIN_KW].sort_values(
        "P(90+ | keyword)", ascending=False).head(20)
    pos_kw.round(3).to_csv(os.path.join(TAB, "top20_90plus_keywords.csv"))
    neg_kw = kw[kw["Count (all)"] >= MIN_KW].sort_values("P(90+ | keyword)").head(20)
    neg_kw.round(3).to_csv(os.path.join(TAB, "top20_90minus_keywords.csv"))

    # keywords chosen by all three models (top-20 indicators of 90+)
    nb_top = set(kw["NB log-ratio"].nlargest(20).index)
    svm_top = set(kw["SVM weight"].nlargest(20).index)
    dt_top = set(pd.Series(best_dt.feature_importances_, index=features)
                 .nlargest(20).index)
    agree = sorted(nb_top & svm_top)
    pd.DataFrame({"Keywords in NB & SVM top-20 for 90+": agree}).to_csv(
        os.path.join(TAB, "keywords_agreed_by_models.csv"), index=False)
    print("Most common keywords:", ", ".join(common.index[:10]))
    print("Strongest 90+ keywords:", ", ".join(pos_kw.index[:10]))
    print("NB & SVM agree on 90+ keywords:", ", ".join(agree))
    print("…also in DT top-20:", ", ".join(sorted(nb_top & svm_top & dt_top)))

    fig, ax = plt.subplots(figsize=(7, 5.5))
    c = common.iloc[::-1]
    ax.barh(c.index, c["Count 90-"], color="#8C8C8C", label="90-")
    ax.barh(c.index, c["Count 90+"], left=c["Count 90-"], color="#7B2D43",
            label="90+")
    ax.set_xlabel("Number of wines")
    ax.set_title("Top 20 most common keywords")
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(os.path.join(FIG, "E1_common_keywords.png"))
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    p = pos_kw.iloc[::-1]
    axes[0].barh(p.index, p["P(90+ | keyword)"] * 100, color="#7B2D43")
    axes[0].set_xlabel("% of wines with keyword rated 90+")
    axes[0].set_title("Top 20 keywords of 90+ wines")
    axes[0].set_xlim(0, 100)
    n = neg_kw.iloc[::-1]
    axes[1].barh(n.index, n["P(90+ | keyword)"] * 100, color="#8C8C8C")
    axes[1].set_xlabel("% of wines with keyword rated 90+")
    axes[1].set_title("Top 20 keywords of 90- wines")
    axes[1].set_xlim(0, 100)
    fig.suptitle(f"Keywords appearing in at least {MIN_KW} reviews", fontsize=9, y=0.02)
    fig.savefig(os.path.join(FIG, "E2_90plus_vs_90minus_keywords.png"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    sub = kw[kw["Count (all)"] >= MIN_KW]
    ax.scatter(sub["% of 90- wines"], sub["% of 90+ wines"], s=12, alpha=0.6,
               color="#555555")
    lim = max(sub["% of 90- wines"].max(), sub["% of 90+ wines"].max()) + 2
    ax.plot([0, lim], [0, lim], "k--", lw=0.8)
    gap = (sub["% of 90+ wines"] - sub["% of 90- wines"]).abs()
    for name in gap.nlargest(16).index:
        ax.annotate(name, (sub.loc[name, "% of 90- wines"],
                           sub.loc[name, "% of 90+ wines"]), fontsize=7)
    ax.set_xlabel("% of 90- wines containing keyword")
    ax.set_ylabel("% of 90+ wines containing keyword")
    ax.set_title("Keyword frequency: 90+ vs 90- wines")
    fig.savefig(os.path.join(FIG, "E3_keyword_scatter.png"))
    plt.close(fig)

    # ------------------------------------------- data for the HTML dashboard
    def r(v, n=4):
        return round(float(v), n)

    def metric_dict(m):
        return {k: (int(v) if k in ("TP", "FP", "FN", "TN") else r(v))
                for k, v in m.items()}

    def roc_points(scores):
        fpr, tpr, _ = roc_curve(y_te, scores)
        return [[r(a), r(b)] for a, b in zip(fpr, tpr)]

    def kw_rows(df):
        return [{"keyword": k, "all": int(row["Count (all)"]),
                 "pos": int(row["Count 90+"]), "neg": int(row["Count 90-"]),
                 "pctPos": r(row["% of 90+ wines"], 2),
                 "pctNeg": r(row["% of 90- wines"], 2),
                 "p90": None if pd.isna(row["P(90+ | keyword)"])
                 else r(row["P(90+ | keyword)"]),
                 "nb": r(row["NB log-ratio"]), "svm": r(row["SVM weight"])}
                for k, row in df.iterrows()]

    kw_count = X.sum(1)
    llr_s = pd.Series(llr, index=features).sort_values()
    dash = {
        "dataset": {
            "file": DATA_FILE, "wines": int(len(y)), "attributes": int(X.shape[1]),
            "train": int(len(y_tr)), "test": int(len(y_te)), "seed": SEED,
            "folds": N_FOLDS, "testSize": TEST_SIZE,
            "pos": int((y == 1).sum()), "neg": int((y == 0).sum()),
            "trainPos": int((y_tr == 1).sum()), "trainNeg": int((y_tr == 0).sum()),
            "testPos": int((y_te == 1).sum()), "testNeg": int((y_te == 0).sum()),
            "kwHist": [{"n": int(n), "pos": int(((kw_count == n) & (y == 1)).sum()),
                        "neg": int(((kw_count == n) & (y == 0)).sum())}
                       for n in range(1, int(kw_count.max()) + 1)],
        },
        "models": {
            m: {"label": comp.index[i], "test": metric_dict(summary[m]),
                "cv": [r(a) for a in cv_scores[m]],
                "auc": r(comp["AUC"].iloc[i]), "time": r(times[m], 5),
                "roc": roc_points(probas[m])}
            for i, m in enumerate(models)
        },
        "nb": {
            "alpha": [{"alpha": a, "cv": r(v)} for a, v in zip(alphas, a_acc)],
            "topKeywords": [{"keyword": k, "value": r(v)} for k, v in
                            pd.concat([llr_s.head(10), llr_s.tail(10)]).items()],
        },
        "dt": {
            "bestDepth": best_d,
            "depths": [{"depth": int(d), "train": r(row["Train accuracy"]),
                        "test": r(row["Accuracy"]), "precision": r(row["Precision"]),
                        "recall": r(row["Recall (Sensitivity)"]),
                        "specificity": r(row["Specificity"]),
                        "f1": r(row["F1-score"]), "cvMean": r(row["CV mean"]),
                        "cvStd": r(row["CV std"]), "leaves": int(row["Leaves"])}
                       for d, row in dt_df.iterrows()],
            "importance": [{"keyword": k, "value": r(v)}
                           for k, v in imp.iloc[::-1].items()],
        },
        "svm": {
            "supportVectors": int(svm.n_support_.sum()),
            "kernels": [{"kernel": row["Kernel"], "C": row["C"],
                         "cv": r(row["CV mean"]), "std": r(row["CV std"])}
                        for _, row in svm_df.iterrows()],
            "topWeights": [{"keyword": k, "value": r(v)} for k, v in
                           pd.concat([coef.head(10), coef.tail(10)]).items()],
        },
        "keywords": {
            "minCount": MIN_KW,
            "common": kw_rows(common),
            "pos": kw_rows(pos_kw),
            "neg": kw_rows(neg_kw),
            "all": kw_rows(kw.sort_values("Count (all)", ascending=False)),
            "agreed": agree,
            "agreedAll": sorted(nb_top & svm_top & dt_top),
        },
    }
    dash_dir = "dashboard"
    os.makedirs(dash_dir, exist_ok=True)
    with open(os.path.join(dash_dir, "data.js"), "w") as f:
        f.write("// Generated by wine_classification.py - do not edit by hand.\n")
        f.write("window.WINE_RESULTS = ")
        json.dump(dash, f, indent=1)
        f.write(";\n")

    write_report(dt_df, comp, best_d, common, pos_kw, neg_kw, agree, svm,
                 split_desc, MIN_KW)
    print(f"\nAll outputs saved to ./{OUT}/ (dashboard data: ./dashboard/data.js)")


def write_report(dt_df, comp, best_d, common, pos_kw, neg_kw, agree, svm,
                 split_desc, min_kw):
    pct = lambda v: f"{v * 100:.2f}%"
    L = ["# IV) Results – Wine 90+ Classification\n",
         split_desc + "\n"]

    nb = comp.iloc[0]
    L += ["## A) Naive Bayes\n",
          f"- Test accuracy **{pct(nb['Accuracy'])}**, precision {pct(nb['Precision'])}, "
          f"recall {pct(nb['Recall (Sensitivity)'])}, specificity {pct(nb['Specificity'])}, "
          f"F1 {pct(nb['F1-score'])}",
          f"- 5-fold CV accuracy {pct(nb['5-fold CV mean'])} ± {pct(nb['5-fold CV std'])}",
          "- Figures: A1–A4\n"]

    L += ["## B) Decision Tree (depth 5–9)\n",
          "| Depth | Train acc | Test acc | Precision | Recall | F1 | CV mean | Leaves |",
          "|---|---|---|---|---|---|---|---|"]
    for d, r in dt_df.iterrows():
        L.append(f"| {d} | {pct(r['Train accuracy'])} | {pct(r['Accuracy'])} | "
                 f"{pct(r['Precision'])} | {pct(r['Recall (Sensitivity)'])} | "
                 f"{pct(r['F1-score'])} | {pct(r['CV mean'])} | {int(r['Leaves'])} |")
    L += [f"\nBest depth by cross-validation: **{best_d}**. Figures: B1–B5\n"]

    s = comp.iloc[2]
    L += ["## C) SVM (linear kernel)\n",
          f"- Test accuracy **{pct(s['Accuracy'])}**, precision {pct(s['Precision'])}, "
          f"recall {pct(s['Recall (Sensitivity)'])}, specificity {pct(s['Specificity'])}, "
          f"F1 {pct(s['F1-score'])}",
          f"- 5-fold CV accuracy {pct(s['5-fold CV mean'])} ± {pct(s['5-fold CV std'])}",
          f"- Support vectors: {svm.n_support_.sum()}",
          "- Figures: C1–C4\n"]

    L += ["## D) Comparison\n",
          "| Model | Accuracy | Precision | Recall | Specificity | F1 | AUC | 5-fold CV |",
          "|---|---|---|---|---|---|---|---|"]
    for m, r in comp.iterrows():
        L.append(f"| {m} | {pct(r['Accuracy'])} | {pct(r['Precision'])} | "
                 f"{pct(r['Recall (Sensitivity)'])} | {pct(r['Specificity'])} | "
                 f"{pct(r['F1-score'])} | {r['AUC']:.3f} | "
                 f"{pct(r['5-fold CV mean'])} ± {pct(r['5-fold CV std'])} |")
    L += ["\nFigures: D1–D4\n"]

    L += ["## E) Knowledge gained\n",
          "- **Most common keywords:** " + ", ".join(common.index[:15]),
          f"- **Strongest 90+ keywords** (≥{min_kw} reviews): " + ", ".join(
              f"{k} ({v * 100:.0f}%)" for k, v in pos_kw["P(90+ | keyword)"][:15].items()),
          f"- **Strongest 90- keywords** (≥{min_kw} reviews): " + ", ".join(
              f"{k} ({v * 100:.0f}%)" for k, v in neg_kw["P(90+ | keyword)"][:15].items()),
          "- **90+ keywords agreed on by Naive Bayes and SVM:** " + ", ".join(agree),
          "- Figures: E1–E3\n"]
    with open(os.path.join(OUT, "RESULTS_SUMMARY.md"), "w") as f:
        f.write("\n".join(L))


if __name__ == "__main__":
    main()
