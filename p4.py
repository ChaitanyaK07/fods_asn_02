#problem 4


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import SequentialFeatureSelector, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

from utils import (
    FEATURES,
    Report,
    agree,
    f4,
    load_data,
    pca_fit,
    project,
    save_fig,
    sci,
    standardise,
    table,
)

KS = [1, 2, 3, 5, 8]
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

def new_model():
    return LogisticRegression(C=1.0, max_iter=1000)

def test_accuracy(A_tr, y_tr, A_te, y_te):
    model = new_model()
    model.fit(A_tr, y_tr)
    return model.score(A_te, y_te)


def f_scores(Z, y):
    classes = np.unique(y)
    N = Z.shape[0]
    C = len(classes)
    overall_mean = Z.mean(axis=0)

    between = np.zeros(Z.shape[1])
    within = np.zeros(Z.shape[1])


    for c in classes:
        Zc = Z[y == c]
        class_mean = Zc.mean(axis=0)
        between += len(Zc) * (class_mean - overall_mean) ** 2
        within += ((Zc - class_mean) ** 2).sum(axis=0)

    return (between / (C - 1)) / (within / (N - C))


# greedy forward selection, returns features in the order they were picked
# ties go to the lower column index (strict > below)


def forward_selection(Z, y, k_max):
    chosen = []

    for step in range(k_max):
        best_j = None
        best_score = -np.inf

        for j in range(Z.shape[1]):
            if j in chosen:
                continue
            cols = chosen + [j]
            scores = cross_val_score(new_model(), Z[:, cols], y, cv=folds, scoring="accuracy")

            if scores.mean() > best_score:
                best_score = scores.mean()
                best_j = j

        chosen.append(best_j)

    return chosen

def names(idx):
    return ", ".join(FEATURES[j] for j in idx)

def main():
    rep = Report("p4_results")
    X_tr, y_tr, X_te, y_te = load_data()
    Z_tr, Z_te = standardise(X_tr, X_te)
    n_te = len(y_te)

    baseline = test_accuracy(Z_tr, y_tr, Z_te, y_te)
    threshold = baseline - 1 / n_te

    rep.header("Baseline (all 13 standardised features)")
    rep(f"test accuracy = {f4(baseline)}   (threshold baseline - 1/{n_te} = {f4(threshold)})")

    # (A) pca
    _, _, U = pca_fit(Z_tr)

    # (B) f-score ranking

    F = f_scores(Z_tr, y_tr)
    ranking = list(np.argsort(-F, kind="stable"))

    rep.header("F-score ranking (best first)")

    for i, j in enumerate(ranking, start=1):
        rep(f"{i:>2d}. {FEATURES[j]:<30s} F = {F[j]:.4f}")


    #compare with sklearn

    F_sk, _ = f_classif(Z_tr, y_tr)
    dF = np.max(np.abs(F - F_sk))
    rep(f"check vs sklearn f_classif: max |diff| = {sci(dF)}  ({agree(dF)})")


    # (C) forward selection, run once up to 8 and take prefixes
    sfs_order = forward_selection(Z_tr, y_tr, max(KS))
    
    rep.header("SFS selection order (own implementation)")
    rep(names(sfs_order))

    # compare with sklearn's selector for each k
    rep("check vs sklearn SequentialFeatureSelector:")
    for k in KS:
        sk = SequentialFeatureSelector(new_model(), n_features_to_select=k,
                                               direction="forward", scoring="accuracy", cv=folds)

        sk.fit(Z_tr, y_tr)
        sk_set = set(np.flatnonzero(sk.get_support()))
        result = "match" if sk_set == set(sfs_order[:k]) else "MISMATCH"
        rep(f"  k={k}: {result}")

    rows = []
    for k in KS:

        acc_pca = test_accuracy(project(Z_tr, U, k), y_tr, project(Z_te, U, k), y_te)
        top_f = ranking[:k]
        acc_f = test_accuracy(Z_tr[:, top_f], y_tr, Z_te[:, top_f], y_te)

        top_sfs = sfs_order[:k]
        acc_sfs = test_accuracy(Z_tr[:, top_sfs], y_tr, Z_te[:, top_sfs], y_te)

        rows.append({"k": k, "(A) PCA": acc_pca, "(B) F-score": acc_f, "(C) SFS": acc_sfs})

    results = pd.DataFrame(rows)
    rep.header("Test accuracy by k")
    rep(table(results))
    results.to_csv("outputs/p4_results.csv", index=False)



    rep.header(f"Smallest k with accuracy >= baseline - 1/{n_te}")
    for method in ["(A) PCA", "(B) F-score", "(C) SFS"]:
        good = results[results[method] >= threshold - 1e-12]["k"]
        if len(good) > 0:
            rep(f"{method:<12s}: {good.min()}")
        else:
            rep(f"{method:<12s}: not reached for k in table")


    rep.header("Selected features, (B) vs (C)")
    for k in [3, 8]:
        f_set = set(ranking[:k])
        sfs_set = set(sfs_order[:k])
        shared = sorted(f_set & sfs_set)
        rep(f"k={k}")
        rep(f"  (B) F-score: {names(ranking[:k])}")
        rep(f"  (C) SFS    : {names(sfs_order[:k])}")
        rep(f"  shared     : {len(shared)}  -> {names(shared) or '-'}")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(results["k"], results["(A) PCA"], marker="o", ms=13, mfc="none",
                mew=1.8, lw=1.5, label="(A) PCA", zorder=3)
    ax.plot(results["k"], results["(B) F-score"], marker="s", ms=9, mfc="none",
                mew=1.8, lw=1.5, label="(B) F-score", zorder=4)
    ax.plot(results["k"], results["(C) SFS"], marker="^", ms=6, mfc="none",
                mew=1.8, lw=1.5, label="(C) SFS", zorder=5)
    ax.axhline(baseline, ls="--", color="black", label=f"baseline (all 13) = {baseline:.4f}")
    
    ax.set_xticks(KS)
    ax.set_xlabel("Number of features / components k")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Logistic regression test accuracy vs k")
    ax.legend(loc="lower right")
    save_fig(fig, "p4_accuracy_vs_k.png")
    
    rep.save()











if __name__ == "__main__":
    main()





