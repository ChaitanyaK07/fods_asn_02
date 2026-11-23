#problem 3

# effect of preprocessing on pca + 2d projection

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.neighbors import KNeighborsClassifier


from utils import (
    CLASS_COLORS,
    FEATURES,
    Report,
    centre,
    explained_ratio,
    load_data,
    minmax_centre,
    pca_fit,
    project,
    save_fig,
    standardise,
    table,
)


def main():

    rep = Report("p3_results")
    X_tr, y_tr, X_te, y_te = load_data()

    variants = [
        ("V1 (centred)", centre),
        ("V2 (min-max)", minmax_centre),
        ("V3 (z-score)", standardise),
    ]

    rows = []
    projections = []

    for name, prep in variants:

        P_tr, P_te = prep(X_tr, X_te)
        _, lam, U = pca_fit(P_tr)
        r, _ = explained_ratio(lam)


        C_tr = project(P_tr, U, 2)
        C_te = project(P_te, U, 2)


        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(C_tr, y_tr)
        acc = knn.score(C_te, y_te)

        top_feature = FEATURES[np.argmax(np.abs(U[:, 0]))]

        rows.append({
                    "variant": name,
                    "r1": r[0],
                    "r2": r[1],
                    "r1+r2": r[0] + r[1],
                    "top PC1 feature": top_feature,
                    "5-NN test acc": acc,
                })

        projections.append((name, C_tr, C_te, r))

    rep.header("Preprocessing comparison")
    rep(table(pd.DataFrame(rows)))

    var = X_tr.var(axis=0, ddof=1)
    top3 = np.argsort(var)[::-1][:3]

    rep.header("V1: three features with largest raw sample variance")

    for j in top3:
        share = var[j] / var.sum()
        rep(f"{FEATURES[j]:<30s} s^2 = {var[j]:.4f}   (share of tr(S) = {share:.4f})")


    # plot: one panel per variant, train = dots, test = crosses

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, (name, C_tr, C_te, r) in zip(axes, projections):
        for c, col in CLASS_COLORS.items():
            tr_mask = y_tr == c
            te_mask = y_te == c

            ax.scatter(C_tr[tr_mask, 0], C_tr[tr_mask, 1], c=col, marker="o",
                                   s=28, alpha=0.75, edgecolors="none")
            ax.scatter(C_te[te_mask, 0], C_te[te_mask, 1], c=col, marker="x",
                                   s=40, linewidths=1.5)

        ax.set_title(f"{name}: r1={r[0]:.4f}, r2={r[1]:.4f}")
        ax.set_xlabel("PC1 score")
        ax.set_ylabel("PC2 score")
        ax.grid(alpha=0.3)

    
    handles = []
    for c, col in CLASS_COLORS.items():
        handles.append(Line2D([], [], ls="", marker="o", color=col, label=f"class {c}"))
    handles.append(Line2D([], [], ls="", marker="o", color="grey", label="train"))
    handles.append(Line2D([], [], ls="", marker="x", color="grey", label="test"))

    for ax in axes:
        ax.legend(handles=handles, fontsize=8, loc="best")


    fig.suptitle("PC1-PC2 projection under three preprocessing variants")
    save_fig(fig, "p3_projections.png")
    
    # discussion

    rep.header("Discussion")
    rep("(i) V1 does no scaling, so tr(S) is just the sum of the raw feature variances, and")
    rep("proline alone (s^2 = 94906.7109) makes up 99.73% of it. So r1 is at least proline's")
    rep("share, which gives r1 = 0.9978 with PC1 basically pointing along proline, while V3 works")
    rep("on the correlation matrix where every feature has variance 1, and gets r1 = 0.3573.")
    rep("(ii) V1 keeps 99.98% of the variance in two components but gives the worst 5-NN")
    rep("accuracy, 0.7222, against 0.9444 for V3 and 0.9815 for V2: its projection separates")
    rep("class 0 (high proline) while classes 1 and 2 overlap. Variance captured depends on the")
    rep("units the features are measured in, so a high value does not mean the components carry")
    rep("class-relevant structure. (iii) I would use V3 (z-score), because PCA on the correlation")
    rep("matrix does not depend on measurement units and weights every feature equally. V2 scored")
    rep("slightly higher here, but the gap is only two test samples out of 54, and min-max scaling")
    rep("depends on the single most extreme training values, which makes it sensitive to outliers.")

    rep.save()
    
if __name__ == "__main__":
    main()