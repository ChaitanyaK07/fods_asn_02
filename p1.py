# problem 1

# implementing pca from scratch

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from utils import (
    D,
    FEATURES,
    Report,
    agree,
    f4,
    fix_signs,
    load_data,
    pca_fit,
    sci,
    standardise,
    table,
)


def main() -> None:

    rep = Report("p1_results")
    X_tr, y_tr, X_te, y_te = load_data()

    Z_tr, _ = standardise(X_tr, X_te) #standardize

    #pca from scratch

    S, lam, U = pca_fit(Z_tr)
    rep.header("Eigenvalues of S (descending)")
    for j, v in enumerate(lam, start=1):
        rep(f"lambda {j} = {v:.4f}")

    #sanity_checks

    ortho = np.linalg.norm(U.T @ U - np.eye(D), ord="fro")
    trS = np.trace(S)
    trace_gap = trS - lam.sum()
    rep.header("Orthonormality and trace checks")

    rep(f"||U^T U - I||_F       = {sci(ortho)}  ({agree(ortho)})")
    rep(f"tr(S) - sum(lambda)   = {sci(trace_gap)}  ({agree(abs(trace_gap))})")
    rep(f"tr(S)                 = {f4(trS)}  (expected D = {D} for z-scored data)")



    #verification against sklearn

    pca = PCA(n_components=D, svd_solver="full").fit(Z_tr)
    lam_sk = pca.explained_variance_
    U_sk = fix_signs(pca.components_.T)
    d_lam = np.max(np.abs(lam - lam_sk))
    d_U = np.max(np.abs(U - U_sk))


    rep.header("Verification vs sklearn.decomposition.PCA")
    rep(f"max |lambda - explained_variance_|   = {sci(d_lam)}  ({agree(d_lam)})")
    rep(f"max |U - components_^T| (sign-fixed) = {sci(d_U)}  ({agree(d_U)})")

    load = pd.DataFrame({"feature": FEATURES, "u1": U[:, 0], "u2": U[:, 1]})
    rep.header("Loadings of u1 and u2")
    rep(table(load))


    for col, pc in (("u1", "PC1"), ("u2", "PC2")):
        top = load.reindex(load[col].abs().sort_values(ascending=False).index).head(3)
        desc = ", ".join(f"{r.feature} ({r[col]:+.4f})" for _, r in top.iterrows())
        rep(f"Top-3 |loading| on {pc}: {desc}")

    rep.save()


if __name__ == "__main__":
    main()

