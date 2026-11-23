# problem 2

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import (
    D,
    Report,
    agree,
    explained_ratio,
    f4,
    load_data,
    pca_fit,
    reconstruct,
    save_fig,
    sci,
    standardise,
    table,
)


TAUS = (0.80, 0.90, 0.95)
ELBOW_THRESHOLD = 0.10


def recon_error(Z: np.ndarray, U: np.ndarray, k: int) -> float:
    resid = Z - reconstruct(Z, U, k)
    return float(np.sum(resid**2) / (Z.shape[0] - 1))


def main() -> None:
    rep = Report("p2_results")
    X_tr, _, X_te, _ = load_data()
    Z_tr, Z_te = standardise(X_tr, X_te)

    S, lam, U = pca_fit(Z_tr)
    r, R = explained_ratio(lam)
    ks = np.arange(1, D + 1)

    # choosing k

    # elbow_k is last k where r_k its still >= 0.10

    k_elbow = 0

    for k in ks:
        if r[k-1] >= 0.10:
            k_elbow = k

    taus = [0.80, 0.90, 0.95]
    k_tau = {}

    for t in taus:
        for k in ks:
            if R[k-1] >= t:
                k_tau[t] = k
                break


    # kaiser: keep eigenvalues above the average eigenvalue
    avg_lam = np.trace(S) / D
    k_kaiser = np.sum(lam > avg_lam)


    rep.header("Choice of k")
    rep(f"k_elbow (last k with r_k >= 0.10) = {k_elbow}   R(k_elbow) = {f4(R[k_elbow - 1])}")

    for t in taus:
        rep(f"k_{t:.2f} = {k_tau[t]}   R = {f4(R[k_tau[t] - 1])}")
    rep(f"Kaiser (lambda > tr(S)/D = {f4(avg_lam)}): {k_kaiser} components")

    # reconstruction error on train and test for every k


    J_tr = []
    J_te = []

    for k in ks:
        J_tr.append(recon_error(Z_tr, U, k))
        J_te.append(recon_error(Z_te, U, k))

    J_tr = np.array(J_tr)
    J_te = np.array(J_te)

    results = pd.DataFrame({
        "k": ks,
        "r_k": r,
        "R(k)": R,
        "J_train(k)": J_tr,
        "J_test(k)": J_te,
    })

    rep.header("Per-k table")
    rep(table(results))
    results.to_csv("outputs/p2_results.csv", index=False)

    # check J_train(k) = sum of the eigenvalues we dropped

    tail = []
    for k in ks:
        tail.append(lam[k:].sum())
    tail = np.array(tail)

    gap = np.max(np.abs(J_tr - tail))
    gap2 = np.max(np.abs(J_tr / np.trace(S) - (1 - R)))

    rep.header("Reconstruction-error identity")
    rep(f"max_k |J_train(k) - sum_(j>k) lambda_j| = {sci(gap)}  ({agree(gap)})")
    rep(f"max_k |J_train(k)/tr(S) - (1 - R(k))|  = {sci(gap2)}  ({agree(gap2)})")

    steps = np.diff(J_te)
    decreasing = "yes" if np.all(steps <= 1e-12) else "NO"
    rep.header("Is J_test(k) non-increasing?")
    rep(f"max_k [J_test(k+1) - J_test(k)] = {sci(steps.max())}  -> {decreasing}")


    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(ks, r, color="#4C72B0", label="$r_j$")
    ax.axvline(k_elbow, ls="--", color="crimson", label=f"$k_{{elbow}}$ = {k_elbow}")
    ax.axhline(0.10, ls=":", color="grey", label="0.10 threshold")
    ax.set_xticks(ks)
    ax.set_xlabel("Component j")
    ax.set_ylabel("Explained variance ratio $r_j$")
    ax.set_title("Scree plot (z-scored training data)")
    ax.legend()
    save_fig(fig, "p2_scree.png")


    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ks, R, "o-", color="#4C72B0", label="$R(k)$")

    colors = ["#55A868", "#C44E52", "#8172B2"]
    for t, c in zip(taus, colors):
        k = k_tau[t]
        ax.axhline(t, ls="--", color=c, label=f"τ = {t:.2f} ($k_{{{t:.2f}}}$ = {k})")
    
            # circle the chosen k so a near miss like R(9)=0.9452 isn't misread
        ax.vlines(k, 0, R[k - 1], colors=c, linestyles=":", lw=1.2)
        ax.plot(k, R[k - 1], "o", ms=11, mfc="none", mec=c, mew=1.8)
        ax.annotate(f"R({k})={R[k - 1]:.4f}", (k, R[k - 1]),
                    xytext=(8, -18), textcoords="offset points",
                    fontsize=8, color=c, zorder=5,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))

    ax.set_xticks(ks)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Number of components k")
    ax.set_ylabel("Cumulative explained variance $R(k)$")
    ax.set_title("Cumulative explained variance")
    ax.legend(loc="lower right")
    save_fig(fig, "p2_cumulative.png")


    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ks, J_tr, "o-", label="$J_{train}(k)$")
    ax.plot(ks, J_te, "s--", label="$J_{test}(k)$")
    ax.set_xticks(ks)
    ax.set_xlabel("Number of components k")
    ax.set_ylabel("Reconstruction error $J(k)$")
    ax.set_title("Reconstruction error vs number of components")
    ax.legend()
    save_fig(fig, "p2_reconstruction_error.png")
    
    rep.save()

if __name__ == "__main__":
    main()
    
    
    




    






