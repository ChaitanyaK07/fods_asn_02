#  common utility file for other problem files

from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as  pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"


FEATURES = [
    "alcohol",
    "malic_acid",
    "ash",
    "alcalinity_of_ash",
    "magnesium",
    "total_phenols",
    "flavanoids",
    "nonflavanoid_phenols",
    "proanthocyanins",
    "color_intensity",
    "hue",
    "od280/od315_of_diluted_wines",
    "proline",
]


TARGET = "class"
D = len(FEATURES)
TOL = 1e-8
CLASS_COLORS = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c"}

def ensure_dirs() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)


# data loading

def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    #returns X_tr, y_tr, x_te, y_te
    train = pd.read_csv(DATA_DIR/ "wine_train.csv")
    test = pd.read_csv(DATA_DIR / "wine_test.csv")
    for df, name in ((train, "train"), (test, "test")):
        missing = set(FEATURES + [TARGET]) - set(df.columns)
        if missing:
            raise ValueError(f"wine_{name}.csv is missing columns: {missing}")
    X_tr = train[FEATURES].to_numpy(dtype=float)
    X_te = test[FEATURES].to_numpy(dtype=float)
    y_tr = train[TARGET].to_numpy(dtype=int)
    y_te = test[TARGET].to_numpy(dtype=int)

    return X_tr, y_tr,X_te, y_te

# common_setups

def standardise(X_tr : np.ndarray, X_te : np.ndarray) -> tuple[np.ndarray, np.ndarray]:

    mu = X_tr.mean(axis=0)
    sd = X_tr.std(axis = 0, ddof = 1)

    return (X_tr - mu) / sd, (X_te - mu) / sd

def centre(X_tr: np.ndarray, X_te: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = X_tr.mean(axis=0)
    return (X_tr - mu), (X_te - mu)

def minmax_centre(X_tr: np.ndarray, X_te: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = X_tr.min(axis=0), X_tr.max(axis=0)
    S_tr = (X_tr - lo) / (hi - lo)
    S_te = (X_te - lo) / (hi - lo)

    mu = S_tr.mean(axis=0)
    return S_tr - mu, S_te - mu

def covariance(X: np.ndarray)->np.ndarray:
    Xc = X - X.mean(axis=0)
    return Xc.T @ Xc / (X.shape[0] - 1)


def fix_signs(U: np.ndarray) -> np.ndarray:
    U = U.copy()
    idx = np.argmax(np.abs(U), axis=0)
    signs = np.sign(U[idx, np.arange(U.shape[1])])
    signs[signs == 0] = 1.0
    return U * signs


def eig_sorted(S: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lam, U = np.linalg.eigh(S)
    order = np.argsort(lam)[::-1]
    return lam[order], fix_signs(U[:, order])

def pca_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    S = covariance(X)
    lam, U = eig_sorted(S)
    return S, lam, U


def explained_ratio(lam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r = lam / lam.sum()
    return r, np.cumsum(r)


def project(Z: np.ndarray, U: np.ndarray, k: int) -> np.ndarray:
    return Z @ U[:, :k]

def reconstruct(Z: np.ndarray, U: np.ndarray, k: int) -> np.ndarray:
    Uk = U[:, :k]
    return Z @ Uk @ Uk.T


def f4(x: float) -> str:
    return f"{x:.4f}"

def sci(x: float) -> str:
    return f"{x:.1e}"

def agree(diff: float) -> str:
    return "AGREE" if diff < TOL else "DISAGREE"

def table(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    """Plain-text table with 4-dp floats (no extra dependency like tabulate)."""
    return df.to_string(
        index=False,
        float_format=lambda v: floatfmt.format(v),
        justify="right",
    )

class Report:
    # for report writing

    def __init__(self, name: str):
        ensure_dirs()
        self.path = OUT_DIR / f"{name}.txt"
        self.lines: list[str] = []

    def __call__(self, *parts: object) -> None:
        line = " ".join(str(p) for p in parts)
        print(line)
        self.lines.append(line)

    def header(self, text: str) -> None:
        self("")
        self(text)
        self("-" * len(text))

    def save(self) -> None:
        self.path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")
        print(f"\n[saved] {self.path.relative_to(ROOT)}")

def save_fig(fig: plt.Figure, name: str) -> Path:
    ensure_dirs()
    path = FIG_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[saved] {path.relative_to(ROOT)}")
    return path