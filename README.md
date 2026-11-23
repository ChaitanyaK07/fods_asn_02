# CS F320 Foundations of Data Science - Assignment 2: PCA

## Layout

```
fods-a2/
├── data/
│   ├── wine_train.csv          # N = 124
│   └── wine_test.csv           # N_test = 54
├── utils.py                    # data loading, preprocessing, PCA core (C1–C6), reporting/plot helpers
├── p1.py      # P1: eigendecomposition of S, checks, sklearn verification, loadings
├── p2.py# P2: scree/cumulative plots, k selection, J_train/J_test, identity check
├── p3.py         # P3: V1/V2/V3 comparison, 2-D projections, 5-NN check
├── p4.py # P4: PCA vs F-score filter vs SFS wrapper (logistic regression)
├── requirements.txt
└── outputs/                   
```

## Setup and run (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# run each problem
python p1.py
python p2.py
python p3.py
python p4.py

