"""Small hyperparameter grid, evaluated with the same grouped 5-fold CV as experiments.py.

Each setting is cached in outputs/tuning_cache, so the script can be interrupted and resumed.
Note: settings are selected on the same folds used for reporting, so the best scores are
slightly optimistic (no nested CV); differences between settings are what matter here.

Usage (from any folder): python code/tuning.py
"""
import json
import os

import numpy as np
import pandas as pd

import common as C
import experiments as E
from common import DATA_DIR, ROOT

# (experiment, hyperparameter, value, model factory, augmentation strategy, augmentation kwargs)
GRID = (
    [("M1 LogReg", "C", c, (lambda c=c: C.text_logreg(C=c)), "none", {}) for c in [1, 10, 100]]
    + [("M2 SVM", "C", c, (lambda c=c: C.text_svm(C=c)), "none", {}) for c in [0.1, 0.5, 1, 2]]
    + [("M2 SVM", "min_df", m, (lambda m=m: C.text_svm(min_df=m)), "none", {}) for m in [1, 2, 5]]
    + [("M3 SVM+kw", "kw_weight", w, (lambda w=w: C.kw_svm(kw_weight=w)), "kwdrop", {}) for w in [1, 3, 10]]
    + [("M2 SVM + A2", "target", t, C.text_svm, "minority", {"target": t}) for t in [150, 300]]
    + [("M2 SVM + A2", "p_drop", p, C.text_svm, "minority", {"p_drop": p}) for p in [0.1, 0.4]]
)


def run(df, splits, make, aug, aug_kw):
    y = df["medical_specialty"].values
    p = np.empty(len(df), dtype=object)
    folds = []
    for tr, te in splits:
        train = C.augment(df.iloc[tr], aug, **aug_kw)
        p[te] = make().fit(train, train["medical_specialty"]).predict(df.iloc[te])
        folds.append(E.scores(y[te], p[te]))
    nk = ~df["has_kw"].values
    return {"acc": np.mean([f["acc"] for f in folds]),
            "macro_f1": np.mean([f["macro_f1"] for f in folds]),
            "macro_f1_std": np.std([f["macro_f1"] for f in folds]),
            "min_f1": np.mean([f["min_f1"] for f in folds]),
            "nokw_macro_f1": E.scores(y[nk], p[nk])["macro_f1"]}


def main():
    out = ROOT / "outputs"
    cache = out / "tuning_cache"
    os.makedirs(cache, exist_ok=True)
    df = C.load_train(DATA_DIR / "train.csv")
    splits = E.folds(df)
    rows = []
    for exp, hp, val, make, aug, aug_kw in GRID:
        f = cache / f"{exp.replace(' ', '_').replace('+', 'p')}_{hp}_{val}.json"
        if f.exists():
            r = json.load(open(f))
        else:
            r = {"experiment": exp, "hyperparameter": hp, "value": val, **run(df, splits, make, aug, aug_kw)}
            json.dump(r, open(f, "w"))
        rows.append(r)
        print(f"{exp:12s} {hp:9s}={val:<6} acc={r['acc']:.3f} mF1={r['macro_f1']:.3f}±{r['macro_f1_std']:.3f} "
              f"minF1={r['min_f1']:.3f} no-kw mF1={r['nokw_macro_f1']:.3f}", flush=True)
    pd.DataFrame(rows).to_csv(out / "tuning_results.csv", index=False)


if __name__ == "__main__":
    main()
