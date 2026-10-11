"""Runs all experiments with grouped 5-fold CV on train.csv (the official test set is never used here).

Usage (from any folder): python code/experiments.py
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score)
from sklearn.model_selection import StratifiedGroupKFold

import common as C
from common import DATA_DIR, ROOT

SHORT = {"Cardiovascular-Pulmonary": "Cardio", "Dermatology": "Derm", "Gastroenterology": "Gastro",
         "General Medicine": "GenMed", "Neurology": "Neuro", "Neurosurgery": "NeuroSurg",
         "Obstetrics-Gynecology": "ObGyn", "Ophthalmology": "Ophth", "Orthopedic": "Ortho",
         "Psychiatry-Psychology": "Psych", "Radiology": "Radio", "Surgery": "Surgery"}


N_FOLDS = 5


def folds(df, seed=C.SEED):
    """5 folds, stratified by label and GROUPED by transcription, so identical notes
    (which often carry different labels) never appear in both train and test of a fold.
    Each fold's held-out part is one of our own test sets."""
    groups = df["transcription"].factorize()[0]
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    return list(sgkf.split(df, df["medical_specialty"], groups))


def scores(y, p):
    f = f1_score(y, p, labels=C.LABELS, average=None, zero_division=0)
    return dict(acc=accuracy_score(y, p), macro_f1=f.mean(), min_f1=f.min(),
                min_f1_class=C.LABELS[int(f.argmin())])


def evaluate(name, make, aug, df, splits):
    """Out-of-fold predictions; metrics reported as mean and std over folds."""
    p = np.empty(len(df), dtype=object)
    per_fold = []
    for tr, te in splits:
        train = C.augment(df.iloc[tr], aug)
        model = make().fit(train, train["medical_specialty"])
        p[te] = model.predict(df.iloc[te])
        per_fold.append(scores(df["medical_specialty"].values[te], p[te]))
    y = df["medical_specialty"].values
    nk = ~df["has_kw"].values
    row = {"model": name, "augmentation": aug}
    for k in ["acc", "macro_f1", "min_f1"]:
        v = [f[k] for f in per_fold]
        row[f"all_{k}"], row[f"all_{k}_std"] = np.mean(v), np.std(v)
    row["all_min_f1_class"] = scores(y, p)["min_f1_class"]
    row.update({f"nokw_{k}": v for k, v in scores(y[nk], p[nk]).items()})
    print(f"{name:26s} {aug:16s} acc={row['all_acc']:.3f}±{row['all_acc_std']:.3f} "
          f"mF1={row['all_macro_f1']:.3f} minF1={row['all_min_f1']:.3f} | "
          f"no-kw acc={row['nokw_acc']:.3f} mF1={row['nokw_macro_f1']:.3f}")
    return row, p


def plot_cm(y, p, title, path):
    cm = confusion_matrix(y, p, labels=C.LABELS)
    names = [SHORT[l] for l in C.LABELS]
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(names)), names, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(names)), names, fontsize=8)
    for i in range(len(names)):
        for j in range(len(names)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Gold"); ax.set_title(title, fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=str(DATA_DIR / "train.csv"))
    ap.add_argument("--out", default=str(ROOT / "outputs"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    df = C.load_train(a.train)
    splits = folds(df)
    print(f"{N_FOLDS} grouped folds, test sizes {[len(te) for _, te in splits]}; "
          f"rows without keywords: {(~df.has_kw).sum()}")

    configs = [
        ("M0 baseline NB (desc)", C.baseline_nb, ["none"]),
        ("M1 text LogReg", C.text_logreg, ["none", "kwdrop", "minority", "kwdrop+minority"]),
        ("M2 text SVM", C.text_svm, ["none", "kwdrop", "minority", "kwdrop+minority"]),
        ("M3 text+keyword SVM", C.kw_svm, ["none", "kwdrop", "minority", "kwdrop+minority"]),
        ("M4 hybrid rule+text SVM", lambda: C.HybridKeywordModel(C.text_svm()),
         ["none", "kwdrop", "minority", "kwdrop+minority"]),
    ]
    # each configuration is cached in outputs/cache so the script can be resumed
    cache = f"{a.out}/cache"; os.makedirs(cache, exist_ok=True)
    rows, preds = [], {}
    for name, make, augs in configs:
        for aug in augs:
            f = f"{cache}/{name.split()[0]}_{aug}.json"
            if os.path.exists(f):
                r = json.load(open(f)); p = np.array(r.pop("oof"), dtype=object)
            else:
                r, p = evaluate(name, make, aug, df, splits)
                json.dump({**r, "oof": list(p)}, open(f, "w"))
            rows.append(r); preds[(name, aug)] = p
    res = pd.DataFrame(rows)
    res.to_csv(f"{a.out}/results_table.csv", index=False)

    # best = highest macro-F1 overall, ties broken by no-keyword macro-F1
    best = res.sort_values(["all_macro_f1", "nokw_macro_f1"], ascending=False).iloc[0]
    key = (best["model"], best["augmentation"])
    print("\nBEST:", key)
    json.dump({"model": key[0], "augmentation": key[1]}, open(f"{a.out}/best_config.json", "w"))
    y = df["medical_specialty"].values
    rep = classification_report(y, preds[key], labels=C.LABELS, digits=3, zero_division=0)
    open(f"{a.out}/per_class_best.txt", "w").write(f"{key}\n\n{rep}")
    print(rep)

    plot_cm(y, preds[key], f"{key[0]} + {key[1]} (5-fold out-of-fold)", f"{a.out}/cm_best.png")
    textonly = ("M2 text SVM", "kwdrop+minority")
    plot_cm(y, preds[textonly], "Text-only SVM (5-fold out-of-fold)", f"{a.out}/cm_text_svm.png")

    # error dump for qualitative analysis
    for k, fname in [(key, "errors_best.csv"), (textonly, "errors_text_svm.csv")]:
        e = df.assign(pred=preds[k])
        e = e[e["pred"] != e["medical_specialty"]]
        dup = df.groupby("transcription")["medical_specialty"].agg(lambda s: "|".join(sorted(set(s))))
        e = e.assign(labels_of_same_text_in_train=e["transcription"].map(dup))
        e[["medical_specialty", "pred", "has_kw", "kw_first", "sample_name", "description",
           "labels_of_same_text_in_train", "transcription"]].to_csv(f"{a.out}/{fname}", sep=";", index=False)
        print(f"{fname}: {len(e)} errors")


if __name__ == "__main__":
    main()
