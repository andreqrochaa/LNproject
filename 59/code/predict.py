"""Trains the best configuration on the FULL train.csv and writes results.txt for the official test set.

Usage (from any folder): python code/predict.py
"""
import argparse

import common as C
from common import DATA_DIR, ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=str(DATA_DIR / "train.csv"))
    ap.add_argument("--test", default=str(DATA_DIR / "test_no_labels.csv"))
    ap.add_argument("--out", default=str(ROOT / "results.txt"))
    a = ap.parse_args()

    train = C.augment(C.load_train(a.train), "kwdrop")      # best augmentation in CV
    test = C.load_test(a.test)
    model = C.HybridKeywordModel(C.text_svm()).fit(train, train["medical_specialty"])  # M4
    pred = model.predict(test)

    assert len(pred) == len(test)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("\n".join(pred) + "\n")
    print(f"wrote {len(pred)} predictions to {a.out}")
    print(test.assign(pred=pred)["pred"].value_counts().to_string())


if __name__ == "__main__":
    main()
