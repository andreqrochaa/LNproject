"""Shared utilities: data loading, cleaning, features, augmentation, models."""
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
from nltk.stem import PorterStemmer
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC

SEED = 42
# project layout: <root>/code/*.py, <root>/datasets/*.csv and <root>/results.txt
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "datasets"
FIELDS = ["description", "sample_name", "transcription", "keywords"]
LABELS = [
    "Cardiovascular-Pulmonary", "Dermatology", "Gastroenterology",
    "General Medicine", "Neurology", "Neurosurgery", "Obstetrics-Gynecology",
    "Ophthalmology", "Orthopedic", "Psychiatry-Psychology", "Radiology", "Surgery",
]
# first keyword string -> label (e.g. "cardiovascular / pulmonary" -> "Cardiovascular-Pulmonary")
KW2LABEL = {l.lower().replace("-", " / "): l for l in LABELS}


# ---------------------------------------------------------------- data
def load_train(path):
    df = pd.read_csv(path, sep=";").fillna("")
    n0 = len(df)
    df = df[df["medical_specialty"].isin(LABELS)].reset_index(drop=True)  # drop 3 corrupted rows
    print(f"[train] {n0} rows read, {n0 - len(df)} corrupted rows removed -> {len(df)}")
    return add_derived(df)


def load_test(path):
    df = pd.read_csv(path, sep=";", header=None, names=FIELDS).fillna("")
    print(f"[test] {len(df)} rows")
    return add_derived(df)


def add_derived(df):
    df = df.copy()
    for c in FIELDS:
        df[c] = df[c].astype(str).str.strip()
    kws = df["keywords"].str.split(",")
    df["kw_first"] = kws.str[0].str.strip().str.lower()
    # keywords without the first entry (which names the specialty)
    df["kw_rest"] = kws.str[1:].str.join(",").str.strip()
    df["has_kw"] = df["kw_first"] != ""
    return df


# ---------------------------------------------------------------- text
_stemmer = PorterStemmer()


def stem_text(s):
    return " ".join(_stemmer.stem(w) for w in re.findall(r"[a-z]+", s.lower()))


def clean_transcription(s):
    # transcriptions use ',' as section separator (",PLAN:1.") -> make it a space
    s = re.sub(r",(?=[A-Z ])", " ", s)
    return re.sub(r"\s+", " ", s)


# ---------------------------------------------------------------- models
def tfidf(**kw):
    base = dict(sublinear_tf=True, lowercase=True, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b")
    base.update(kw)
    return TfidfVectorizer(**base)


def baseline_nb():
    """Course baseline: MNB on TF-IDF of the stemmed, lowercased description."""
    return Pipeline([
        ("prep", ColumnTransformer([("d", TfidfVectorizer(preprocessor=stem_text), "description")])),
        ("clf", MultinomialNB()),
    ])


def text_features(min_df=2):
    """One TF-IDF block per field (no first keyword), so field-specific vocabularies keep separate weights."""
    return [
        ("desc", tfidf(ngram_range=(1, 2), min_df=1), "description"),
        ("name", tfidf(ngram_range=(1, 2), min_df=1), "sample_name"),
        ("trans", tfidf(ngram_range=(1, 2), min_df=min_df, max_df=0.9,
                        preprocessor=lambda s: clean_transcription(s).lower()), "transcription"),
        ("kwrest", tfidf(ngram_range=(1, 1), min_df=1), "kw_rest"),
    ]


def text_svm(C=0.5):
    """Field-wise TF-IDF + linear SVM. Uses text only (first keyword hidden)."""
    return Pipeline([
        ("prep", ColumnTransformer(text_features())),
        ("clf", LinearSVC(C=C, class_weight="balanced", random_state=SEED)),
    ])


def text_logreg(C=10.0):
    return Pipeline([
        ("prep", ColumnTransformer(text_features())),
        ("clf", OneVsRestClassifier(LogisticRegression(C=C, class_weight="balanced", solver="liblinear"))),
    ])


def kw_svm(C=0.5, kw_weight=3.0):
    """Text features + one-hot first keyword (weighted). Single joint model."""
    feats = text_features() + [("kwfirst", OneHotEncoder(handle_unknown="ignore"), ["kw_first"])]
    return Pipeline([
        ("prep", ColumnTransformer(feats, transformer_weights={"kwfirst": kw_weight})),
        ("clf", LinearSVC(C=C, class_weight="balanced", random_state=SEED)),
    ])


class HybridKeywordModel:
    """Rule on the first keyword when present; otherwise a text-only classifier."""

    def __init__(self, text_model):
        self.text_model = text_model

    def fit(self, X, y):
        self.text_model.fit(X, y)
        return self

    def predict(self, X):
        pred = np.asarray(self.text_model.predict(X), dtype=object)
        rule = X["kw_first"].map(KW2LABEL)
        hit = rule.notna().values
        pred[hit] = rule[hit].values
        return pred


# ---------------------------------------------------------------- augmentation
def aug_keyword_dropout(df, frac=1.0, seed=SEED):
    """Copies of training rows with all keywords removed, mimicking keyword-less test rows."""
    src = df[df["has_kw"]].sample(frac=frac, random_state=seed)
    out = src.copy()
    out["keywords"], out["kw_first"], out["kw_rest"], out["has_kw"] = "", "", "", False
    return out


def _sentences(text):
    return [s for s in re.split(r"(?<=[.!?])\s+|,(?=[A-Z][A-Z ]+:)", text) if s.strip()]


def perturb_text(text, rng, p_drop=0.2):
    """Random sentence deletion + local shuffle of sentence order (keeps clinical vocabulary)."""
    sents = _sentences(text)
    if len(sents) < 3:
        words = text.split()
        keep = [w for w in words if rng.random() > p_drop] or words
        return " ".join(keep)
    kept = [s for s in sents if rng.random() > p_drop] or sents
    # shuffle within windows of 3 so the note stays roughly ordered
    for i in range(0, len(kept), 3):
        win = kept[i:i + 3]
        rng.shuffle(win)
        kept[i:i + 3] = win
    return " ".join(kept)


def aug_minority(df, target=150, seed=SEED):
    """Oversample classes below `target` with perturbed copies (sentence drop/shuffle)."""
    rng = random.Random(seed)
    rows = []
    for lab, grp in df.groupby("medical_specialty"):
        need = target - len(grp)
        for i in range(max(0, need)):
            r = grp.iloc[i % len(grp)].copy()
            r["transcription"] = perturb_text(r["transcription"], rng)
            r["description"] = perturb_text(r["description"], rng, p_drop=0.15)
            rows.append(r)
    return pd.DataFrame(rows)


def augment(df, strategy):
    if strategy == "none":
        return df
    parts = [df]
    if "kwdrop" in strategy:
        parts.append(aug_keyword_dropout(df))
    if "minority" in strategy:
        parts.append(aug_minority(df))
    return pd.concat(parts, ignore_index=True)
