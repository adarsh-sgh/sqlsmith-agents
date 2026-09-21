"""Difficulty classifier that routes easy questions to the single-agent path.

Binary target (easy vs complex) because that is the only decision the router makes; the 3-way
difficulty label is still kept for eval reporting. Two interchangeable backends over the same
features (binary TF-IDF + a few structural counts): scikit-learn LogisticRegression, and an
equivalent PyTorch linear softmax model trained full-batch.
"""

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import FunctionTransformer
from torch import nn

from sqlsmith.gold import GOLD, ROUTER_EXTRA

LABELS = ["easy", "complex"]
ENTITIES = ["customer", "order", "product", "categor", "review", "item", "country", "month", "stock", "rating", "revenue", "spend"]
AGGREGATES = ["each", "per ", "top ", "most", "fewest", "highest", "lowest", "than", "never", "both", "at least",
              "every", "average", "total", "distinct", "percentage", "rank"]


def to_label(difficulty):
    return "easy" if difficulty == "easy" else "complex"


def training_data():
    pairs = [(g.question, g.difficulty) for g in GOLD] + ROUTER_EXTRA
    return [q for q, _ in pairs], [to_label(d) for _, d in pairs]


def structural_features(questions):
    """Length, entity-noun count, aggregate/comparative keyword count, and question-opener flags."""
    rows = []
    for q in questions:
        l = q.lower()
        rows.append([
            len(l.split()) / 10,
            sum(e in l for e in ENTITIES),
            sum(a in l for a in AGGREGATES),
            l.count(","),
            l.startswith("how many"),
            l.startswith("which"),
            l.startswith(("list", "show")),
        ])
    return np.array(rows, dtype=float)


def features():
    tfidf = TfidfVectorizer(ngram_range=(1, 2), binary=True, use_idf=False, norm=None)
    return make_union(tfidf, FunctionTransformer(structural_features))


class SklearnRouter:
    def __init__(self, C=1.0):
        self.pipe = make_pipeline(features(), LogisticRegression(C=C, max_iter=2000))

    def fit(self, qs, ys):
        self.pipe.fit(qs, ys)
        return self

    def predict(self, question):
        return self.predict_many([question])[0]

    def predict_many(self, questions):
        """One vectorised call for a batch; this is what the serving micro-batcher uses."""
        return [str(y) for y in self.pipe.predict(list(questions))]


class TorchRouter:
    """features -> nn.Linear -> softmax, Adam with L2 to mirror LogisticRegression's regularisation."""

    def __init__(self, epochs=300, lr=0.05, weight_decay=1e-3, seed=0):
        self.feats = features()
        self.epochs, self.lr, self.weight_decay, self.seed = epochs, lr, weight_decay, seed
        self.model = None

    def _x(self, qs, fit=False):
        m = self.feats.fit_transform(qs) if fit else self.feats.transform(qs)
        return torch.tensor(np.asarray(m.todense() if hasattr(m, "todense") else m), dtype=torch.float32)

    def fit(self, qs, ys):
        torch.manual_seed(self.seed)
        x = self._x(qs, fit=True)
        y = torch.tensor([LABELS.index(l) for l in ys])
        self.model = nn.Linear(x.shape[1], len(LABELS))
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.CrossEntropyLoss()
        for _ in range(self.epochs):
            opt.zero_grad()
            loss_fn(self.model(x), y).backward()
            opt.step()
        return self

    def predict(self, question):
        return self.predict_many([question])[0]

    def predict_many(self, questions):
        with torch.no_grad():
            return [LABELS[int(i)] for i in self.model(self._x(list(questions))).argmax(dim=1)]


BACKENDS = {"sklearn": SklearnRouter, "torch": TorchRouter}


def train_router(backend="sklearn"):
    qs, ys = training_data()
    return BACKENDS[backend]().fit(qs, ys)


def cross_val_accuracy(backend="sklearn", folds=5, seed=0):
    """Stratified k-fold accuracy so both backends are scored identically."""
    qs, ys = training_data()
    qs, ys = np.array(qs), np.array(ys)
    hits = 0
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed).split(qs, ys):
        r = BACKENDS[backend]().fit(list(qs[tr]), list(ys[tr]))
        hits += sum(r.predict(q) == y for q, y in zip(qs[te], ys[te]))
    return hits / len(qs)
