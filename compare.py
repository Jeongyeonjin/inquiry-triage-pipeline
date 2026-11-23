# -*- coding: utf-8 -*-
"""규칙 기반 vs 학습 모델을 같은 기준으로 비교한다. 학습은 dev로만 하고, 평가는 holdout으로만 한다.
실행: python compare.py"""
from evaluate import load, evaluate
from classifier import RuleClassifier
from ml_classifier import MLClassifier


def metrics(out):
    n = len(out)
    acc = sum(o["pred"] == o["true_category"] for o in out) / n
    auto = [o for o in out if o["status"] == "processed"]
    auto_ok = sum(o["pred"] == o["true_category"] for o in auto)
    review = sum(o["status"] == "needs_review" for o in out) / n
    return acc, len(auto), auto_ok, review


if __name__ == "__main__":
    dev, hold = load("labeled_100.csv", "dev"), load("labeled_100.csv", "holdout")
    X, y = [r["text"] for r in dev], [r["true_category"] for r in dev]
    candidates = [("규칙 기반", RuleClassifier()), ("TF-IDF 모델", MLClassifier().fit(X, y))]
    try:   # SBERT는 설치돼 있고 모델을 내려받을 수 있을 때만 비교에 넣는다
        from sbert_classifier import SBERTClassifier
        candidates.append(("SBERT 모델", SBERTClassifier().fit(X, y)))
    except Exception as e:
        print(f"[SBERT 건너뜀] {type(e).__name__}: {e}\n")
    print(f"{'':12}{'분류 정확도':>12}{'자동 처리':>10}{'그중 정답':>10}{'검수 비율':>10}")
    for name, clf in candidates:
        acc, na, ok, rev = metrics(evaluate(clf, hold))
        print(f"{name:12}{acc:>12.0%}{na:>10}{ok:>10}{rev:>10.0%}")
    print("\n※ holdout 40건 기준. 학습 모델은 dev 60건으로만 학습함.")
