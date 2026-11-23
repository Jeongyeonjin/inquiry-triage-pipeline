# -*- coding: utf-8 -*-
"""정답이 달린 문의(labeled_samples.csv)로 분류기와 라우팅을 평가한다.
실행: python evaluate.py
보는 지표:
  - 분류 정확도: 분류기가 낸 유형이 정답과 같은 비율
  - 자동 처리 정확도: processed(자동 처리)로 간 건 중 정답인 비율  ← 자동화가 사고를 내지 않는지
  - 검수 비율: 사람에게 간 비율                                       ← 자동화가 일을 얼마나 줄이는지"""
import csv, os, tempfile
from collections import Counter
from store import Store
from pipeline import Pipeline
from classifier import RuleClassifier


def load(path="labeled_samples.csv", only_set=None):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if only_set is None or r.get("set") == only_set]


def evaluate(classifier, rows):
    store = Store(os.path.join(tempfile.mkdtemp(), "eval.db"))
    pipe = Pipeline(store, classifier, sleep=lambda s: None)
    out = []
    for r in rows:
        iid, _ = store.add_inquiry(r["id"], r["text"])
        status = pipe.process(iid)
        got = store.get(iid)
        out.append({**r, "status": status, "pred": got["category"], "conf": got["confidence"]})
    return out


def summarize(out):
    n = len(out)
    correct = sum(o["pred"] == o["true_category"] for o in out)
    auto = [o for o in out if o["status"] == "processed"]
    auto_ok = sum(o["pred"] == o["true_category"] for o in auto)
    review = sum(o["status"] == "needs_review" for o in out)
    print(f"문의 {n}건")
    print(f"분류 정확도: {correct}/{n} = {correct / n:.0%}")
    print(f"자동 처리: {len(auto)}건, 그중 정답 {auto_ok}건 = {auto_ok / len(auto):.0%}" if auto else "자동 처리 없음")
    print(f"사람 검수: {review}건 = {review / n:.0%}")
    print("\n[자동 처리됐는데 틀린 건]  ← 가장 위험한 오류")
    bad_auto = [o for o in auto if o["pred"] != o["true_category"]]
    for o in bad_auto:
        print(f"  {o['id']} 정답={o['true_category']} 예측={o['pred']}({o['conf']}) | {o['text']}")
    if not bad_auto:
        print("  없음")
    print("\n[검수로 보냈는데 분류는 맞은 건]  ← 불필요한 사람 작업")
    unneeded = [o for o in out if o["status"] == "needs_review" and o["pred"] == o["true_category"]]
    print(f"  {len(unneeded)}건 (환불·클레임 정책에 의한 검수 포함)")
    print("\n[분류가 틀린 건 전체]")
    for o in out:
        if o["pred"] != o["true_category"]:
            print(f"  {o['id']} 정답={o['true_category']} 예측={o['pred']}({o['conf']}, {o['status']}) | {o['text']}")


if __name__ == "__main__":
    # 사용법: python evaluate.py [파일.csv] [dev|holdout]
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "labeled_samples.csv"
    only = sys.argv[2] if len(sys.argv) > 2 else None
    summarize(evaluate(RuleClassifier(), load(path, only)))
