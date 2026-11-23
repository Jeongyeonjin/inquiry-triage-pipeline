# -*- coding: utf-8 -*-
"""일일 보고: 처리 건수, 상태별·유형별 건수, 예외(검수) 비율, 실패 목록, LLM 비용."""
from collections import Counter


def daily_report(store, since, until):
    rows = store.between(since, until)
    total = len(rows)
    by_status = Counter(r["status"] for r in rows)
    by_category = Counter(r["category"] for r in rows if r["category"])
    failures = [r for r in rows if r["status"] == "failed"]
    review_rate = by_status["needs_review"] / total if total else 0.0
    cost = sum(r["cost"] or 0 for r in rows)
    return {"total": total, "by_status": dict(by_status), "by_category": dict(by_category),
            "review_rate": review_rate, "cost": cost,
            "failures": [(r["id"], r["reason"]) for r in failures]}


def format_report(rep):
    lines = [f"문의 {rep['total']}건",
             "상태별: " + ", ".join(f"{k} {v}" for k, v in sorted(rep["by_status"].items())),
             "유형별: " + ", ".join(f"{k} {v}" for k, v in sorted(rep["by_category"].items())),
             f"사람 검수 비율: {rep['review_rate']:.0%}",
             f"LLM 비용 합계: {rep['cost']:.2f}"]
    if rep["failures"]:
        lines.append("실패:")
        lines += [f"  - #{i}: {reason}" for i, reason in rep["failures"]]
    return "\n".join(lines)
