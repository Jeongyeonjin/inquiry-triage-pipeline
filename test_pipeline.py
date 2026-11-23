# -*- coding: utf-8 -*-
import time
import pytest
from store import Store
from pipeline import Pipeline
from classifier import RuleClassifier, LLMClassifier, TransientError, InvalidOutput, Result, parse_llm_output
from report import daily_report


@pytest.fixture
def store(tmp_path):
    return Store(str(tmp_path / "t.db"))


def make(store, classifier=None, **kw):
    alerts = []
    p = Pipeline(store, classifier or RuleClassifier(), notify=alerts.append, sleep=lambda s: None, **kw)
    return p, alerts


def run(store, p, text, ext="x1"):
    iid, _ = store.add_inquiry(ext, text)
    return iid, p.process(iid)


def test_clear_inquiry_is_processed_automatically(store):
    p, _ = make(store)
    iid, status = run(store, p, "택배가 언제 도착하나요?")
    assert status == "processed"
    assert store.get(iid)["category"] == "배송"


def test_refund_always_goes_to_review_even_if_confident(store):
    p, _ = make(store)
    iid, status = run(store, p, "환불해 주세요")
    assert status == "needs_review"
    assert store.get(iid)["category"] == "환불"


def test_ambiguous_inquiry_goes_to_review(store):
    p, _ = make(store)
    _, status = run(store, p, "배송이 늦어서 결제를 취소하고 싶어요")   # 배송+결제+환불 키워드가 섞임
    assert status == "needs_review"


def test_empty_and_too_long_text_go_to_review(store):
    p, _ = make(store)
    assert run(store, p, "   ", "a")[1] == "needs_review"
    assert run(store, p, "가" * 2001, "b")[1] == "needs_review"


def test_duplicate_external_id_is_not_created_twice(store):
    first, created1 = store.add_inquiry("same", "안녕하세요")
    second, created2 = store.add_inquiry("same", "안녕하세요")
    assert created1 and not created2 and first == second


class Flaky:
    def __init__(self, fail_times):
        self.calls, self.fail_times = 0, fail_times

    def classify(self, text):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise TransientError("timeout")
        return Result("배송", 0.95, "ok", 100, 20)


def test_transient_error_is_retried_then_succeeds(store):
    flaky = Flaky(fail_times=2)
    p, alerts = make(store, flaky)
    iid, status = run(store, p, "배송 문의")
    assert status == "processed" and flaky.calls == 3 and alerts == []
    assert [l["level"] for l in store.logs_for(iid) if l["step"] == "분류"] == ["warn", "warn", "info"]


def test_retries_exhausted_marks_failed_and_alerts(store):
    p, alerts = make(store, Flaky(fail_times=99), max_retries=3)
    iid, status = run(store, p, "배송 문의")
    assert status == "failed" and len(alerts) == 1 and "재시도 3회" in store.get(iid)["reason"]


def test_unexpected_error_is_recorded_not_swallowed(store):
    class Broken:
        def classify(self, text):
            raise KeyError("boom")
    p, alerts = make(store, Broken())
    iid, status = run(store, p, "배송 문의")
    assert status == "failed" and "KeyError" in store.get(iid)["reason"] and alerts


@pytest.mark.parametrize("raw", [
    "죄송합니다. 분류할 수 없습니다.",                                   # JSON 아님
    '{"category": "스팸", "confidence": 0.9, "summary": "x"}',            # 없는 분류
    '{"category": "배송", "confidence": 1.7, "summary": "x"}',            # 범위 밖
    '{"category": "배송", "confidence": 0.9, "summary": ""}',             # 빈 요약
    '[1, 2]',                                                              # 객체 아님
])
def test_bad_llm_output_is_rejected(raw):
    with pytest.raises(InvalidOutput):
        parse_llm_output(raw)


def test_llm_output_with_code_fence_is_accepted():
    r = parse_llm_output('```json\n{"category": "배송", "confidence": 0.8, "summary": "배송 문의"}\n```', 10, 5)
    assert r.category == "배송" and r.tokens_in == 10


def test_bad_llm_output_goes_to_review_not_failed(store):
    p, alerts = make(store, LLMClassifier(lambda prompt: ("분류 불가", 50, 5)))
    _, status = run(store, p, "배송 문의")
    assert status == "needs_review" and alerts == []


def test_cost_is_recorded_from_tokens(store):
    llm = LLMClassifier(lambda prompt: ('{"category": "배송", "confidence": 0.9, "summary": "s"}', 1000, 500))
    p, _ = make(store, llm, price_in_per_1k=2.0, price_out_per_1k=4.0)
    iid, _ = run(store, p, "배송 문의")
    assert store.get(iid)["cost"] == pytest.approx(1000 / 1000 * 2.0 + 500 / 1000 * 4.0)


def test_human_review_updates_status_and_logs(store):
    p, _ = make(store)
    iid, _ = run(store, p, "환불해 주세요")
    p.review(iid, "교환", "고객이 교환으로 변경 요청")
    row = store.get(iid)
    assert row["status"] == "reviewed" and row["category"] == "교환"
    with pytest.raises(ValueError):
        p.review(iid, "배송")   # 이미 검수 끝난 건은 다시 검수할 수 없다


def test_report_counts(store):
    p, _ = make(store)
    run(store, p, "택배가 언제 도착하나요?", "1")   # 자동 처리
    run(store, p, "환불해 주세요", "2")              # 검수 대기
    rep = daily_report(store, 0, time.time() + 10)
    assert rep["total"] == 2 and rep["by_status"] == {"processed": 1, "needs_review": 1}
    assert rep["review_rate"] == 0.5
