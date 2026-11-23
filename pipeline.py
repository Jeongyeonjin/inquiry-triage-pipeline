# -*- coding: utf-8 -*-
"""문의 한 건을 처리하는 흐름: 검사 → 분류(재시도) → 라우팅(자동 처리 / 사람 검수 / 실패) → 비용 기록.
모든 단계는 logs 테이블에 남긴다."""
import time
from classifier import TransientError, InvalidOutput

RISKY_CATEGORIES = {"환불", "클레임"}   # 금전·분쟁이 걸린 유형은 신뢰도가 높아도 사람이 본다
MAX_TEXT_LEN = 2000


class Pipeline:
    def __init__(self, store, classifier, notify=print, max_retries=3, sleep=time.sleep,
                 review_threshold=0.7, price_in_per_1k=0.0, price_out_per_1k=0.0):
        # price_*: 사용하는 LLM 서비스의 요금표를 보고 직접 채운다. 0이면 비용은 0으로 기록된다.
        self.store, self.classifier, self.notify = store, classifier, notify
        self.max_retries, self.sleep, self.review_threshold = max_retries, sleep, review_threshold
        self.price_in, self.price_out = price_in_per_1k, price_out_per_1k

    def process(self, inquiry_id):
        s = self.store
        row = s.get(inquiry_id)
        text = (row["text"] or "").strip()

        # 1) 입력 검사
        if not text:
            return self._to_review(inquiry_id, "입력 검사", "빈 문의")
        if len(text) > MAX_TEXT_LEN:
            return self._to_review(inquiry_id, "입력 검사", f"문의가 {MAX_TEXT_LEN}자를 넘음({len(text)}자)")
        s.log(inquiry_id, "입력 검사", "info", "통과")

        # 2) 분류 (일시 오류는 재시도)
        result = None
        for attempt in range(1, self.max_retries + 1):
            try:
                result = self.classifier.classify(text)
                s.log(inquiry_id, "분류", "info", f"{result.category} / 신뢰도 {result.confidence:.2f} (시도 {attempt})")
                break
            except TransientError as e:
                s.log(inquiry_id, "분류", "warn", f"일시 오류, 시도 {attempt}/{self.max_retries}: {e}")
                if attempt == self.max_retries:
                    return self._fail(inquiry_id, f"분류 재시도 {self.max_retries}회 모두 실패: {e}")
                self.sleep(2 ** attempt)
            except InvalidOutput as e:
                s.log(inquiry_id, "분류", "warn", f"출력 형식 오류: {e}")
                return self._to_review(inquiry_id, "분류", f"분류기 출력 형식 오류: {e}")
            except Exception as e:  # 예상 못 한 오류: 숨기지 않고 실패로 기록하고 알린다
                return self._fail(inquiry_id, f"예상 못 한 오류 {type(e).__name__}: {e}")

        # 3) 비용 기록
        cost = result.tokens_in / 1000 * self.price_in + result.tokens_out / 1000 * self.price_out
        s.update(inquiry_id, category=result.category, confidence=result.confidence, summary=result.summary,
                 tokens_in=result.tokens_in, tokens_out=result.tokens_out, cost=cost)

        # 4) 라우팅
        if result.category in RISKY_CATEGORIES:
            return self._to_review(inquiry_id, "라우팅", f"예외 유형({result.category})은 사람 확인")
        if result.confidence < self.review_threshold:
            return self._to_review(inquiry_id, "라우팅", f"신뢰도 {result.confidence:.2f} < {self.review_threshold}")
        s.update(inquiry_id, status="processed")
        s.log(inquiry_id, "라우팅", "info", "자동 처리 완료")
        return "processed"

    def review(self, inquiry_id, category, note=""):
        """사람이 검수한 결과를 반영한다."""
        row = self.store.get(inquiry_id)
        if row is None or row["status"] != "needs_review":
            raise ValueError("검수 대기 상태의 문의가 아닙니다")
        self.store.update(inquiry_id, status="reviewed", category=category)
        self.store.log(inquiry_id, "검수", "info", f"{row['category']} → {category}. {note}".strip())

    def _to_review(self, inquiry_id, step, reason):
        self.store.update(inquiry_id, status="needs_review", reason=reason)
        self.store.log(inquiry_id, step, "warn", f"검수 대기: {reason}")
        return "needs_review"

    def _fail(self, inquiry_id, reason):
        self.store.update(inquiry_id, status="failed", reason=reason)
        self.store.log(inquiry_id, "실패", "error", reason)
        self.notify(f"[실패] 문의 #{inquiry_id}: {reason}")
        return "failed"
