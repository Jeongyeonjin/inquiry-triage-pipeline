# -*- coding: utf-8 -*-
"""문의 분류기. 규칙 기반(RuleClassifier)과 LLM 기반(LLMClassifier) 둘 다 같은 형태로 결과를 낸다."""
import json, re
from dataclasses import dataclass

CATEGORIES = ["배송", "환불", "교환", "상품문의", "결제", "클레임", "기타"]


class TransientError(Exception):
    """다시 시도하면 될 수 있는 오류 (타임아웃, 429 등)."""


class InvalidOutput(Exception):
    """분류기가 돌려준 값이 약속한 형식이 아닐 때. 재시도하지 않고 사람이 확인하게 한다."""


@dataclass
class Result:
    category: str
    confidence: float
    summary: str
    tokens_in: int = 0
    tokens_out: int = 0


RULES = [
    ("환불", ["환불", "취소", "반품", "돌려받"]),
    ("교환", ["교환", "바꾸", "바꿔", "다른 색", "다른 모델", "한 치수"]),
    ("배송", ["배송", "택배", "언제 와", "도착", "송장", "발송", "출고"]),
    ("결제", ["결제", "카드", "입금", "영수증", "쿠폰", "세금계산서", "적립", "할부"]),
    ("클레임", ["불량", "파손", "고장", "최악", "화가", "깨진", "찢어", "망가", "냄새", "유통기한",
             "욕을", "협박", "신고", "사기", "실망", "반말", "불친절", "던져", "답이 없", "너무하"]),
    ("상품문의", ["재입고", "성분", "사용법", "재고", "방수", "재질", "크기", "가로세로", "건전지",
              "설명서", "인증", "차이", "정품", "보증", "세탁", "안전"]),
]


class RuleClassifier:
    """키워드가 한 분류에만 걸리면 확신(0.9), 여러 분류에 걸리면 애매(0.5), 안 걸리면 기타(0.3)."""

    def classify(self, text):
        hits = [cat for cat, words in RULES if any(w in text for w in words)]
        if len(hits) == 1:
            category, conf = hits[0], 0.9
        elif len(hits) > 1:
            category, conf = hits[0], 0.5
        else:
            category, conf = "기타", 0.3
        return Result(category, conf, text.strip().replace("\n", " ")[:40])


PROMPT = """다음 고객 문의를 분류하세요. 반드시 JSON 한 개만 출력합니다.
형식: {{"category": "<{cats}> 중 하나", "confidence": 0~1 사이 숫자, "summary": "한 줄 요약"}}
문의: {text}"""


def parse_llm_output(raw, tokens_in=0, tokens_out=0):
    """LLM 응답을 믿지 않고 검증한다. 형식이 틀리면 InvalidOutput."""
    cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise InvalidOutput(f"JSON이 아님: {e}") from e
    if not isinstance(data, dict):
        raise InvalidOutput("JSON 객체가 아님")
    category, conf, summary = data.get("category"), data.get("confidence"), data.get("summary")
    if category not in CATEGORIES:
        raise InvalidOutput(f"허용되지 않은 분류: {category!r}")
    if not isinstance(conf, (int, float)) or isinstance(conf, bool) or not 0 <= conf <= 1:
        raise InvalidOutput(f"confidence가 0~1 숫자가 아님: {conf!r}")
    if not isinstance(summary, str) or not summary.strip():
        raise InvalidOutput("summary가 비어 있음")
    return Result(category, float(conf), summary.strip(), tokens_in, tokens_out)


class LLMClassifier:
    """call_llm(prompt) -> (응답 텍스트, 입력 토큰, 출력 토큰) 함수를 받아서 쓴다. llm_client.py 참고."""

    def __init__(self, call_llm):
        self.call_llm = call_llm

    def classify(self, text):
        prompt = PROMPT.format(cats="/".join(CATEGORIES), text=text)
        raw, tin, tout = self.call_llm(prompt)
        return parse_llm_output(raw, tin, tout)
