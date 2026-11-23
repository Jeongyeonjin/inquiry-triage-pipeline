# -*- coding: utf-8 -*-
"""서버 없이 파이프라인을 직접 돌려 보는 시뮬레이션. 정상 문의와 일부러 만든 실패 사례를 섞어 넣는다.
실행: python simulate.py
※ 문의 문장은 모두 직접 지어낸 가짜 데이터입니다."""
import os, time, tempfile
from store import Store
from pipeline import Pipeline
from classifier import RuleClassifier, TransientError
from report import daily_report, format_report

SAMPLES = [
    ("S01", "택배가 아직 안 왔어요. 언제 도착하나요?"),
    ("S02", "결제 영수증을 다시 받고 싶어요"),
    ("S03", "이 상품 재입고 예정이 있나요?"),
    ("S04", "사이즈가 안 맞아서 교환하고 싶습니다"),
    ("S05", "환불해 주세요"),                                   # 예외 유형 → 검수
    ("S06", "상품이 파손된 채로 왔어요. 최악이네요"),            # 예외 유형 → 검수
    ("S07", "배송이 늦어서 결제를 취소하고 싶어요"),             # 여러 유형이 섞임 → 검수
    ("S08", "안녕하세요"),                                       # 키워드 없음 → 검수
    ("S09", "   "),                                              # 빈 문의 → 검수
    ("S10", "택배 언제 와요?"),                                   # 일시 오류 후 재시도로 성공
    ("S11", "배송 문의드립니다"),                                 # 계속 실패 → failed + 알림
    ("S01", "택배가 아직 안 왔어요. 언제 도착하나요?"),           # 같은 ID 재전송 → 중복
]


class FlakyRule:
    """S10은 처음 2번 타임아웃, S11은 항상 타임아웃. 실제 LLM API가 흔들릴 때를 흉내 낸다."""
    def __init__(self):
        self.rule, self.calls = RuleClassifier(), {}

    def classify(self, text):
        key = "S10" if "택배 언제 와요" in text else "S11" if "배송 문의드립니다" in text else None
        if key:
            self.calls[key] = self.calls.get(key, 0) + 1
            if key == "S11" or self.calls[key] <= 2:
                raise TransientError("API timeout (시뮬레이션)")
        return self.rule.classify(text)


if __name__ == "__main__":
    db = os.path.join(tempfile.mkdtemp(), "sim.db")
    store = Store(db)
    pipe = Pipeline(store, FlakyRule(), notify=lambda m: print("ALERT:", m), sleep=lambda s: None)
    start = time.time()
    for ext, text in SAMPLES:
        iid, created = store.add_inquiry(ext, text)
        if not created:
            print(f"{ext}: 중복 전송 → 다시 처리하지 않음 (#{iid})")
            continue
        print(f"{ext}: {pipe.process(iid)}")
    print()
    print(format_report(daily_report(store, start - 1, time.time() + 1)))
