# -*- coding: utf-8 -*-
"""LLM 호출부. 여기만 직접 채우면 된다.

사용하는 LLM 서비스의 공식 문서를 보고 아래 함수를 구현하세요.
- API 키는 코드에 쓰지 말고 환경변수로 읽는다 (예: os.environ["LLM_API_KEY"]).
- 타임아웃·429(요청 과다)·5xx 오류는 classifier.TransientError 로 바꿔서 raise 한다.
  → 파이프라인이 알아서 재시도한다.
- 반환값: (응답 텍스트, 입력 토큰 수, 출력 토큰 수). 토큰 수는 응답에 들어 있는 사용량 값을 쓴다.
"""


def call_llm(prompt):
    raise NotImplementedError("사용하는 LLM API 호출을 여기에 연결하세요.")
