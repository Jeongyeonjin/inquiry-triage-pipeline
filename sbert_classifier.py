# -*- coding: utf-8 -*-
"""SBERT 임베딩 + 로지스틱 회귀 분류기.
SBERT(사전학습 모델)는 그대로 쓰고, 문장 벡터 위에 얹은 분류기만 우리 데이터로 학습한다.
RuleClassifier, MLClassifier와 같은 형태(classify(text) -> Result)라서 파이프라인에 그대로 꽂힌다.

모델 이름은 Hugging Face 이름이다. 처음 실행할 때 모델을 내려받는다(인터넷 필요).
- "jhgan/ko-sroberta-multitask": 한국어용 SBERT
- "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": 다국어용(더 가벼움)
모델 이름이 바뀌었거나 내려받기에 실패하면 위 둘 중 다른 것으로 바꿔 보세요."""
from sklearn.linear_model import LogisticRegression
from classifier import Result

DEFAULT_MODEL = "jhgan/ko-sroberta-multitask"


class SBERTClassifier:
    def __init__(self, model_name=DEFAULT_MODEL, C=10.0, encode=None):
        # encode: 테스트용으로 임베딩 함수를 직접 넣을 수 있다 (texts -> 2차원 배열). 평소에는 None.
        self.model_name, self.C, self._encode, self.clf = model_name, C, encode, None

    def _embed(self, texts):
        if self._encode is None:
            from sentence_transformers import SentenceTransformer   # 필요할 때만 불러온다
            self._model = SentenceTransformer(self.model_name)
            self._encode = lambda t: self._model.encode(t, show_progress_bar=False)
        return self._encode(list(texts))

    def fit(self, texts, labels):
        self.clf = LogisticRegression(C=self.C, max_iter=2000).fit(self._embed(texts), labels)
        return self

    def classify(self, text):
        if self.clf is None:
            raise RuntimeError("fit()을 먼저 호출하세요")
        probs = self.clf.predict_proba(self._embed([text]))[0]
        i = probs.argmax()
        return Result(self.clf.classes_[i], float(probs[i]), text.strip().replace("\n", " ")[:40])
