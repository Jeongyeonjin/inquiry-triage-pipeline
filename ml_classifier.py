# -*- coding: utf-8 -*-
"""직접 학습시키는 작은 분류 모델: 글자 n-gram TF-IDF + 로지스틱 회귀.
RuleClassifier, LLMClassifier와 같은 형태(classify(text) -> Result)라서 파이프라인에 그대로 꽂을 수 있다.
신뢰도는 모델이 낸 확률의 최댓값이다 (확률이 잘 보정돼 있다는 보장은 없다)."""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from classifier import Result


class MLClassifier:
    def __init__(self, C=10.0):
        # 한국어는 띄어쓰기·조사가 불규칙해서 단어보다 글자 2~3개 조각(char n-gram)이 안정적이다
        self.model = make_pipeline(
            TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 3), sublinear_tf=True),
            LogisticRegression(C=C, max_iter=2000),
        )
        self.fitted = False

    def fit(self, texts, labels):
        self.model.fit(texts, labels)
        self.fitted = True
        return self

    def classify(self, text):
        if not self.fitted:
            raise RuntimeError("fit()을 먼저 호출하세요")
        probs = self.model.predict_proba([text])[0]
        i = probs.argmax()
        return Result(self.model.classes_[i], float(probs[i]), text.strip().replace("\n", " ")[:40])
