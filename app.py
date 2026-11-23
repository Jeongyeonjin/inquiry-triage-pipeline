# -*- coding: utf-8 -*-
"""웹훅 서버. 실행: uvicorn app:app --reload   (fastapi, uvicorn 필요)
이 파일은 얇게 유지했다. 실제 처리 로직은 pipeline.py에 있고, 테스트도 거기에 있다."""
import os, time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from store import Store
from pipeline import Pipeline
from classifier import RuleClassifier, LLMClassifier
from report import daily_report, format_report
from notify import notify

store = Store(os.environ.get("DB_PATH", "inquiries.db"))
if os.environ.get("USE_LLM") == "1":
    from llm_client import call_llm
    classifier = LLMClassifier(call_llm)
else:
    classifier = RuleClassifier()
pipeline = Pipeline(store, classifier, notify=notify)
app = FastAPI(title="문의 자동 분류 데모")


class InquiryIn(BaseModel):
    external_id: str = Field(min_length=1)
    text: str


class ReviewIn(BaseModel):
    category: str
    note: str = ""


@app.post("/webhook/inquiry")
def receive(body: InquiryIn):
    inquiry_id, created = store.add_inquiry(body.external_id, body.text)
    if not created:   # 같은 ID가 다시 오면 다시 처리하지 않는다
        return {"id": inquiry_id, "duplicate": True, "status": store.get(inquiry_id)["status"]}
    return {"id": inquiry_id, "status": pipeline.process(inquiry_id)}


@app.get("/review")
def review_queue():
    return store.list_by_status("needs_review")


@app.post("/review/{inquiry_id}")
def review(inquiry_id: int, body: ReviewIn):
    try:
        pipeline.review(inquiry_id, body.category, body.note)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"id": inquiry_id, "status": "reviewed"}


@app.get("/inquiry/{inquiry_id}/logs")
def logs(inquiry_id: int):
    return store.logs_for(inquiry_id)


@app.get("/report")
def report(hours: int = 24):
    now = time.time()
    rep = daily_report(store, now - hours * 3600, now + 1)
    return {**rep, "text": format_report(rep)}
