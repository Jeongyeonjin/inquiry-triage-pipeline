# -*- coding: utf-8 -*-
"""알림. 지금은 화면 출력 + alerts.log 기록. Slack 등으로 보내려면 notify() 안에 웹훅 호출을 추가한다."""
import time


def notify(message, path="alerts.log"):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
    print(line)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")
