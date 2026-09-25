import os
import json
import uuid

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse

from data_agent import data_agent

app = FastAPI(title="AI Data Analyst Agent")

REPORTS_DIR = "/app/reports"
os.makedirs(REPORTS_DIR, exist_ok=True)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/data/analyze")
async def analyze_data(file: UploadFile = File(...)):
    raw_bytes = await file.read()
    result = data_agent.invoke({"filename": file.filename, "raw_bytes": raw_bytes})

    report_id = str(uuid.uuid4())
    report_path = os.path.join(REPORTS_DIR, f"{report_id}.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(result["report_html"])

    data_payload = {
        "cleaning_summary": result.get("cleaning_summary"),
        "stats": result.get("stats"),
        "anomalies": result.get("anomalies"),
        "forecast": result.get("forecast"),
        "insights_text": result.get("insights_text"),
    }
    data_path = os.path.join(REPORTS_DIR, f"{report_id}.json")
    with open(data_path, "w", encoding="utf-8") as f:
        json.dump(data_payload, f)

    return {
        "report_id": report_id,
        "cleaning_summary": result.get("cleaning_summary"),
        "anomalies": result.get("anomalies"),
        "forecast": result.get("forecast"),
        "insights": result.get("insights_text"),
        "telegram_summary": result.get("telegram_summary"),
        "report_url": f"/data/report/{report_id}",
    }


@app.get("/data/report/{report_id}", response_class=HTMLResponse)
def get_report(report_id: str):
    report_path = os.path.join(REPORTS_DIR, f"{report_id}.html")
    if not os.path.exists(report_path):
        raise HTTPException(404, "Report not found")
    with open(report_path, "r", encoding="utf-8") as f:
        return f.read()


@app.get("/data/report-data/{report_id}")
def get_report_data(report_id: str):
    data_path = os.path.join(REPORTS_DIR, f"{report_id}.json")
    if not os.path.exists(data_path):
        raise HTTPException(404, "Report data not found")
    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)
