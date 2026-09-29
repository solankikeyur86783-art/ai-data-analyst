import os
import json
import uuid

from fastapi import FastAPI, HTTPException, UploadFile, File, Body
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from data_agent import data_agent, _call_groq

app = FastAPI(title="AI Data Analyst Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

REPORTS_DIR = "/app/reports"
STATIC_DIR = "/app/static"
os.makedirs(REPORTS_DIR, exist_ok=True)

app.mount("/dashboard", StaticFiles(directory=STATIC_DIR, html=True), name="dashboard")


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
        "dashboard_url": f"/dashboard/index.html?report_id={report_id}",
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


@app.post("/data/chat/{report_id}")
def chat_with_report(report_id: str, payload: dict = Body(...)):
    data_path = os.path.join(REPORTS_DIR, f"{report_id}.json")
    if not os.path.exists(data_path):
        raise HTTPException(404, "Report data not found")
    with open(data_path, "r", encoding="utf-8") as f:
        report_data = json.load(f)

    user_message = payload.get("message", "").strip()
    if not user_message:
        raise HTTPException(400, "message is required")

    system_prompt = (
        "You are a friendly, concise data analyst assistant embedded in a dashboard. "
        "Answer the user's question using ONLY the dataset summary JSON below. "
        "If the answer isn't derivable from this data, say so honestly. "
        "Keep answers short (2-4 sentences) and use specific numbers when possible.\n\n"
        f"Dataset summary JSON:\n{json.dumps(report_data)}"
    )
    try:
        answer = _call_groq(system_prompt, user_message)
    except Exception as e:
        answer = f"Sorry, I could not process that right now ({e})."

    return {"answer": answer}
