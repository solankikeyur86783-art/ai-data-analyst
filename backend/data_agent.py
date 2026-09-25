import io
import base64
import os
from typing import TypedDict, List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from langgraph.graph import StateGraph, END
import httpx

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")


class DataState(TypedDict, total=False):
    filename: str
    raw_bytes: bytes
    df: Any
    cleaning_summary: Dict[str, Any]
    stats: Dict[str, Any]
    date_col: str
    numeric_col: str
    forecast: List[Dict[str, Any]]
    anomalies: List[Dict[str, Any]]
    insights_text: str
    report_html: str
    telegram_summary: str


def _read_dataframe(filename: str, raw_bytes: bytes) -> pd.DataFrame:
    if filename.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(raw_bytes))
    return pd.read_csv(io.BytesIO(raw_bytes))


def clean_node(state: DataState) -> DataState:
    df = _read_dataframe(state["filename"], state["raw_bytes"])
    rows_before = len(df)
    df = df.drop_duplicates()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({"": None, "nan": None, "None": None})
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    state["df"] = df
    state["cleaning_summary"] = {
        "rows_before": rows_before,
        "rows_after": len(df),
        "duplicates_removed": rows_before - len(df),
        "columns": list(df.columns),
    }
    return state


def stats_node(state: DataState) -> DataState:
    df = state["df"]
    numeric_df = df.select_dtypes(include=[np.number])
    stats = {}
    for col in numeric_df.columns:
        stats[col] = {
            "mean": float(numeric_df[col].mean()),
            "median": float(numeric_df[col].median()),
            "min": float(numeric_df[col].min()),
            "max": float(numeric_df[col].max()),
            "std": float(numeric_df[col].std() or 0),
        }
    state["stats"] = stats

    date_col = None
    for col in df.columns:
        if "date" in col.lower() or "time" in col.lower():
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().sum() > len(df) * 0.5:
                date_col = col
                break
    if date_col:
        state["date_col"] = date_col
    if len(numeric_df.columns) > 0:
        state["numeric_col"] = numeric_df.columns[0]
    return state


def route_after_stats(state: DataState) -> str:
    if state.get("date_col") and state.get("numeric_col"):
        return "forecast"
    return "anomalies"


def forecast_node(state: DataState) -> DataState:
    df = state["df"]
    date_col = state["date_col"]
    numeric_col = state["numeric_col"]
    temp = df[[date_col, numeric_col]].copy()
    temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
    temp = temp.dropna(subset=[date_col])
    temp[numeric_col] = pd.to_numeric(temp[numeric_col], errors="coerce").fillna(0)

    if temp.empty or temp[date_col].nunique() < 3:
        state["forecast"] = []
        return state

    monthly = temp.set_index(date_col).resample("MS")[numeric_col].sum().reset_index()
    monthly = monthly[monthly[numeric_col] != 0]
    if len(monthly) < 3:
        state["forecast"] = []
        return state

    x = np.arange(len(monthly))
    y = monthly[numeric_col].values
    coeffs = np.polyfit(x, y, 1)
    trend = np.poly1d(coeffs)
    last_date = monthly[date_col].iloc[-1]

    forecast_points = []
    for i in range(1, 4):
        predicted = float(trend(len(monthly) - 1 + i))
        future_date = (last_date + pd.DateOffset(months=i)).strftime("%Y-%m")
        forecast_points.append({"period": future_date, "predicted_value": round(predicted, 2)})
    state["forecast"] = forecast_points
    return state


def anomalies_node(state: DataState) -> DataState:
    df = state["df"]
    numeric_df = df.select_dtypes(include=[np.number])
    anomalies = []
    for col in numeric_df.columns:
        series = numeric_df[col]
        std = series.std()
        if not std or std == 0:
            continue
        mean = series.mean()
        z_scores = (series - mean) / std
        outliers = df[abs(z_scores) > 3]
        for idx, row in outliers.head(5).iterrows():
            anomalies.append({
                "row": int(idx),
                "column": col,
                "value": float(row[col]),
                "z_score": round(float(z_scores.loc[idx]), 2),
            })
    state["anomalies"] = anomalies[:10]
    return state


def _call_groq(system_prompt: str, user_prompt: str) -> str:
    with httpx.Client(timeout=30) as client:
        resp = client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={
                "model": GROQ_MODEL,
                "temperature": 0.3,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            },
        )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def insights_node(state: DataState) -> DataState:
    summary = {
        "cleaning": state.get("cleaning_summary", {}),
        "stats": state.get("stats", {}),
        "anomaly_count": len(state.get("anomalies", [])),
        "forecast": state.get("forecast", []),
    }
    system_prompt = (
        "You are a data analyst assistant. Given a JSON summary of a dataset's cleaning, "
        "statistics, anomaly count, and forecast, write a concise 3-5 sentence executive "
        "summary highlighting the most important findings. Be specific with numbers. "
        "Plain text only, no markdown."
    )
    try:
        state["insights_text"] = _call_groq(system_prompt, f"Dataset summary:\n{summary}")
    except Exception as e:
        state["insights_text"] = f"Could not generate AI insights: {e}"
    return state


def _fig_to_base64() -> str:
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def report_node(state: DataState) -> DataState:
    df = state["df"]
    numeric_col = state.get("numeric_col")
    charts_html = ""

    if numeric_col:
        plt.figure(figsize=(7, 3.5))
        df[numeric_col].dropna().head(200).reset_index(drop=True).plot(kind="line", color="#f59e0b")
        plt.title(f"{numeric_col} — trend (first 200 rows)")
        plt.tight_layout()
        img1 = _fig_to_base64()
        charts_html += f'<img src="data:image/png;base64,{img1}" style="max-width:100%;margin-bottom:20px;"/>'

        plt.figure(figsize=(7, 3.5))
        df[numeric_col].dropna().plot(kind="hist", bins=30, color="#0ea5e9")
        plt.title(f"{numeric_col} — distribution")
        plt.tight_layout()
        img2 = _fig_to_base64()
        charts_html += f'<img src="data:image/png;base64,{img2}" style="max-width:100%;"/>'

    forecast = state.get("forecast", [])
    forecast_html = ""
    if forecast:
        rows = "".join(f"<tr><td>{f['period']}</td><td>{f['predicted_value']}</td></tr>" for f in forecast)
        forecast_html = f"<h3>Forecast (next 3 periods)</h3><table border='1' cellpadding='6' style='border-collapse:collapse;'><tr><th>Period</th><th>Predicted Value</th></tr>{rows}</table>"

    anomalies = state.get("anomalies", [])
    anomalies_html = ""
    if anomalies:
        rows = "".join(
            f"<tr><td>{a['row']}</td><td>{a['column']}</td><td>{a['value']}</td><td>{a['z_score']}</td></tr>"
            for a in anomalies
        )
        anomalies_html = f"<h3>Anomalies detected ({len(anomalies)})</h3><table border='1' cellpadding='6' style='border-collapse:collapse;'><tr><th>Row</th><th>Column</th><th>Value</th><th>Z-score</th></tr>{rows}</table>"

    cleaning = state.get("cleaning_summary", {})
    html = f"""<html><head><meta charset="utf-8"><title>Data Analysis Report</title></head>
    <body style="font-family:sans-serif;max-width:800px;margin:40px auto;color:#1e293b;">
    <h1>AI Data Analyst Report</h1>
    <p><strong>Rows:</strong> {cleaning.get('rows_before')} → {cleaning.get('rows_after')} after cleaning ({cleaning.get('duplicates_removed')} duplicates removed)</p>
    <h2>AI Insights</h2>
    <p>{state.get('insights_text','')}</p>
    {charts_html}
    {forecast_html}
    {anomalies_html}
    </body></html>"""
    state["report_html"] = html

    telegram_lines = [
        "📊 Data Analysis Report",
        f"Rows: {cleaning.get('rows_before')} → {cleaning.get('rows_after')}",
        f"Anomalies found: {len(anomalies)}",
    ]
    if forecast:
        telegram_lines.append(f"Next period forecast: {forecast[0]['period']} → {forecast[0]['predicted_value']}")
    telegram_lines.append("")
    telegram_lines.append(state.get("insights_text", "")[:500])
    state["telegram_summary"] = "\n".join(telegram_lines)
    return state


def build_data_graph():
    graph = StateGraph(DataState)
    graph.add_node("clean", clean_node)
    graph.add_node("stats", stats_node)
    graph.add_node("forecast", forecast_node)
    graph.add_node("anomalies", anomalies_node)
    graph.add_node("insights", insights_node)
    graph.add_node("report", report_node)

    graph.set_entry_point("clean")
    graph.add_edge("clean", "stats")
    graph.add_conditional_edges("stats", route_after_stats, {"forecast": "forecast", "anomalies": "anomalies"})
    graph.add_edge("forecast", "anomalies")
    graph.add_edge("anomalies", "insights")
    graph.add_edge("insights", "report")
    graph.add_edge("report", END)
    return graph.compile()


data_agent = build_data_graph()
