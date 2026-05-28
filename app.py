import os
import json
from flask import Flask, request, jsonify, render_template
from openai import AzureOpenAI
from migrate_to_azure_sql import migrate
from huff_engine import predict_site

from db import get_connection, test_connection
app = Flask(__name__)


# -------------------------
# Azure OpenAI Setup
# -------------------------
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)

DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")


# -------------------------
# Routes
# -------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/dbcheck")
def dbcheck():
    try:
        ok = test_connection()
        return jsonify({"ok": ok})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500





@app.route("/db_structure")
def db_structure():
    """Return all Azure SQL user tables with row counts for assignment verification."""
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT
                    SCHEMA_NAME(t.schema_id) AS schema_name,
                    t.name AS table_name,
                    SUM(p.rows) AS row_count
                FROM sys.tables AS t
                INNER JOIN sys.partitions AS p
                    ON t.object_id = p.object_id
                WHERE p.index_id IN (0, 1)
                GROUP BY SCHEMA_NAME(t.schema_id), t.name
                ORDER BY t.name
                """
            )
            rows = cursor.fetchall()

            # Build a single UNION ALL query that fetches up to 5 rows per table as JSON, then execute once
            table_list = [(schema_name, table_name, int(row_count)) for schema_name, table_name, row_count in rows]
            preview_selects = []
            for schema_name, table_name, _ in table_list:
                # ensure literals are safe in SQL string
                s_schema = schema_name.replace("'", "''")
                s_table = table_name.replace("'", "''")
                preview_sql = (
                    f"SELECT N'{s_schema}' AS schema_name, N'{s_table}' AS table_name, "
                    f"ISNULL((SELECT TOP (5) * FROM [{s_schema}].[{s_table}] FOR JSON PATH), '[]') AS preview_json"
                )
                preview_selects.append(preview_sql)

            tables = []
            if preview_selects:
                combined_sql = "\nUNION ALL\n".join(preview_selects)
                try:
                    cursor.execute(combined_sql)
                    preview_rows = cursor.fetchall()
                    # preview_rows: list of tuples (schema_name, table_name, preview_json)
                    preview_map = { (r[0], r[1]): (r[2] or '[]') for r in preview_rows }
                except Exception:
                    app.logger.exception("Failed to fetch combined previews")
                    preview_map = {}
            else:
                preview_map = {}

            for schema_name, table_name, row_count in table_list:
                pj = preview_map.get((schema_name, table_name), '[]')
                try:
                    preview = json.loads(pj)
                except Exception:
                    app.logger.exception("Failed to parse preview JSON for %s.%s", schema_name, table_name)
                    preview = []

                tables.append({
                    "schema": schema_name,
                    "table_name": table_name,
                    "row_count": int(row_count),
                    "preview": preview,
                })

        return jsonify({"ok": True, "tables": tables})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# -------------------------
# Run Huff Model
# -------------------------

@app.route("/api/run_huff", methods=["POST"])
def api_run_huff():
    try:
        from huff_engine import run_huff_model

        data = request.get_json()

        candidate_lat = data.get("candidate_lat")
        candidate_lon = data.get("candidate_lon")
        business_category = data.get("business_category")
        floor_area = data.get("floor_area")

        if None in [candidate_lat, candidate_lon, business_category, floor_area]:
            return jsonify({"ok": False, "error": "Missing required inputs"}), 400

        result = predict_site(
            candidate_lat=candidate_lat,
            candidate_lon=candidate_lon,
            business_category=business_category,
            store_area_sq_m=floor_area,
        )

        explanation = generate_explanation(result)

        return jsonify({
            "ok": True,
            "result": result,
            "explanation": explanation
        })

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# -------------------------
# Ask Follow-up Questions
# -------------------------

@app.route("/api/ask", methods=["POST"])
def api_ask():
    try:
        data = request.get_json()
        question = data.get("question")
        result = data.get("result")

        if not question or not result:
            return jsonify({"ok": False, "error": "Missing question or result"}), 400

        answer = answer_question(question, result)

        return jsonify({"ok": True, "answer": answer})

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# -------------------------
# LLM Functions
# -------------------------

def generate_explanation(result):
    prompt = f"""
You are an expert in retail location analytics.

A Huff-style gravity model has been run with the following results:

Predicted visits: {result.get("predicted_visits")}
Market share: {result.get("market_share")}
Runtime (ms): {result.get("runtime_ms")}

Competitors (sample):
{result.get("competitors")[:3]}

Explain clearly:
1. What the predicted visits and market share mean
2. What factors likely influenced the result
3. Keep it short and intuitive (3-5 sentences)
"""

    response = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[
            {"role": "system", "content": "You explain analytics results clearly."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.4
    )

    return response.choices[0].message.content


def answer_question(question, result):
    prompt = f"""
You are assisting with a retail location analysis using a Huff model.

Model result:
{result}

User question:
{question}

Answer clearly and concisely, grounded in the model output.
Do not invent data.
"""

    response = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[
            {"role": "system", "content": "You are a helpful data science assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5
    )

    return response.choices[0].message.content

# -------------------------
# Run locally
# -------------------------
if __name__ == "__main__":
    migrate()
    app.run(host="0.0.0.0", port=8000, debug=True)
