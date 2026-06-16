import os
import json
import re
from flask import Flask, request, jsonify, render_template
from openai import AzureOpenAI
from migrate_to_azure_sql import migrate

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

        # use the Flask-friendly wrapper implemented in huff_engine
        result = run_huff_model(
            candidate_lat=candidate_lat,
            candidate_lon=candidate_lon,
            business_category=business_category,
            floor_area=floor_area,
        )

        explanation = generate_explanation(result)

        return jsonify({
            "ok": True,
            "result": result,
            "explanation": explanation
        })

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# =========================================================
# AI CONTROLLER  (/api/chat)
# ---------------------------------------------------------
# The model drives the conversation and calls the Huff engine
# as a TOOL. It never computes numbers itself: every figure it
# states comes from run_huff_model's actual return value.
# =========================================================

CHAT_SYSTEM_PROMPT = """You are the guided assistant for an AI-Assisted Location \
Decision Support System for Worcester, Massachusetts. The system evaluates candidate \
retail/service locations using a Huff gravity model.

Your job:
- Help the user assemble the four inputs the model needs:
  (1) business_category as a NAICS code (e.g. 4441),
  (2) candidate_lat and (3) candidate_lon for the proposed location
      (the user can also pick this on the map),
  (4) floor_area in square meters.
- Ask for whatever is still missing, one or two items at a time, in plain language.
- As soon as you have all four inputs, call the run_huff_model tool. Do not ask for
  extra confirmation unless the request is genuinely ambiguous.
- After the tool returns, explain the results in 3-5 clear sentences: what the predicted
  visits and market share mean, and what likely drove them.

Hard rules:
- NEVER invent, estimate, or guess numeric results (predicted visits, market share,
  distances, attraction scores). Those come ONLY from the run_huff_model tool. If a
  question needs numbers you don't already have, call the tool.
- If the user changes any input (location, NAICS, or floor area) and wants new results,
  call run_huff_model again with the updated values.
- If the latest tool result already answers the user's question, answer from it directly
  instead of calling the tool again.
- Keep replies concise and grounded in the tool output. If you lack information, say
  exactly what you still need.
- Worcester is roughly latitude 42.2-42.3 and longitude -71.9 to -71.7. Gently flag
  coordinates that fall well outside this range before running.
"""

HUFF_TOOL = {
    "type": "function",
    "function": {
        "name": "run_huff_model",
        "description": (
            "Run the Huff gravity model for a candidate retail/service location in "
            "Worcester, MA. Returns predicted visits, estimated market share, and nearby "
            "competitor attraction scores. Always call this to obtain results; never "
            "estimate the numbers yourself."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "candidate_lat": {
                    "type": "number",
                    "description": "Latitude of the candidate location in decimal degrees (Worcester ~42.2-42.3).",
                },
                "candidate_lon": {
                    "type": "number",
                    "description": "Longitude of the candidate location in decimal degrees (Worcester ~ -71.9 to -71.7).",
                },
                "business_category": {
                    "type": "string",
                    "description": "NAICS code for the business category, e.g. '4441'.",
                },
                "floor_area": {
                    "type": "number",
                    "description": "Proposed store floor area in square meters (positive number).",
                },
            },
            "required": ["candidate_lat", "candidate_lon", "business_category", "floor_area"],
        },
    },
}


def _execute_run_huff(args):
    """Call the real Huff engine and attach the inputs used (so the UI can sync the map)."""
    from huff_engine import run_huff_model

    result = run_huff_model(
        candidate_lat=args["candidate_lat"],
        candidate_lon=args["candidate_lon"],
        business_category=str(args["business_category"]),
        floor_area=args["floor_area"],
    )

    if isinstance(result, dict):
        result.setdefault("candidate_lat", args["candidate_lat"])
        result.setdefault("candidate_lon", args["candidate_lon"])
        result.setdefault("business_category", str(args["business_category"]))
        result.setdefault("floor_area", args["floor_area"])

    return result


def _compact_result_for_model(result):
    """Trim the tool result before sending it back to the model to save tokens."""
    competitors = result.get("competitors") or []
    return {
        "predicted_visits": result.get("predicted_visits"),
        "market_share": result.get("market_share"),
        "runtime_ms": result.get("runtime_ms"),
        "notes": result.get("notes"),
        "competitor_count": len(competitors),
        "top_competitors": competitors[:5],
    }


@app.route("/api/chat", methods=["POST"])
def api_chat():
    """
    Stateless chat controller.

    Request JSON:
      {
        "messages": [ {role, content, ...} ],   # full prior history (no system msg)
        "selected_location": {"lat": .., "lon": ..} | null
      }

    Response JSON:
      {
        "ok": true,
        "reply": "<assistant text>",
        "messages": [ ...updated history... ],   # store and resend this next turn
        "huff_result": { ...full model output... } | null
      }
    """
    try:
        data = request.get_json(force=True) or {}
        client_messages = data.get("messages") or []
        selected = data.get("selected_location")

        # Build the working conversation: system prompt + optional map context + history.
        convo = [{"role": "system", "content": CHAT_SYSTEM_PROMPT}]
        if selected and selected.get("lat") is not None and selected.get("lon") is not None:
            convo.append({
                "role": "system",
                "content": (
                    f"The user has currently selected this candidate location on the map: "
                    f"latitude {selected['lat']}, longitude {selected['lon']}. Treat these as "
                    f"the candidate coordinates unless the user provides different ones."
                ),
            })
        convo.extend(client_messages)

        new_messages = list(client_messages)  # history we will return to the client
        huff_result = None
        reply_text = None

        # Tool-calling loop, capped to avoid runaway calls.
        for _ in range(5):
            response = client.chat.completions.create(
                model=DEPLOYMENT,
                messages=convo,
                tools=[HUFF_TOOL],
                tool_choice="auto",
                temperature=0.3,
            )
            msg = response.choices[0].message

            assistant_entry = {"role": "assistant", "content": msg.content}
            if msg.tool_calls:
                assistant_entry["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ]
            convo.append(assistant_entry)
            new_messages.append(assistant_entry)

            # No tool call -> this is the final natural-language reply.
            if not msg.tool_calls:
                reply_text = msg.content or ""
                break

            # Execute each requested tool call and feed results back to the model.
            for tc in msg.tool_calls:
                if tc.function.name == "run_huff_model":
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                        result = _execute_run_huff(args)
                        huff_result = result  # keep full result for the front-end
                        tool_payload = _compact_result_for_model(result)
                    except Exception as ex:
                        app.logger.exception("run_huff_model tool failed")
                        tool_payload = {"error": str(ex)}
                else:
                    tool_payload = {"error": f"Unknown tool: {tc.function.name}"}

                tool_entry = {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(tool_payload),
                }
                convo.append(tool_entry)
                new_messages.append(tool_entry)
        else:
            # Loop exhausted without a plain-text reply.
            reply_text = reply_text or (
                "I wasn't able to finish that request. Could you rephrase or restate the inputs?"
            )

        return jsonify({
            "ok": True,
            "reply": reply_text or "",
            "messages": new_messages,
            "huff_result": huff_result,
        })

    except Exception as e:
        app.logger.exception("api_chat failed")
        return jsonify({"ok": False, "error": str(e)}), 500


# -------------------------
# Ask Follow-up Questions  (legacy; superseded by /api/chat)
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
# LLM Functions  (used by the legacy /api/run_huff and /api/ask endpoints)
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


@app.route('/admin/create_tables', methods=['GET', 'POST'])
def admin_create_tables():

    key = request.args.get('key', '')
    if key != '12345678':
        return jsonify({'ok': False, 'error': 'Unauthorized or missing key'}), 401

    sql_path = os.path.join(app.root_path, 'sql', 'create_tables.sql')
    if not os.path.exists(sql_path):
        return jsonify({'ok': False, 'error': f'SQL file not found at {sql_path}'}), 400

    try:
        with open(sql_path, 'r', encoding='utf-8') as f:
            sql_text = f.read()

        # split on GO statements that are on their own line, ignoring case and surrounding whitespace
        blocks = [b.strip() for b in re.split(r'^\s*GO\s*$', sql_text, flags=re.IGNORECASE | re.MULTILINE) if b.strip()]

        with get_connection() as conn:
            cursor = conn.cursor()
            for block in blocks:
                try:
                    cursor.execute(block)
                except Exception as ex:
                    # log the error and the block that caused it, but continue with the next blocks
                    app.logger.exception('Failed to execute SQL block')
                    return jsonify({'ok': False, 'error': f'DDL Execution Failed: {str(ex)}', 'sql_block': block}), 500
            
            conn.commit()

        return jsonify({'ok': True, 'msg': 'Tables created successfully'})

    except Exception as e:
        app.logger.exception('create_tables failed')
        return jsonify({'ok': False, 'error': str(e)}), 500
    
@app.route('/admin/insert_geojson', methods=['GET', 'POST'])
def admin_insert_geojson():
    """
    Read `worcester_cbgs_map.geojson` and insert/update features into the `dbo.cbg_geometries` table.

    Access control: requires query parameter `key=12345678`.
    """
    key = request.args.get('key', '')
    if key != '12345678':
        return jsonify({'ok': False, 'error': 'Unauthorized or missing key'}), 401

    geojson_path = os.path.join(app.root_path, 'static', 'data', 'worcester_cbgs_map.geojson')
    if not os.path.exists(geojson_path):
        return jsonify({'ok': False, 'error': f'GeoJSON not found at {geojson_path}'}), 400

    try:
        with open(geojson_path, 'r', encoding='utf-8') as gf:
            gj = json.load(gf)

        features = gj.get('features', [])
        inserted = 0
        errors = []

        with get_connection() as conn:
            cursor = conn.cursor()
            for feat in features:
                props = feat.get('properties') or {}
                geoid = props.get('GEOID10') or props.get('GEOID') or props.get('geoid')
                geometry = feat.get('geometry')

                if not geoid:
                    errors.append({'reason': 'missing geoid', 'properties': props})
                    continue

                geom_json = json.dumps(geometry, ensure_ascii=False)

                try:
                    # Check existence and perform upsert (update if exists, insert otherwise)
                    cursor.execute('SELECT 1 FROM dbo.cbg_geometries WHERE geoid = ?', (geoid,))
                    if cursor.fetchone():
                        cursor.execute('UPDATE dbo.cbg_geometries SET geometry = ? WHERE geoid = ?', (geom_json, geoid))
                    else:
                        cursor.execute('INSERT INTO dbo.cbg_geometries (geoid, geometry) VALUES (?, ?)', (geom_json, geoid))
                    inserted += 1
                except Exception as ex:
                    # Record per-row insertion error and continue processing other features
                    errors.append({'geoid': geoid, 'error': str(ex)})

            conn.commit()

        return jsonify({'ok': True, 'inserted': inserted, 'errors': errors})

    except Exception as e:
        app.logger.exception('insert_geojson failed')
        return jsonify({'ok': False, 'error': str(e)}), 500

@app.route("/api/get_cbg_map", methods=["GET"])
def get_cbg_map():
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT geoid, geometry FROM cbg_geometries")
        rows = cursor.fetchall()
        conn.close()

        features = []
        for row in rows:
            geoid = row[0]
            geometry = json.loads(row[1])
            features.append({
                "type": "Feature",
                "properties": {"geoid": geoid},
                "geometry": geometry
            })

        return jsonify({
            "type": "FeatureCollection",
            "features": features
        })

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
