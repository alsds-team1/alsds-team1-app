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
                        cursor.execute('INSERT INTO dbo.cbg_geometries (geoid, geometry) VALUES (?, ?)', (geoid, geom_json))
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
