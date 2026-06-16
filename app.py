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

NAICS_CALIBRATED = {
    "3399": "Other Miscellaneous Manufacturing",
    "4441": "Building Material and Supplies Dealers",
    "6214": "Outpatient Care Centers",
    "311811": "Bakeries and Tortilla Manufacturing",
    "441310": "Automotive Parts, Accessories, and Tire Stores",
    "445310": "Beer, Wine, and Liquor Stores",
    "447110": "Gasoline Stations",
    "448310": "Jewelry, Luggage, and Leather Goods Stores",
    "452319": "General Merchandise Stores, including Warehouse Clubs and Supercenters",
    "453991": "Other Miscellaneous Store Retailers",
    "512240": "Sound Recording Industries",
    "517312": "Wired and Wireless Telecommunications Carriers",
    "522110": "Depository Credit Intermediation",
    "522310": "Activities Related to Credit Intermediation",
    "523930": "Other Financial Investment Activities",
    "524113": "Insurance Carriers",
    "531120": "Lessors of Real Estate",
    "531210": "Offices of Real Estate Agents and Brokers",
    "611310": "Colleges, Universities, and Professional Schools",
    "621210": "Offices of Dentists",
    "621511": "Medical and Diagnostic Laboratories",
    "812910": "Other Personal Services",
    "922110": "Justice, Public Order, and Safety Activities"
}

NAICS_FALLBACK = {
    "485": "Transit and Ground Passenger Transportation",
    "487": "Scenic and Sightseeing Transportation",
    "562": "Waste Management and Remediation Services",
    "623": "Nursing and Residential Care Facilities",
    "2382": "Building Equipment Contractors",
    "2383": "Building Finishing Contractors",
    "2389": "Other Specialty Trade Contractors",
    "3231": "Printing and Related Support Activities",
    "4238": "Machinery, Equipment, and Supplies Merchant Wholesalers",
    "4422": "Home Furnishings Stores",
    "4442": "Lawn and Garden Equipment and Supplies Stores",
    "4481": "Clothing Stores",
    "5151": "Radio and Television Broadcasting",
    "5412": "Accounting, Tax Preparation, Bookkeeping, and Payroll Services",
    "5414": "Specialized Design Services",
    "5416": "Management, Scientific, and Technical Consulting Services",
    "5418": "Advertising, Public Relations, and Related Services",
    "5616": "Investigation and Security Services",
    "6115": "Technical and Trade Schools",
    "6215": "Medical and Diagnostic Laboratories",
    "6233": "Continuing Care Retirement Communities and Assisted Living Facilities for the Elderly",
    "6241": "Individual and Family Services",
    "7111": "Performing Arts Companies",
    "8111": "Automotive Repair and Maintenance",
    "8122": "Death Care Services",
    "9231": "Administration of Human Resource Programs",
    "9261": "Administration of Economic Programs",
    "54192": "Other Professional, Scientific, and Technical Services",
    "81211": "Personal Care Services",
    "221111": "Electric Power Generation, Transmission and Distribution",
    "237110": "Utility System Construction",
    "238140": "Foundation, Structure, and Building Exterior Contractors",
    "238150": "Foundation, Structure, and Building Exterior Contractors",
    "238220": "Building Equipment Contractors",
    "238330": "Building Finishing Contractors",
    "238390": "Building Finishing Contractors",
    "312120": "Beverage Manufacturing",
    "312130": "Beverage Manufacturing",
    "323113": "Printing and Related Support Activities",
    "323117": "Printing and Related Support Activities",
    "335220": "Household Appliance Manufacturing",
    "339950": "Other Miscellaneous Manufacturing",
    "423330": "Lumber and Other Construction Materials Merchant Wholesalers",
    "423450": "Professional and Commercial Equipment and Supplies Merchant Wholesalers",
    "423610": "Household Appliances and Electrical and Electronic Goods Merchant Wholesalers",
    "423690": "Household Appliances and Electrical and Electronic Goods Merchant Wholesalers",
    "423720": "Hardware, and Plumbing and Heating Equipment and Supplies Merchant Wholesalers",
    "423730": "Hardware, and Plumbing and Heating Equipment and Supplies Merchant Wholesalers",
    "423740": "Hardware, and Plumbing and Heating Equipment and Supplies Merchant Wholesalers",
    "423820": "Machinery, Equipment, and Supplies Merchant Wholesalers",
    "423830": "Machinery, Equipment, and Supplies Merchant Wholesalers",
    "423850": "Machinery, Equipment, and Supplies Merchant Wholesalers",
    "423910": "Miscellaneous Durable Goods Merchant Wholesalers",
    "424210": "Drugs and Druggists' Sundries Merchant Wholesalers",
    "441110": "Automobile Dealers",
    "441120": "Automobile Dealers",
    "441222": "Other Motor Vehicle Dealers",
    "441228": "Other Motor Vehicle Dealers",
    "441320": "Automotive Parts, Accessories, and Tire Stores",
    "442110": "Furniture Stores",
    "442210": "Home Furnishings Stores",
    "442299": "Home Furnishings Stores",
    "443141": "Electronics and Appliance Stores",
    "443142": "Electronics and Appliance Stores",
    "444110": "Building Material and Supplies Dealers",
    "444120": "Building Material and Supplies Dealers",
    "444130": "Building Material and Supplies Dealers",
    "444190": "Building Material and Supplies Dealers",
    "445110": "Grocery Stores",
    "445120": "Grocery Stores",
    "445210": "Specialty Food Stores",
    "445220": "Specialty Food Stores",
    "445230": "Specialty Food Stores",
    "445292": "Specialty Food Stores",
    "445299": "Specialty Food Stores",
    "446110": "Health and Personal Care Stores",
    "446120": "Health and Personal Care Stores",
    "446191": "Health and Personal Care Stores",
    "446199": "Health and Personal Care Stores",
    "448140": "Clothing Stores",
    "448190": "Clothing Stores",
    "448210": "Shoe Stores",
    "448320": "Jewelry, Luggage, and Leather Goods Stores",
    "451110": "Sporting Goods, Hobby, and Musical Instrument Stores",
    "451120": "Sporting Goods, Hobby, and Musical Instrument Stores",
    "451130": "Sporting Goods, Hobby, and Musical Instrument Stores",
    "451140": "Sporting Goods, Hobby, and Musical Instrument Stores",
    "451211": "Book Stores and News Dealers",
    "452210": "Department Stores",
    "452311": "General Merchandise Stores, including Warehouse Clubs and Supercenters",
    "453110": "Florists",
    "453210": "Office Supplies, Stationery, and Gift Stores",
    "453220": "Office Supplies, Stationery, and Gift Stores",
    "453310": "Used Merchandise Stores",
    "453910": "Other Miscellaneous Store Retailers",
    "453920": "Other Miscellaneous Store Retailers",
    "453998": "Other Miscellaneous Store Retailers",
    "484210": "Specialized Freight Trucking",
    "485210": "Interurban and Rural Bus Transportation",
    "485310": "Taxi and Limousine Service",
    "485999": "Other Transit and Ground Passenger Transportation",
    "488119": "Support Activities for Air Transportation",
    "488190": "Support Activities for Air Transportation",
    "488410": "Support Activities for Road Transportation",
    "488510": "Freight Transportation Arrangement",
    "491110": "Postal Service",
    "492110": "Couriers and Express Delivery Services",
    "512131": "Motion Picture and Video Industries",
    "515210": "Cable and Other Subscription Programming",
    "518210": "Data Processing, Hosting, and Related Services",
    "519120": "Other Information Services",
    "522130": "Depository Credit Intermediation",
    "522298": "Nondepository Credit Intermediation",
    "522390": "Activities Related to Credit Intermediation",
    "523999": "Other Financial Investment Activities",
    "524210": "Agencies, Brokerages, and Other Insurance Related Activities",
    "531110": "Lessors of Real Estate",
    "531130": "Lessors of Real Estate",
    "531190": "Lessors of Real Estate",
    "531311": "Activities Related to Real Estate",
    "532111": "Automotive Equipment Rental and Leasing",
    "532120": "Automotive Equipment Rental and Leasing",
    "532282": "Consumer Goods Rental",
    "532284": "Consumer Goods Rental",
    "532289": "Consumer Goods Rental",
    "532310": "General Rental Centers",
    "532412": "Commercial and Industrial Machinery and Equipment Rental and Leasing",
    "532490": "Commercial and Industrial Machinery and Equipment Rental and Leasing",
    "541120": "Legal Services",
    "541213": "Accounting, Tax Preparation, Bookkeeping, and Payroll Services",
    "541219": "Accounting, Tax Preparation, Bookkeeping, and Payroll Services",
    "541940": "Other Professional, Scientific, and Technical Services",
    "551114": "Management of Companies and Enterprises",
    "561320": "Employment Services",
    "561720": "Services to Buildings and Dwellings",
    "561730": "Services to Buildings and Dwellings",
    "561790": "Services to Buildings and Dwellings",
    "562211": "Waste Treatment and Disposal",
    "611110": "Elementary and Secondary Schools",
    "611210": "Junior Colleges",
    "611511": "Technical and Trade Schools",
    "611519": "Technical and Trade Schools",
    "611620": "Other Schools and Instruction",
    "611630": "Other Schools and Instruction",
    "611691": "Other Schools and Instruction",
    "611692": "Other Schools and Instruction",
    "611699": "Other Schools and Instruction",
    "621111": "Offices of Physicians",
    "621112": "Offices of Physicians",
    "621310": "Offices of Other Health Practitioners",
    "621320": "Offices of Other Health Practitioners",
    "621330": "Offices of Other Health Practitioners",
    "621340": "Offices of Other Health Practitioners",
    "621399": "Offices of Other Health Practitioners",
    "621420": "Outpatient Care Centers",
    "621492": "Outpatient Care Centers",
    "621493": "Outpatient Care Centers",
    "621498": "Outpatient Care Centers",
    "621610": "Home Health Care Services",
    "621991": "Other Ambulatory Health Care Services",
    "622110": "General Medical and Surgical Hospitals",
    "622210": "Psychiatric and Substance Abuse Hospitals",
    "622310": "Specialty (except Psychiatric and Substance Abuse) Hospitals",
    "623110": "Nursing Care Facilities (Skilled Nursing Facilities)",
    "623312": "Continuing Care Retirement Communities and Assisted Living Facilities for the Elderly",
    "624110": "Individual and Family Services",
    "624120": "Individual and Family Services",
    "624190": "Individual and Family Services",
    "624221": "Community Food and Housing, and Emergency and Other Relief Services",
    "624410": "Child Day Care Services",
    "711211": "Spectator Sports",
    "711310": "Promoters of Performing Arts, Sports, and Similar Events",
    "712110": "Museums, Historical Sites, and Similar Institutions",
    "712120": "Museums, Historical Sites, and Similar Institutions",
    "712130": "Museums, Historical Sites, and Similar Institutions",
    "712190": "Museums, Historical Sites, and Similar Institutions",
    "713110": "Amusement Parks and Arcades",
    "713210": "Gambling Industries",
    "713910": "Other Amusement and Recreation Industries",
    "713940": "Other Amusement and Recreation Industries",
    "713950": "Other Amusement and Recreation Industries",
    "713990": "Other Amusement and Recreation Industries",
    "721110": "Traveler Accommodation",
    "722320": "Special Food Services",
    "722410": "Drinking Places (Alcoholic Beverages)",
    "722511": "Restaurants and Other Eating Places",
    "722513": "Restaurants and Other Eating Places",
    "722515": "Restaurants and Other Eating Places",
    "811111": "Automotive Repair and Maintenance",
    "811121": "Automotive Repair and Maintenance",
    "811122": "Automotive Repair and Maintenance",
    "811191": "Automotive Repair and Maintenance",
    "811192": "Automotive Repair and Maintenance",
    "811198": "Automotive Repair and Maintenance",
    "811211": "Electronic and Precision Equipment Repair and Maintenance",
    "811412": "Personal and Household Goods Repair and Maintenance",
    "811420": "Personal and Household Goods Repair and Maintenance",
    "811430": "Personal and Household Goods Repair and Maintenance",
    "811490": "Personal and Household Goods Repair and Maintenance",
    "812112": "Personal Care Services",
    "812191": "Personal Care Services",
    "812199": "Personal Care Services",
    "812210": "Death Care Services",
    "812220": "Death Care Services",
    "812320": "Drycleaning and Laundry Services",
    "812930": "Other Personal Services",
    "812990": "Other Personal Services",
    "813110": "Religious Organizations",
    "813219": "Grantmaking and Giving Services",
    "813410": "Civic and Social Organizations",
    "922120": "Justice, Public Order, and Safety Activities",
    "922160": "Justice, Public Order, and Safety Activities",
    "926120": "Administration of Economic Programs"
}

NAICS_WHITELIST = {**NAICS_CALIBRATED, **NAICS_FALLBACK}


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
# help function
# -------------------------

def resolve_naics_code(user_input, naics_whitelist, naics_calibrated, client, deployment):
    """
    Core logic function to classify business descriptions into NAICS codes.
    
    Args:
        user_input (str): The raw business description provided by the user.
        naics_whitelist (dict): A dictionary of authorized NAICS codes and names.
        naics_calibrated (set/list): A collection of NAICS codes with pre-calibrated model parameters.
        client: The AI model client instance (e.g., OpenAI/Azure client).
        deployment (str): The specific model or deployment identifier.

    Returns:
        dict: A structured dictionary containing 'ok' status, and 'data' or 'error' message.
    """
    if not user_input or not user_input.strip():
        return {"ok": False, "error": "No input provided."}

    # Format whitelist into a string for the AI prompt context
    whitelist_text = "\n".join(f"{code}: {name}" for code, name in naics_whitelist.items())

    # Define system instructions for the classification task
    system_prompt = (
        "You are an expert NAICS code classifier. "
        "Analyze the business description and return the best matching NAICS code from the whitelist. "
        "Respond ONLY with a JSON object. Format: "
        '{"naics_code": "4441", "category_name": "...", "confidence": "high"}'
        "If no match is reasonable, set confidence to 'low'."
    )

    try:
        # Request completion from the model
        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Business description: {user_input}\n\nWhitelist:\n{whitelist_text}"}
            ],
            temperature=0.1
        )

        raw_content = response.choices[0].message.content.strip()
        
        # Clean markdown formatting if present in the model output
        if "```" in raw_content:
            raw_content = raw_content.replace("```json", "").replace("```", "").strip()

        # Parse AI response to JSON
        parsed = json.loads(raw_content)
    except Exception as e:
        return {"ok": False, "error": f"Model inference or parsing failed: {str(e)}"}

    naics_code = str(parsed.get("naics_code", "")).strip()
    
    # Validate result against the authorized whitelist
    if naics_code not in naics_whitelist:
        return {"ok": False, "error": "No matching records found for this business type."}

    # Determine metadata based on classification results
    category_name = parsed.get("category_name", naics_whitelist[naics_code])
    confidence = parsed.get("confidence", "low")
    is_fallback = naics_code not in naics_calibrated

    # Construct success response structure
    result = {
        "naics_code": naics_code,
        "category_name": category_name,
        "is_fallback": is_fallback
    }
    
    # Flag low confidence matches for manual verification
    if confidence == "low":
        result["warning"] = "Low confidence match. Please confirm this category."

    return {"ok": True, "data": result}

@app.route("/api/trans_naics", methods=["POST"])
def trans_naics():
    # 1. Retrieve data
    data = request.get_json(silent=True) or {}
    user_input = data.get("user_input", "")

    # 2. Invoke the classifier function
    res = resolve_naics_code(
        user_input, 
        NAICS_WHITELIST, 
        NAICS_CALIBRATED, 
        client, 
        DEPLOYMENT
    )

    # 3. Handle response
    if not res["ok"]:
        return jsonify({"ok": False, "error": res["error"]}), 400
    
    return jsonify({"ok": True, **res["data"]})


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


# @app.route('/admin/create_tables', methods=['GET', 'POST'])
# def admin_create_tables():

#     key = request.args.get('key', '')
#     if key != '12345678':
#         return jsonify({'ok': False, 'error': 'Unauthorized or missing key'}), 401

#     sql_path = os.path.join(app.root_path, 'sql', 'create_tables.sql')
#     if not os.path.exists(sql_path):
#         return jsonify({'ok': False, 'error': f'SQL file not found at {sql_path}'}), 400

#     try:
#         with open(sql_path, 'r', encoding='utf-8') as f:
#             sql_text = f.read()

#         # split on GO statements that are on their own line, ignoring case and surrounding whitespace
#         blocks = [b.strip() for b in re.split(r'^\s*GO\s*$', sql_text, flags=re.IGNORECASE | re.MULTILINE) if b.strip()]

#         with get_connection() as conn:
#             cursor = conn.cursor()
#             for block in blocks:
#                 try:
#                     cursor.execute(block)
#                 except Exception as ex:
#                     # log the error and the block that caused it, but continue with the next blocks
#                     app.logger.exception('Failed to execute SQL block')
#                     return jsonify({'ok': False, 'error': f'DDL Execution Failed: {str(ex)}', 'sql_block': block}), 500
            
#             conn.commit()

#         return jsonify({'ok': True, 'msg': 'Tables created successfully'})

#     except Exception as e:
#         app.logger.exception('create_tables failed')
#         return jsonify({'ok': False, 'error': str(e)}), 500
    
# @app.route('/admin/insert_geojson', methods=['GET', 'POST'])
# def admin_insert_geojson():
#     """
#     Read `worcester_cbgs_map.geojson` and insert/update features into the `dbo.cbg_geometries` table.

#     Access control: requires query parameter `key=12345678`.
#     """
#     key = request.args.get('key', '')
#     if key != '12345678':
#         return jsonify({'ok': False, 'error': 'Unauthorized or missing key'}), 401

#     geojson_path = os.path.join(app.root_path, 'static', 'data', 'worcester_cbgs_map.geojson')
#     if not os.path.exists(geojson_path):
#         return jsonify({'ok': False, 'error': f'GeoJSON not found at {geojson_path}'}), 400

#     try:
#         with open(geojson_path, 'r', encoding='utf-8') as gf:
#             gj = json.load(gf)

#         features = gj.get('features', [])
#         inserted = 0
#         errors = []

#         with get_connection() as conn:
#             cursor = conn.cursor()
#             for feat in features:
#                 props = feat.get('properties') or {}
#                 geoid = props.get('GEOID10') or props.get('GEOID') or props.get('geoid')
#                 geometry = feat.get('geometry')

#                 if not geoid:
#                     errors.append({'reason': 'missing geoid', 'properties': props})
#                     continue

#                 geom_json = json.dumps(geometry, ensure_ascii=False)

#                 try:
#                     # Check existence and perform upsert (update if exists, insert otherwise)
#                     cursor.execute('SELECT 1 FROM dbo.cbg_geometries WHERE geoid = ?', (geoid,))
#                     if cursor.fetchone():
#                         cursor.execute('UPDATE dbo.cbg_geometries SET geometry = ? WHERE geoid = ?', (geom_json, geoid))
#                     else:
#                         cursor.execute('INSERT INTO dbo.cbg_geometries (geoid, geometry) VALUES (?, ?)', (geoid, geom_json))
#                     inserted += 1
#                 except Exception as ex:
#                     # Record per-row insertion error and continue processing other features
#                     errors.append({'geoid': geoid, 'error': str(ex)})

#             conn.commit()

#         return jsonify({'ok': True, 'inserted': inserted, 'errors': errors})

#     except Exception as e:
#         app.logger.exception('insert_geojson failed')
#         return jsonify({'ok': False, 'error': str(e)}), 500

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
