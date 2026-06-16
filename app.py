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

_DEMAND_CATEGORIES = None


def _get_demand_categories():
    """Distinct category names that actually have demand data in Worcester (cached).

    The NAICS whitelist promises more categories than the dataset can serve, which is how
    dead-ends like NAICS 4442 slip through. This queries the database once and treats it as
    the source of truth for which categories can produce a non-empty prediction.
    """
    global _DEMAND_CATEGORIES
    if _DEMAND_CATEGORIES is None:
        try:
            with get_connection() as conn:
                cur = conn.cursor()
                cur.execute("SELECT DISTINCT top_category FROM category_demand")
                rows = cur.fetchall()
            _DEMAND_CATEGORIES = {
                str(r[0]).strip().lower() for r in rows if r and r[0] is not None
            }
        except Exception:
            app.logger.exception("Could not load demand categories")
            _DEMAND_CATEGORIES = set()
    return _DEMAND_CATEGORIES


def resolve_naics_code(user_input, naics_whitelist, naics_calibrated, client, deployment):
    """
    Classify a business description into a NAICS code and tier the result.

    The tier (`mark`) is computed in Python -- never trusting the model to self-rate -- and
    uses the database as the source of truth for whether a prediction is even possible:
        mark 0 = calibrated AND has demand data -> confident prediction
        mark 1 = has demand but not calibrated  -> rough prediction (fallback alpha/beta)
        mark 2 = no usable data (no demand, or code not recognized) -> cannot predict
    """
    if not user_input or not user_input.strip():
        return {"ok": False, "error": "No input provided."}

    # Format whitelist into a string for the AI prompt context
    whitelist_text = "\n".join(f"{code}: {name}" for code, name in naics_whitelist.items())

    # The model only needs to return the code and name; Python computes the tier (mark).
    system_prompt = (
        "You are an expert NAICS code classifier. "
        "Analyze the business description and return the best matching NAICS code from the whitelist. "
        "Respond ONLY with a JSON object. Format: "
        '{"naics_code": "4441", "category_name": "..."}'
    )

    try:
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

        parsed = json.loads(raw_content)
    except Exception as e:
        return {"ok": False, "error": f"Model inference or parsing failed: {str(e)}"}

    naics_code = str(parsed.get("naics_code", "")).strip()
    category_name = parsed.get("category_name", naics_whitelist.get(naics_code, "Unknown Category"))

    # Tier the result. The whitelist is broader than the dataset, so being recognized is not
    # enough -- we also confirm the category has demand data before calling it predictable.
    if naics_code in naics_whitelist:
        canonical_name = naics_whitelist[naics_code]
        key = canonical_name.strip().lower()
        has_demand = key in _get_demand_categories()
        # Calibration is keyed by category NAME, not code: the expanded whitelist maps many
        # granular codes (e.g. 444130) onto a calibrated category ("Building Material and
        # Supplies Dealers"), so a code-only check would wrongly downgrade them to mark 1.
        if isinstance(naics_calibrated, dict):
            calibrated_names = {str(v).strip().lower() for v in naics_calibrated.values()}
            is_calibrated = key in calibrated_names
        else:
            is_calibrated = naics_code in naics_calibrated
        if is_calibrated and has_demand:
            mark = 0   # calibrated category + has data
        elif has_demand:
            mark = 1   # data exists but uncalibrated (rough)
        else:
            mark = 2   # recognized but no demand data (e.g. NAICS 4442) -> dead end
    else:
        mark = 2       # not in the whitelist at all

    result = {
        "naics_code": naics_code,
        "category_name": category_name,
        "mark": mark,
        "supported": mark != 2,
    }
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
- If the tool result has "no_demand_data": true, the dataset has no demand or competitor
  data for that category in Worcester. Do NOT interpret the zero visits or zero market
  share as a market signal, and do not claim the location is good or bad. State plainly
  that the dataset has no data for that category, so no prediction can be made, and
  suggest the user try a category the system covers (for example a calibrated NAICS code).
  Never describe the top neighborhoods as "competitors" in this case.
- Worcester is roughly latitude 42.2-42.3 and longitude -71.9 to -71.7. Gently flag
  coordinates that fall well outside this range before running.
"""

# Append the categories that actually have model data, so the controller can steer the
# user to a covered category instead of running a prediction that will come back empty.
CHAT_SYSTEM_PROMPT += (
    "\nSupported categories — predictions are only meaningful for these, because they "
    "have calibrated model data in Worcester:\n"
    + "\n".join(f"  - {name} (NAICS {code})" for code, name in NAICS_CALIBRATED.items())
    + "\n\nIf the user's business is not one of these, tell them it's outside the current "
    "Worcester dataset and offer the closest supported category, instead of running a "
    "prediction that will come back empty.\n"
)

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
        "no_demand_data": result.get("no_demand_data"),
        "total_demand": result.get("total_demand"),
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
