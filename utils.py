import os, time
import boto3
from google.cloud import bigquery, storage
from typing import Union
import pandas as pd
import json, ast, re
from openai import OpenAI

# -------- Function 1 --------
def run_textract_forms(file_path: str, region_name: str = "us-east-1"):
    """
    Call AWS Textract on an image/PDF file and return the analyze_document response.

    Args:
        file_path (str): Path to image or PDF (e.g. TIFF).
        region_name (str): AWS region (default "us-east-1").

    Returns:
        dict: Textract response.
    """
    # Read AWS credentials from environment variables 
    aws_access_key_id = os.environ["AWS_ACCESS_KEY_ID"]
    aws_secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"]

    textract = boto3.client(
        "textract",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=region_name,
    )

    with open(file_path, "rb") as f:
        image_bytes = f.read()

    response = textract.analyze_document(
        Document={"Bytes": image_bytes},
        FeatureTypes=["FORMS"],
    )
    return response



# -------- Function 2 --------
def arrange_textract_kv(response, n_sections: int = 3) -> pd.DataFrame:
    """
    Arrange ALL Textract key–value pairs into correct reading order.
    Returns only [key, value] columns.

    Args:
        response (dict | str): Textract analyze_document response
                               (dict, JSON string, or Python repr string).
        n_sections (int): Expected number of vertical sections (default=3).

    Returns:
        pd.DataFrame: ordered DataFrame with columns [key, value].
    """
    # --- Parse response ---
    if isinstance(response, dict):
        resp = response
    elif isinstance(response, str):
        try:
            resp = json.loads(response)
        except json.JSONDecodeError:
            resp = ast.literal_eval(response)
    else:
        raise TypeError("Unsupported response type; expected dict or str")

    blocks = resp.get("Blocks", [])
    by_id = {b["Id"]: b for b in blocks}

    # --- Helper to extract text ---
    def _text_from(block):
        if not block:
            return ""
        words = []
        for rel in block.get("Relationships", []):
            if rel["Type"] == "CHILD":
                for cid in rel["Ids"]:
                    child = by_id.get(cid, {})
                    bt = child.get("BlockType")
                    if bt == "WORD":
                        words.append(child.get("Text", ""))
                    elif bt == "SELECTION_ELEMENT":
                        words.append("[X]" if child.get("SelectionStatus") == "SELECTED" else "[ ]")
                    elif bt == "LINE":
                        words.append(child.get("Text", ""))
        return " ".join(w for w in words if w).strip()

    # --- Collect key–value pairs with geometry ---
    rows = []
    for b in blocks:
        if b.get("BlockType") == "KEY_VALUE_SET" and "KEY" in b.get("EntityTypes", []):
            key_block = b
            value_block = None
            for rel in key_block.get("Relationships", []):
                if rel["Type"] == "VALUE":
                    for vid in rel["Ids"]:
                        vb = by_id.get(vid)
                        if vb and vb.get("BlockType") == "KEY_VALUE_SET":
                            value_block = vb
                            break

            key_text = _text_from(key_block)
            value_text = _text_from(value_block)

            bb = key_block.get("Geometry", {}).get("BoundingBox", {}) or {}
            rows.append({
                "key": key_text,
                "value": value_text,
                "top": float(bb.get("Top", 0.0)),
                "left": float(bb.get("Left", 0.0)),
                "right": float(bb.get("Left", 0.0)) + float(bb.get("Width", 0.0))
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df[["key", "value"]]

    # --- Section assignment (simple k-means 1D) ---
    def _kmeans_1d(values, k=3, iters=20):
        vs = sorted(values)
        cents = [vs[int((i + 0.5) * len(vs) / k)] for i in range(k)]
        for _ in range(iters):
            buckets = [[] for _ in range(k)]
            for v in values:
                j = min(range(k), key=lambda i: abs(v - cents[i]))
                buckets[j].append(v)
            new_cents = [sum(bucket)/len(bucket) if bucket else cents[i] for i, bucket in enumerate(buckets)]
            if max(abs(a-b) for a, b in zip(cents, new_cents)) < 1e-8:
                break
            cents = new_cents
        order = sorted(range(k), key=lambda i: cents[i])
        rank = {i: r+1 for r, i in enumerate(order)}
        labels = []
        for v in values:
            j = min(range(k), key=lambda i: abs(v - cents[i]))
            labels.append(rank[j])
        return labels

    if len(df) >= max(2, n_sections):
        df["section"] = _kmeans_1d(df["top"].tolist(), k=n_sections)
    else:
        df["section"] = 1

    # --- Sort and return only key/value ---
    df = df.sort_values(["section", "top", "left", "right"]).reset_index(drop=True)
    return df[["key", "value"]]


def normalize_collateral_value(
    df: pd.DataFrame,
    target_key: str = "4. COLLATERAL: This financing statement covers the following collateral:",
    canonicalize_key: bool = True
) -> pd.DataFrame:
    """
    If the collateral key contains extra text after 'collateral:', move that extra text
    to the *front* of the value. Also remove the boilerplate phrase
    'This financing statement covers the following collateral:' from the start of the value
    (regardless of whether it came from the key or was already in the value).
    """
    if not {"key", "value"}.issubset(df.columns):
        raise ValueError("DataFrame must have 'key' and 'value' columns")

    def _norm(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "").strip().lower())

    def _strip_boilerplate(text: str) -> str:
        # Remove common boilerplate at the *start* of the value, tolerant to minor typos
        # e.g. "This financing statement cover(s) the following collateral:"
        pattern = r'^\s*this\s+financing\s+statement\s+cover[s]?\s+the\s+following\s+collateral:\s*'
        return re.sub(pattern, "", (text or ""), flags=re.IGNORECASE).strip()

    target_norm = _norm(target_key)

    out = df.copy()
    for i, row in out.iterrows():
        key_raw = str(row["key"] or "")
        key_norm = _norm(key_raw)

        # Consider it a match if the row's key STARTS with the canonical key (allows extra trailing text)
        if key_norm.startswith(target_norm):
            # Capture any extra text after the last 'collateral:' in the *key*
            m = re.search(r"(.*?\bcollateral:\s*)(.*)$", key_raw, flags=re.IGNORECASE | re.DOTALL)
            current_val = str(row["value"] or "").strip()

            if m:
                suffix = m.group(2).strip()  # extra text after 'collateral:' in key
                if suffix:
                    # Prepend the extra description to the value (avoid duplication)
                    if _norm(current_val).startswith(_norm(suffix)):
                        new_val = current_val
                    else:
                        new_val = f"{suffix} {current_val}".strip()
                else:
                    new_val = current_val
            else:
                new_val = current_val

            # Strip boilerplate from the *start* of the final value
            new_val = _strip_boilerplate(new_val)

            out.at[i, "value"] = new_val
            if canonicalize_key:
                out.at[i, "key"] = target_key.strip()

        else:
            # Not a collateral key, but value might (rarely) contain the boilerplate alone — clean it anyway
            if _norm(key_raw).startswith("collateral"):
                out.at[i, "value"] = _strip_boilerplate(str(row["value"] or ""))

    return out


def coerce_to_string(df_or_str: Union[pd.DataFrame, str]) -> str:
    """
    Convert a [key, value] DataFrame (preferred) or a string into a compact string payload.
    """
    if isinstance(df_or_str, pd.DataFrame):
        # Ensure expected columns exist; reorder if needed
        cols = [c.lower() for c in df_or_str.columns]
        if "key" not in cols or "value" not in cols:
            raise ValueError("model(df): DataFrame must have columns ['key','value'].")
        # Keep only key/value and stringify as JSON lines for robustness
        df2 = df_or_str.copy()
        df2.columns = [c.lower() for c in df2.columns]
        df2 = df2[["key", "value"]]
        return df2.to_json(orient="records", force_ascii=False)
    elif isinstance(df_or_str, str):
        return df_or_str
    else:
        raise TypeError("model(df): argument must be a pandas.DataFrame or str.")


def model(df_str: str):
    """
    Normalize a dataframe string into the strict 49-field JSON schema
    using a saved OpenAI prompt template.

    Args:
        df_str (str): String representation of the dataframe (keys/values).

    Returns:
        dict: JSON object with all 49 schema fields.
    """
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.responses.create(
        prompt={
            "id": "pmpt_68d27e617cd48197939d0164444d8ed705d05d15adccabb2",
            "version": "5"
        },
        input=df_str
    )

    # Extract and return the model’s JSON output
    try:
        return resp.output_parsed
    except Exception:
        text = resp.output[0].content[0].text
        return json.loads(text)

# ----------------------------
#  GCP Helpers
# ----------------------------
def move_blob(storage_client, bucket_name, source_blob_name, destination_blob_name):
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    bucket.copy_blob(blob, bucket, destination_blob_name)
    blob.delete()
    print(f"Moved {source_blob_name} to {destination_blob_name}")
    

from typing import Dict, Any
from google.cloud import bigquery

_TABLE_FQN = "genai-poc-424806.Fusable.raw_data"

def insert_record_bq(record: Dict[str, Any]) -> None:
    """
    Insert one JSON record into BigQuery using streaming insert.
    Avoids SQL quoting problems entirely.
    """
    client = bigquery.Client()
    errors = client.insert_rows_json(_TABLE_FQN, [record])
    if errors:
        # Propagate so caller moves the file to error/
        raise RuntimeError(f"BigQuery insert failed: {errors}")

bq_client = bigquery.Client()

def sql_executor(sql: str):
    """Run a BigQuery SQL and return rows as list of dicts.
    
    """
    if not sql or not sql.strip():
        raise ValueError("Expected a non-empty SQL string.")
    job = bq_client.query(sql)
    result = job.result()
    return [dict(row.items()) for row in result]



def assistant(user_input: int):
    """
    Run a conversation with the OpenAI Assistant.
    Handles tool calls (sql_executor) until assistant completes or hits a terminal state.
    Returns the assistant's final text (expected JSON) or a structured error JSON.
    """
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    thread = client.beta.threads.create()

    # Send user message (document_id integer is expected by your system prompt)
    client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=json.dumps({"document_id": int(user_input)})
    )

    run = client.beta.threads.runs.create(thread_id=thread.id, assistant_id=os.environ["ASSISTANT_ID"])

    TERMINAL = {"completed", "failed", "cancelled", "expired"}
    start_ts = time.time()

    while True:
        time.sleep(1.0)
        run_status = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
        status = run_status.status
        print("Run Status:\n", run_status.model_dump_json(indent=4))

        # Bail out on terminal states
        if status in TERMINAL:
            break

        if status == "requires_action":
            required = run_status.required_action.submit_tool_outputs.model_dump()
            tool_outputs = []

            for action in required.get("tool_calls", []):
                func_name = action["function"]["name"]
                args = json.loads(action["function"]["arguments"] or "{}")
                print(f"Function: {func_name}, Arguments: {args}")

                if func_name in ("sql_executor", "sql_executer"):  # accept both spellings
                    result = sql_executor(sql=args["sql"])  # uses your existing executor
                    output_str = json.dumps(result, ensure_ascii=False)
                else:
                    output_str = json.dumps({"error": f"Unknown tool '{func_name}'"}, ensure_ascii=False)

                tool_outputs.append({"tool_call_id": action["id"], "output": output_str})

            client.beta.threads.runs.submit_tool_outputs(
                thread_id=thread.id,
                run_id=run.id,
                tool_outputs=tool_outputs
            )

        # Optional overall timeout (e.g., 120s)
        if time.time() - start_ts > 120:
            return {
                "document_id": user_input,
                "collateral_details": [],
                "errors": ["ASSISTANT_TIMEOUT"]
            }

    # Terminal state reached
    if status != "completed":
        # Map to your error contract
        return {
            "document_id": user_input,
            "collateral_details": [],
            "errors": ["ASSISTANT_RUN_ERROR", status.upper()]
        }

    # Return final assistant message text (expected to be JSON)
    msgs = client.beta.threads.messages.list(thread_id=thread.id)
    for msg in msgs.data:
        if msg.role == "assistant":
            text = "".join(
                part.text.value for part in msg.content if getattr(part, "type", "") == "text"
            ).strip()
            try:
                return json.loads(text)
            except Exception:
                return text

    return {
        "document_id": user_input,
        "collateral_details": [],
        "errors": ["NO_ASSISTANT_MESSAGE"]
    }


def generate_insert_query_collateral(data: dict) -> str:
    table_name = "`genai-poc-424806.Fusable.ucc_collateral`"
    # Define columns once, in order
    col_order = [
        "Document_Id", "Year", "Manufacturer", "Model", "Equip Description",
        "serial number", "vin", "additional notes", "Debtor Name",
        "Secured Party Name", "Address"
    ]

    rows = []
    details = data.get("collateral_details") or []
    if not details:
        # Nothing to insert—return a noop (or raise) so caller can decide
        return ""

    for item in details:
        # Enforce Document_Id from top-level
        row_dict = {
            "Document_Id": data.get("document_id"),
            "Year": item.get("Year"),
            "Manufacturer": item.get("Manufacturer"),
            "Model": item.get("Model"),
            "Equip Description": item.get("Equip Description"),
            "serial number": item.get("serial number"),
            "vin": item.get("vin"),
            "additional notes": item.get("additional notes"),
            "Debtor Name": item.get("Debtor Name"),
            "Secured Party Name": item.get("Secured Party Name"),
            "Address": item.get("Address"),
        }

        sql_values = []
        for key in col_order:
            value = row_dict.get(key)
            if value is None or value == "":
                sql_values.append("NULL")
            elif key in ("Document_Id", "Year"):
                sql_values.append(str(int(value)))
            else:
                safe = str(value).replace("'", "''")  # BigQuery: double quotes for escaping
                sql_values.append(f"'{safe}'")
        rows.append(f"({', '.join(sql_values)})")

    columns = ", ".join(f"`{c}`" for c in col_order)
    values_clause = ",\n".join(rows)
    query = f"INSERT INTO {table_name} ({columns})\nVALUES\n{values_clause};"
    print("Collateral Insert Query:\n", query)
    return query


def run_insert_query(query: str) -> None:
    """
    Executes an INSERT SQL query in BigQuery.
    
    Args:
        query (str): The full SQL insert statement.
    
    Raises:
        google.api_core.exceptions.GoogleAPIError: On query failure.
    """
    client = bigquery.Client()
    
    try:
        query_job = client.query(query)
        query_job.result()  # Wait for job to finish
        print("✅ Insert successful.")
    except Exception as e:
        print("❌ Insert failed:", str(e))