# main.py
import os
import logging
from typing import Optional
from pathlib import Path
from google.cloud import storage

from utils import (
    run_textract_forms,         # F1: (file_path: str, ...) -> dict
    arrange_textract_kv,        # F2: (response, n_sections=3) -> pd.DataFrame [key,value]
    normalize_collateral_value, # F3: (df, ...) -> pd.DataFrame
    model,                      # F4: (df) -> dict (structured JSON)         # F5: (structured_json: dict) -> str
    insert_record_bq,           # BigQuery helper
    move_blob,                   # GCS helper
    coerce_to_string,
    assistant,
    generate_insert_query_collateral,
    run_insert_query
    
)

# -----------------------------
# Fixed configuration
# -----------------------------
BUCKET_NAME = "fusable"
INPUT_PREFIX = "ucc-input-files/"
PROCESSED_PREFIX = "ucc-processed-files/"
ERROR_PREFIX = "ucc-error-files/"
AWS_REGION = "us-east-1"  # region used inside run_textract_forms if needed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _is_tif(name: str) -> bool:
    n = name.lower()
    return n.endswith(".tif") or n.endswith(".tiff")


def _tmp_path_for(blob_name: str) -> str:
    """
    Create a safe local file path under /tmp mirroring the base name of the blob.
    Cloud Functions allow writing to /tmp only.
    """
    base = os.path.basename(blob_name)
    return str(Path("/tmp") / base)


def process_single_image(blob: storage.Blob, storage_client: storage.Client) -> None:
    """
    Full pipeline for a single TIF/TIFF:
      - Download to /tmp
      - Textract -> KV DataFrame
      - Normalize collateral field
      - Model -> structured JSON
      - Build & run BigQuery INSERT
      - Move to processed or error
    """
    src_name = blob.name
    local_path = _tmp_path_for(src_name)

    try:
        logger.info(f"Downloading: gs://{BUCKET_NAME}/{src_name} -> {local_path}")
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(local_path)

        # 1) Textract (F1)
        #    Your run_textract_forms should read the file at `local_path` and return the Textract response.
        response = run_textract_forms(local_path, AWS_REGION)
        

        # 2) Arrange KV pairs (F2) → only [key, value]
        df = arrange_textract_kv(response, n_sections=3)

        # 3) Fix collateral row if needed (F3)
        df = normalize_collateral_value(df)

        # 4) Model to structured JSON (F4)
        df_to_str = coerce_to_string(df)
        print("df to str : ",df_to_str)
        structured = model(df_to_str)

        base_name = os.path.basename(src_name)        # e.g. "12345.tif"
        doc_id_str = os.path.splitext(base_name)[0].strip()   # -> "12345"

        try:
            structured["document_id"] = int(doc_id_str)
        except ValueError:
            structured["document_id"] = None
            

        # 5) Build INSERT query (F5) and run it
        # insert_sql = build_insert_query(structured)
        # print("Query : ",insert_sql)
        insert_record_bq(structured)

        # 6) Move to processed folder (flatten to basename; keep as-is unless you prefer preserving subfolders)
        dest_blob_name = os.path.join(PROCESSED_PREFIX, os.path.basename(src_name))
        move_blob(storage_client, BUCKET_NAME, src_name, dest_blob_name)
        logger.info(f"✅ Processed: {src_name} → {dest_blob_name}")

        # 7)
        collateral = assistant(int(doc_id_str))
        if collateral and collateral.get("collateral_details"):
            query = generate_insert_query_collateral(collateral)
            if query:
                run_insert_query(query)
        else:
            logger.info("No collateral_details returned; skipping collateral insert.")

    except Exception as e:
        logger.exception(f"❌ Error processing {src_name}: {e}")
        # Move to error folder
        dest_blob_name = os.path.join(ERROR_PREFIX, os.path.basename(src_name))
        try:
            move_blob(storage_client, BUCKET_NAME, src_name, dest_blob_name)
            logger.info(f"Moved to error: {src_name} → {dest_blob_name}")
        except Exception as move_err:
            logger.exception(f"Failed moving {src_name} to error folder: {move_err}")
    finally:
        # Cleanup temp file
        try:
            if os.path.exists(local_path):
                os.remove(local_path)
        except Exception:
            pass


def process_all_images(request: Optional[object] = None) -> None:
    """
    Enumerate TIF/TIFF files in INPUT_PREFIX and process them all.
    Usable both locally and as a Cloud Function helper.
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)

    blobs_iter = bucket.list_blobs(prefix=INPUT_PREFIX)
    tif_blobs = [b for b in blobs_iter if _is_tif(b.name)]

    if not tif_blobs:
        logger.info("✅ No files left to process.")
        return

    logger.info(f"Found {len(tif_blobs)} file(s) to process.")
    for blob in tif_blobs:
        logger.info(f"Processing: {blob.name}")
        process_single_image(blob, storage_client)


# -----------------------------
# Cloud Function HTTP entrypoint
# -----------------------------
def main(request):
    process_all_images(request)
    return ("Batch processing completed", 200)


# -----------------------------
# Local testing
# -----------------------------
if __name__ == "__main__":
    process_all_images()
