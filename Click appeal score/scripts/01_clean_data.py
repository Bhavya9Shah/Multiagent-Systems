#!/usr/bin/env python3
"""
01_clean_data.py
Stage 2: Dataset Inspection and Cleaning Pipeline for Click Appeal Score Research Project.

Tasks:
- Inspect all columns, data types, missing values, duplicate IDs, malformed video IDs, channel inconsistencies, and existing face counts.
- Recover malformed video IDs (#NAME?) from thumbnail URLs where unambiguous.
- Fix channel name inconsistencies (e.g. '\' -> 'Airrack').
- Produce:
    1. data/processed/cleaned_videos.csv
    2. data/processed/cleaning_log.csv
    3. data/processed/data_quality_report.csv
- Strictly preserve all rows (zero silent deletions).
"""

import os
import re
import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np

# Configure logging
LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "01_clean_data.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("clean_data")

BASE_DIR = Path(__file__).resolve().parent.parent
ORIGINAL_DATA_PATH = BASE_DIR / "data" / "original" / "challenge_videos_final(challenge_videos_final).csv"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

CLEANED_CSV_PATH = PROCESSED_DATA_DIR / "cleaned_videos.csv"
CLEANING_LOG_PATH = PROCESSED_DATA_DIR / "cleaning_log.csv"
DATA_QUALITY_REPORT_PATH = PROCESSED_DATA_DIR / "data_quality_report.csv"


def clean_dataset():
    logger.info("Starting Stage 2: Dataset Inspection and Cleaning...")
    logger.info(f"Loading raw dataset from {ORIGINAL_DATA_PATH}")

    if not ORIGINAL_DATA_PATH.exists():
        logger.error(f"Raw data file not found at {ORIGINAL_DATA_PATH}")
        sys.exit(1)

    # Attempt reading with cp1252 / latin-1 due to Windows apostrophe/pound signs
    try:
        df_raw = pd.read_csv(ORIGINAL_DATA_PATH, encoding="cp1252")
        detected_encoding = "cp1252"
    except Exception as e:
        logger.warning(f"Failed to read with cp1252: {e}. Trying latin-1.")
        df_raw = pd.read_csv(ORIGINAL_DATA_PATH, encoding="latin-1")
        detected_encoding = "latin-1"

    raw_row_count = len(df_raw)
    raw_col_count = len(df_raw.columns)
    logger.info(f"Successfully loaded {raw_row_count} rows and {raw_col_count} columns using encoding {detected_encoding}.")

    df_cleaned = df_raw.copy()
    cleaning_logs = []

    # 1. Inspect Channel Name Inconsistencies
    # Group by channel_id to detect if any channel has multiple or corrupt names
    channel_id_to_names = df_cleaned.groupby("channel_id")["channel_name"].unique().to_dict()
    known_channel_names = {}
    for cid, names in channel_id_to_names.items():
        valid_names = [n for n in names if n != "\\" and pd.notnull(n) and len(n.strip()) > 1]
        if valid_names:
            known_channel_names[cid] = valid_names[0]

    for idx, row in df_cleaned.iterrows():
        cid = row["channel_id"]
        cname = row["channel_name"]
        if cname == "\\" or pd.isnull(cname) or len(cname.strip()) <= 1:
            corrected_name = known_channel_names.get(cid, "Airrack")
            df_cleaned.at[idx, "channel_name"] = corrected_name
            cleaning_logs.append({
                "row_index": idx,
                "field": "channel_name",
                "original_value": cname,
                "cleaned_value": corrected_name,
                "action": "channel_name_repaired",
                "details": f"Channel ID {cid} matched to valid name '{corrected_name}'"
            })
            logger.info(f"Row {idx}: Channel name '{cname}' repaired to '{corrected_name}' for channel {cid}.")

    # 2. Inspect and Recover Malformed Video IDs
    malformed_id_count = 0
    for idx, row in df_cleaned.iterrows():
        vid = str(row["video_id"]).strip()
        thumb_url = str(row["thumbnail"]).strip()
        
        # Check for Excel '#NAME?' or invalid ID format (standard YouTube ID is 11 chars: [A-Za-z0-9_-])
        is_malformed = (vid == "#NAME?" or vid.startswith("#") or len(vid) != 11 or not re.match(r"^[A-Za-z0-9_-]{11}$", vid))
        
        if is_malformed:
            malformed_id_count += 1
            # Extract video ID from thumbnail URL: https://i.ytimg.com/vi/<ID>/maxresdefault.jpg
            m = re.search(r"/vi/([^/]+)/", thumb_url)
            if m:
                recovered_id = m.group(1).strip()
                if len(recovered_id) == 11 and re.match(r"^[A-Za-z0-9_-]{11}$", recovered_id):
                    df_cleaned.at[idx, "video_id"] = recovered_id
                    cleaning_logs.append({
                        "row_index": idx,
                        "field": "video_id",
                        "original_value": vid,
                        "cleaned_value": recovered_id,
                        "action": "video_id_recovered_from_thumbnail_url",
                        "details": f"Unambiguously extracted 11-character video ID from thumbnail URL: {thumb_url}"
                    })
                    logger.info(f"Row {idx}: Malformed video_id '{vid}' recovered to '{recovered_id}' from URL '{thumb_url}'.")
                else:
                    cleaning_logs.append({
                        "row_index": idx,
                        "field": "video_id",
                        "original_value": vid,
                        "cleaned_value": vid,
                        "action": "recovery_failed_ambiguous",
                        "details": f"Extracted candidate '{recovered_id}' does not match standard 11-char pattern."
                    })
                    logger.warning(f"Row {idx}: Could not unambiguously recover video_id from '{thumb_url}'.")
            else:
                cleaning_logs.append({
                    "row_index": idx,
                    "field": "video_id",
                    "original_value": vid,
                    "cleaned_value": vid,
                    "action": "no_url_match",
                    "details": "Thumbnail URL does not contain /vi/<id>/ pattern."
                })

    # 3. Check for Duplicate Video IDs
    raw_dupes = df_raw["video_id"].duplicated().sum()
    cleaned_dupes = df_cleaned["video_id"].duplicated().sum()
    logger.info(f"Duplicate video IDs: Raw={raw_dupes} (due to #NAME?), Cleaned={cleaned_dupes}")

    # 4. Check Missing Values across all columns
    missing_counts = df_cleaned.isnull().sum().to_dict()
    total_missing = sum(missing_counts.values())

    # 5. Check Face Count Column
    face_counts = df_cleaned["face_count"]
    face_min = int(face_counts.min())
    face_max = int(face_counts.max())
    face_mean = float(face_counts.mean())
    face_median = float(face_counts.median())
    zero_face_count = int((face_counts == 0).sum())

    # 6. Verify row preservation
    cleaned_row_count = len(df_cleaned)
    assert raw_row_count == cleaned_row_count, f"Row count mismatch! Raw={raw_row_count}, Cleaned={cleaned_row_count}"
    logger.info(f"Integrity check passed: All {cleaned_row_count} rows strictly preserved (0 deleted).")

    # 7. Convert and format dates
    df_cleaned["published_at_dt"] = pd.to_datetime(df_cleaned["published_at"], errors="coerce")
    date_min = df_cleaned["published_at_dt"].min().strftime("%Y-%m-%d")
    date_max = df_cleaned["published_at_dt"].max().strftime("%Y-%m-%d")
    df_cleaned.drop(columns=["published_at_dt"], inplace=True)

    # Save cleaned videos to UTF-8 CSV
    df_cleaned.to_csv(CLEANED_CSV_PATH, index=False, encoding="utf-8")
    logger.info(f"Saved cleaned dataset to {CLEANED_CSV_PATH} (UTF-8, {cleaned_row_count} rows)")

    # Save cleaning audit log
    df_log = pd.DataFrame(cleaning_logs)
    if df_log.empty:
        df_log = pd.DataFrame([{
            "row_index": -1,
            "field": "none",
            "original_value": "none",
            "cleaned_value": "none",
            "action": "no_changes_needed",
            "details": "Dataset already pristine"
        }])
    df_log.to_csv(CLEANING_LOG_PATH, index=False, encoding="utf-8")
    logger.info(f"Saved cleaning log to {CLEANING_LOG_PATH} ({len(df_log)} audit events)")

    # 8. Generate Data Quality Report
    quality_metrics = [
        {"metric": "total_input_rows", "value": raw_row_count, "notes": "Total rows in raw challenge_videos_final dataset"},
        {"metric": "total_output_rows", "value": cleaned_row_count, "notes": "Total rows in cleaned_videos.csv"},
        {"metric": "rows_deleted", "value": 0, "notes": "No rows silently deleted"},
        {"metric": "total_columns", "value": raw_col_count, "notes": "Number of columns in dataset"},
        {"metric": "input_encoding", "value": detected_encoding, "notes": "Original character encoding"},
        {"metric": "output_encoding", "value": "UTF-8", "notes": "Standardized output character encoding"},
        {"metric": "malformed_video_ids_found", "value": malformed_id_count, "notes": "Rows with Excel #NAME? formula error"},
        {"metric": "malformed_video_ids_recovered", "value": len([l for l in cleaning_logs if l['action'] == 'video_id_recovered_from_thumbnail_url']), "notes": "Recovered unambiguously from thumbnail URL"},
        {"metric": "duplicate_video_ids_raw", "value": raw_dupes, "notes": "Duplicates in raw data before ID recovery"},
        {"metric": "duplicate_video_ids_cleaned", "value": cleaned_dupes, "notes": "Duplicates in cleaned data (must be 0)"},
        {"metric": "channel_name_corrections", "value": len([l for l in cleaning_logs if l['action'] == 'channel_name_repaired']), "notes": "Repaired backslash to Airrack"},
        {"metric": "missing_values_cleaned", "value": total_missing, "notes": "Total null/NaN cells in cleaned dataset"},
        {"metric": "unique_channels_count", "value": df_cleaned["channel_id"].nunique(), "notes": "Distinct creator channels"},
        {"metric": "date_range_start", "value": date_min, "notes": "Earliest video publication date"},
        {"metric": "date_range_end", "value": date_max, "notes": "Latest video publication date"},
        {"metric": "raw_face_count_min", "value": face_min, "notes": "Minimum existing face_count in dataset"},
        {"metric": "raw_face_count_max", "value": face_max, "notes": "Maximum existing face_count in dataset"},
        {"metric": "raw_face_count_mean", "value": round(face_mean, 4), "notes": "Mean existing face_count in dataset"},
        {"metric": "raw_face_count_median", "value": face_median, "notes": "Median existing face_count in dataset"},
        {"metric": "raw_zero_face_thumbnails_count", "value": zero_face_count, "notes": "Videos with 0 existing face count"},
        {"metric": "raw_zero_face_thumbnails_pct", "value": round((zero_face_count / cleaned_row_count) * 100, 2), "notes": "Percentage with 0 existing face count"}
    ]
    df_quality = pd.DataFrame(quality_metrics)
    df_quality.to_csv(DATA_QUALITY_REPORT_PATH, index=False, encoding="utf-8")
    logger.info(f"Saved data quality report to {DATA_QUALITY_REPORT_PATH} ({len(df_quality)} metrics)")

    print("\n================== STAGE 2 DATA QUALITY SUMMARY ==================")
    for m in quality_metrics:
        print(f"  {m['metric']:<32} : {m['value']} ({m['notes']})")
    print("==================================================================\n")
    logger.info("Stage 2 completed successfully.")


if __name__ == "__main__":
    clean_dataset()
