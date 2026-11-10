#!/usr/bin/env python3
"""
02_download_thumbnails.py
Stage 3: Thumbnail Retrieval and Verification Pipeline for Click Appeal Score Research Project.

Tasks:
- Use thumbnail URLs from data/processed/cleaned_videos.csv.
- Download all accessible original thumbnails into thumbnails_original/ using video_id as filename.
- Record:
    1. Download success/failure.
    2. HTTP status code.
    3. Image dimensions (width, height).
    4. Image format (e.g., JPEG, PNG).
    5. Failure reason (if any).
    6. Local file path and file size.
- Strictly never fabricate images or report failed downloads as successful.
- Save download manifest to thumbnails_original/download_manifest.csv.
- Resume capability: skips re-downloading if valid file already exists.
"""

import sys
import os
import time
import logging
from pathlib import Path
from io import BytesIO
import requests
import pandas as pd
from PIL import Image

# Configure logging
LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "02_download_thumbnails.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("download_thumbnails")

BASE_DIR = Path(__file__).resolve().parent.parent
CLEANED_CSV_PATH = BASE_DIR / "data" / "processed" / "cleaned_videos.csv"
THUMBNAILS_DIR = BASE_DIR / "thumbnails_original"
THUMBNAILS_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_PATH = THUMBNAILS_DIR / "download_manifest.csv"
PROCESSED_MANIFEST_PATH = BASE_DIR / "data" / "processed" / "download_manifest.csv"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}


def download_thumbnails():
    logger.info("Starting Stage 3: Thumbnail Retrieval and Verification...")
    if not CLEANED_CSV_PATH.exists():
        logger.error(f"Cleaned CSV not found at {CLEANED_CSV_PATH}. Run scripts/01_clean_data.py first.")
        sys.exit(1)

    df = pd.read_csv(CLEANED_CSV_PATH)
    logger.info(f"Loaded {len(df)} records from {CLEANED_CSV_PATH}")

    manifest_records = []
    success_count = 0
    failure_count = 0
    cached_count = 0

    session = requests.Session()
    session.headers.update(HEADERS)

    for idx, row in df.iterrows():
        video_id = str(row["video_id"]).strip()
        url = str(row["thumbnail"]).strip()
        channel_name = str(row["channel_name"]).strip()

        # Target file path (defaulting to .jpg as standard for YouTube thumbnails)
        ext = ".jpg"
        target_path = THUMBNAILS_DIR / f"{video_id}{ext}"

        # Check if already downloaded and valid
        if target_path.exists() and target_path.stat().st_size > 0:
            try:
                with Image.open(target_path) as img:
                    width, height = img.size
                    img_format = img.format
                manifest_records.append({
                    "video_id": video_id,
                    "channel_name": channel_name,
                    "thumbnail_url": url,
                    "status": "SUCCESS",
                    "http_status": 200,
                    "width": width,
                    "height": height,
                    "format": img_format,
                    "file_size_bytes": target_path.stat().st_size,
                    "file_path": str(target_path.relative_to(BASE_DIR)).replace("\\", "/"),
                    "failure_reason": "None (cached/existing)"
                })
                success_count += 1
                cached_count += 1
                logger.info(f"[{idx+1}/{len(df)}] Cached: {video_id} ({width}x{height}, {img_format})")
                continue
            except Exception as e:
                logger.warning(f"Existing file for {video_id} is corrupt ({e}). Re-downloading...")
                try:
                    target_path.unlink()
                except OSError:
                    pass

        # Attempt download
        try:
            resp = session.get(url, timeout=15)
            http_status = resp.status_code

            if http_status == 200:
                img_bytes = resp.content
                # Validate image integrity with PIL
                try:
                    with Image.open(BytesIO(img_bytes)) as img:
                        img.verify()
                    # Re-open for dimensions and format as verify() closes the file
                    with Image.open(BytesIO(img_bytes)) as img:
                        width, height = img.size
                        img_format = img.format or "JPEG"
                        actual_ext = ".jpg" if img_format == "JPEG" else f".{img_format.lower()}"
                        save_path = THUMBNAILS_DIR / f"{video_id}{actual_ext}"
                        with open(save_path, "wb") as f_out:
                            f_out.write(img_bytes)

                    file_size = len(img_bytes)
                    manifest_records.append({
                        "video_id": video_id,
                        "channel_name": channel_name,
                        "thumbnail_url": url,
                        "status": "SUCCESS",
                        "http_status": 200,
                        "width": width,
                        "height": height,
                        "format": img_format,
                        "file_size_bytes": file_size,
                        "file_path": str(save_path.relative_to(BASE_DIR)).replace("\\", "/"),
                        "failure_reason": "None"
                    })
                    success_count += 1
                    logger.info(f"[{idx+1}/{len(df)}] Downloaded: {video_id} ({width}x{height}, {img_format}, {file_size} B)")
                except Exception as img_err:
                    failure_count += 1
                    err_msg = f"Image validation failed: {str(img_err)}"
                    manifest_records.append({
                        "video_id": video_id,
                        "channel_name": channel_name,
                        "thumbnail_url": url,
                        "status": "FAILED",
                        "http_status": 200,
                        "width": None,
                        "height": None,
                        "format": None,
                        "file_size_bytes": len(img_bytes),
                        "file_path": None,
                        "failure_reason": err_msg
                    })
                    logger.error(f"[{idx+1}/{len(df)}] Corrupt image for {video_id}: {err_msg}")
            else:
                failure_count += 1
                err_msg = f"HTTP Error {http_status}"
                manifest_records.append({
                    "video_id": video_id,
                    "channel_name": channel_name,
                    "thumbnail_url": url,
                    "status": "FAILED",
                    "http_status": http_status,
                    "width": None,
                    "height": None,
                    "format": None,
                    "file_size_bytes": 0,
                    "file_path": None,
                    "failure_reason": err_msg
                })
                logger.warning(f"[{idx+1}/{len(df)}] Failed HTTP {http_status} for {video_id} ({url})")

        except Exception as req_err:
            failure_count += 1
            err_msg = f"Network/Connection error: {str(req_err)}"
            manifest_records.append({
                "video_id": video_id,
                "channel_name": channel_name,
                "thumbnail_url": url,
                "status": "FAILED",
                "http_status": None,
                "width": None,
                "height": None,
                "format": None,
                "file_size_bytes": 0,
                "file_path": None,
                "failure_reason": err_msg
            })
            logger.error(f"[{idx+1}/{len(df)}] Exception for {video_id}: {err_msg}")

        # Slight rate limiting to be polite
        time.sleep(0.05)

    df_manifest = pd.DataFrame(manifest_records)
    df_manifest.to_csv(MANIFEST_PATH, index=False, encoding="utf-8")
    df_manifest.to_csv(PROCESSED_MANIFEST_PATH, index=False, encoding="utf-8")
    logger.info(f"Saved download manifest to {MANIFEST_PATH} and {PROCESSED_MANIFEST_PATH}")

    print("\n================== STAGE 3 DOWNLOAD SUMMARY ==================")
    print(f"  Total video records processed : {len(df)}")
    print(f"  Successfully downloaded/verified: {success_count} ({success_count/len(df)*100:.1f}%)")
    print(f"    - Newly downloaded          : {success_count - cached_count}")
    print(f"    - Cached from previous run  : {cached_count}")
    print(f"  Failed downloads              : {failure_count} ({failure_count/len(df)*100:.1f}%)")
    if failure_count > 0:
        failures = df_manifest[df_manifest["status"] == "FAILED"]
        print("\n  Failed items details:")
        for _, f_row in failures.iterrows():
            print(f"    - {f_row['video_id']} ({f_row['channel_name']}): HTTP {f_row['http_status']} | {f_row['failure_reason']}")
    print("==============================================================\n")
    logger.info("Stage 3 completed successfully.")


if __name__ == "__main__":
    download_thumbnails()
