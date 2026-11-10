#!/usr/bin/env python3
"""
04_extract_thumbnail_features.py
Stage 5: Thumbnail Feature Extraction & Visual Audit for Click Appeal Score Research Project.

Tasks:
A. Face Detection:
   - MediaPipe 0.10.21 legacy API: mp.solutions.face_detection
   - Parameters: model_selection=1, min_detection_confidence=0.5
   - f_face = 1 if at least one face detected else 0
   - Record face count, bounding boxes, detection confidences, areas
   - Compare with CSV existing face_count column

B. Expression Detection:
   - FER 25.10.3 + TensorFlow 2.16.1
   - Reuse MediaPipe face bounding boxes via face_rectangles argument
   - Select largest face by Area = w * h
   - f_expr = max(P(surprise), P(fear))
   - If no face detected, assign 0.0
   - If face detected but classification fails, record missing data and processing error (not zero)
   - Note: Model confidence score, not direct measurement of psychological emotion intensity.

C. Contrast:
   - RGB to Luminance: Y = 0.299R + 0.587G + 0.114B in [0, 1]
   - f_contrast = sigma(Y) (population standard deviation)

D. Saturation:
   - HSV Saturation S in [0, 1]
   - f_saturation = sigma(S) (population standard deviation)

E. Visual Audit:
   - Annotated thumbnails with bounding boxes, face IDs, confidence, largest face marker, fear/surprise, f_expr
   - 24-image representative contact sheet (zero, one, multiple faces)

ML models are loaded ONCE and reused across all images. Intermediate results saved with resume capability.
"""

import sys
import os
import json
import logging
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

# Suppress excessive TensorFlow logs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import mediapipe as mp
from fer.fer import FER

# Configure logging
LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "04_extract_thumbnail_features.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("thumbnail_features")

BASE_DIR = Path(__file__).resolve().parent.parent
CLEANED_CSV_PATH = BASE_DIR / "data" / "processed" / "cleaned_videos.csv"
THUMBNAILS_DIR = BASE_DIR / "thumbnails_original"
ANNOTATED_DIR = BASE_DIR / "annotated_thumbnails"
ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)
ANALYSIS_DIR = BASE_DIR / "analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
FEATURES_DIR = BASE_DIR / "features"
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FEATURES_CSV = FEATURES_DIR / "thumbnail_features.csv"
CONTACT_SHEET_ANNOTATED = ANNOTATED_DIR / "contact_sheet_24_representative.jpg"
CONTACT_SHEET_ANALYSIS = ANALYSIS_DIR / "contact_sheet.jpg"


def annotate_image(img_bgr, faces_data, largest_face_idx, f_expr, p_fear, p_surprise, video_id, channel_name):
    """Draw visual audit annotations on thumbnail."""
    annotated = img_bgr.copy()
    H, W, _ = annotated.shape

    # Draw header bar
    cv2.rectangle(annotated, (0, 0), (W, 40), (20, 20, 20), -1)
    header_text = f"{channel_name} | {video_id} | Detected Faces: {len(faces_data)} | f_expr: {f_expr if not np.isnan(f_expr) else 'ERR':.3f} (Fear:{p_fear:.2f}, Surpr:{p_surprise:.2f})"
    cv2.putText(annotated, header_text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    if not faces_data:
        # No face banner
        cv2.rectangle(annotated, (W // 4, H // 2 - 30), (3 * W // 4, H // 2 + 30), (0, 0, 180), -1)
        cv2.putText(annotated, "NO FACE DETECTED (f_face=0, f_expr=0.0)", (W // 4 + 20, H // 2 + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        return annotated

    for idx, face in enumerate(faces_data):
        x, y, w, h = face["x"], face["y"], face["w"], face["h"]
        conf = face["confidence"]
        is_largest = (idx == largest_face_idx)

        if is_largest:
            # Highlight largest face in gold/yellow with thicker border
            color = (0, 215, 255)  # Gold in BGR
            thickness = 4
            label = f"LARGEST FACE #{idx+1} [Conf: {conf:.2f}] Area: {w*h}px"
            expr_label = f"f_expr: {f_expr:.2f} (Fear: {p_fear:.2f}, Surpr: {p_surprise:.2f})"
        else:
            # Secondary faces in bright green
            color = (0, 255, 0)
            thickness = 2
            label = f"Face #{idx+1} [Conf: {conf:.2f}]"
            expr_label = None

        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, thickness)

        # Label background
        label_y = max(20, y - 10)
        cv2.rectangle(annotated, (x, label_y - 18), (x + len(label) * 11, label_y + 4), color, -1)
        cv2.putText(annotated, label, (x + 2, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

        if expr_label:
            cv2.rectangle(annotated, (x, y + h + 4), (x + len(expr_label) * 11, y + h + 24), (0, 0, 0), -1)
            cv2.putText(annotated, expr_label, (x + 2, y + h + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 215, 255), 1, cv2.LINE_AA)

    return annotated


def create_contact_sheet(df_results, representative_ids, output_paths):
    """Generate a clean 24-image contact sheet of representative thumbnails."""
    logger.info(f"Generating contact sheet with {len(representative_ids)} representative thumbnails...")
    cols = 4
    rows = (len(representative_ids) + cols - 1) // cols
    tile_w = 400
    tile_h = 225

    sheet_w = cols * tile_w
    sheet_h = rows * tile_h + 80  # Top banner space
    contact_sheet = Image.new("RGB", (sheet_w, sheet_h), color=(30, 30, 30))
    draw = ImageDraw.Draw(contact_sheet)

    # Top banner title
    banner_text = "Click Appeal Score — Representative Visual Audit Contact Sheet (Zero, Single & Multiple Faces)"
    draw.text((sheet_w // 2 - 380, 25), banner_text, fill=(255, 255, 255))

    for idx, vid in enumerate(representative_ids):
        ann_path = ANNOTATED_DIR / f"{vid}_annotated.jpg"
        if not ann_path.exists():
            ann_path = THUMBNAILS_DIR / f"{vid}.jpg"

        if ann_path.exists():
            try:
                with Image.open(ann_path) as tile_img:
                    tile_resized = tile_img.resize((tile_w, tile_h), Image.Resampling.LANCZOS)
                    r = idx // cols
                    c = idx % cols
                    pos_x = c * tile_w
                    pos_y = 80 + r * tile_h
                    contact_sheet.paste(tile_resized, (pos_x, pos_y))
                    # Border
                    draw.rectangle([pos_x, pos_y, pos_x + tile_w - 1, pos_y + tile_h - 1], outline=(60, 60, 60), width=1)
            except Exception as e:
                logger.warning(f"Could not load image for contact sheet: {vid} ({e})")

    for p in output_paths:
        contact_sheet.save(p, "JPEG", quality=90)
        logger.info(f"Saved contact sheet to {p}")


def extract_thumbnail_features():
    logger.info("Starting Stage 5: Thumbnail Feature Extraction & Visual Audit...")

    if not CLEANED_CSV_PATH.exists():
        logger.error(f"Cleaned CSV not found at {CLEANED_CSV_PATH}")
        sys.exit(1)

    df_cleaned = pd.read_csv(CLEANED_CSV_PATH)
    logger.info(f"Loaded {len(df_cleaned)} videos for visual processing.")

    # 1. Initialize ML Models ONCE (per instructions)
    logger.info("Initializing MediaPipe FaceDetection (model_selection=1, min_detection_confidence=0.5)...")
    mp_face_detection = mp.solutions.face_detection
    face_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

    logger.info("Initializing FER Emotion Detector (TensorFlow 2.16.1 backend)...")
    fer_detector = FER()

    # Load existing features if resuming
    existing_results = {}
    if OUTPUT_FEATURES_CSV.exists():
        try:
            df_existing = pd.read_csv(OUTPUT_FEATURES_CSV)
            for _, r in df_existing.iterrows():
                existing_results[str(r["video_id"]).strip()] = r.to_dict()
            logger.info(f"Found {len(existing_results)} already processed thumbnails. Will resume.")
        except Exception as e:
            logger.warning(f"Could not read existing feature CSV: {e}")

    results = []
    total_videos = len(df_cleaned)

    for idx, row in df_cleaned.iterrows():
        video_id = str(row["video_id"]).strip()
        channel_name = str(row["channel_name"]).strip()
        csv_face_count = int(row["face_count"])

        # Check resume cache
        ann_path = ANNOTATED_DIR / f"{video_id}_annotated.jpg"
        if video_id in existing_results and ann_path.exists():
            results.append(existing_results[video_id])
            if (idx + 1) % 20 == 0 or idx + 1 == total_videos:
                logger.info(f"[{idx+1}/{total_videos}] Resumed/Cached: {video_id}")
            continue

        img_path = THUMBNAILS_DIR / f"{video_id}.jpg"
        if not img_path.exists():
            logger.error(f"Thumbnail image missing at {img_path}!")
            # Record failed record
            results.append({
                "video_id": video_id,
                "channel_name": channel_name,
                "thumbnail_path": None,
                "image_width": None,
                "image_height": None,
                "csv_face_count": csv_face_count,
                "detected_face_count": None,
                "face_count_disagreement": None,
                "f_face": None,
                "faces_detected_json": json.dumps([]),
                "largest_face_id": None,
                "largest_face_box": None,
                "largest_face_area": None,
                "p_angry": None,
                "p_disgust": None,
                "p_fear": None,
                "p_happy": None,
                "p_sad": None,
                "p_surprise": None,
                "p_neutral": None,
                "f_expr": np.nan,
                "expression_status": "MISSING_IMAGE",
                "f_contrast": np.nan,
                "f_saturation": np.nan,
                "annotated_image_path": None
            })
            continue

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            logger.error(f"cv2.imread failed on {img_path}")
            continue

        H, W, _ = img_bgr.shape
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # ------------------------------------------------------------------
        # Feature C: Contrast (Luminance Population Standard Deviation)
        # Y = 0.299R + 0.587G + 0.114B in [0, 1]
        # ------------------------------------------------------------------
        R_norm = img_rgb[:, :, 0].astype(np.float64) / 255.0
        G_norm = img_rgb[:, :, 1].astype(np.float64) / 255.0
        B_norm = img_rgb[:, :, 2].astype(np.float64) / 255.0
        Y = 0.299 * R_norm + 0.587 * G_norm + 0.114 * B_norm
        f_contrast = float(np.std(Y, ddof=0))

        # ------------------------------------------------------------------
        # Feature D: Saturation (HSV Saturation Population Standard Deviation)
        # S in [0, 1]
        # ------------------------------------------------------------------
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        S_norm = img_hsv[:, :, 1].astype(np.float64) / 255.0
        f_saturation = float(np.std(S_norm, ddof=0))

        # ------------------------------------------------------------------
        # Feature A: Face Detection (MediaPipe 0.10.21)
        # ------------------------------------------------------------------
        mp_results = face_detector.process(img_rgb)
        detections = mp_results.detections or []
        detected_face_count = len(detections)
        f_face = 1 if detected_face_count >= 1 else 0

        faces_data = []
        for f_idx, d in enumerate(detections):
            score = float(d.score[0])
            bb = d.location_data.relative_bounding_box
            x = max(0, int(bb.xmin * W))
            y = max(0, int(bb.ymin * H))
            w = min(W - x, max(1, int(bb.width * W)))
            h = min(H - y, max(1, int(bb.height * H)))
            area = w * h
            faces_data.append({
                "face_id": f_idx + 1,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "confidence": round(score, 4),
                "area": area,
                "is_largest": False
            })

        # ------------------------------------------------------------------
        # Feature B: Expression Detection (FER 25.10.3)
        # ------------------------------------------------------------------
        p_angry = 0.0
        p_disgust = 0.0
        p_fear = 0.0
        p_happy = 0.0
        p_sad = 0.0
        p_surprise = 0.0
        p_neutral = 0.0
        f_expr = 0.0
        largest_face_idx = None
        largest_face_id = None
        largest_face_box = None
        largest_face_area = None
        expr_status = "NO_FACE"

        if detected_face_count == 0:
            # Defined rule: If no face detected, assign zero
            f_expr = 0.0
            expr_status = "NO_FACE"
        else:
            # Select largest valid face by A_i = w_i * h_i
            largest_face_idx = max(range(len(faces_data)), key=lambda i: faces_data[i]["area"])
            faces_data[largest_face_idx]["is_largest"] = True
            largest_face = faces_data[largest_face_idx]
            largest_face_id = largest_face["face_id"]
            largest_face_box = f"({largest_face['x']},{largest_face['y']},{largest_face['w']},{largest_face['h']})"
            largest_face_area = largest_face["area"]

            # Format rectangle for FER: (x, y, w, h)
            rect = (largest_face["x"], largest_face["y"], largest_face["w"], largest_face["h"])

            try:
                # Classify emotion on largest face rectangle
                fer_output = fer_detector.detect_emotions(img_bgr, face_rectangles=[rect])
                if fer_output and len(fer_output) > 0 and "emotions" in fer_output[0]:
                    emotions = fer_output[0]["emotions"]
                    p_angry = float(emotions.get("angry", 0.0))
                    p_disgust = float(emotions.get("disgust", 0.0))
                    p_fear = float(emotions.get("fear", 0.0))
                    p_happy = float(emotions.get("happy", 0.0))
                    p_sad = float(emotions.get("sad", 0.0))
                    p_surprise = float(emotions.get("surprise", 0.0))
                    p_neutral = float(emotions.get("neutral", 0.0))

                    # Formula: f_expr = max(P(surprise), P(fear))
                    f_expr = float(max(p_surprise, p_fear))
                    expr_status = "SUCCESS"
                else:
                    # Face detected but classification failed: record missing data and processing error, NOT zero
                    f_expr = np.nan
                    expr_status = "ERROR_CLASSIFICATION_FAILED"
                    logger.warning(f"Video {video_id}: Face detected but FER returned empty emotions.")
            except Exception as e:
                f_expr = np.nan
                expr_status = f"ERROR_{str(e)[:50]}"
                logger.error(f"Video {video_id}: FER classification exception: {e}")

        # Visual audit annotation
        annotated_img = annotate_image(
            img_bgr, faces_data, largest_face_idx, f_expr, p_fear, p_surprise, video_id, channel_name
        )
        ann_save_path = ANNOTATED_DIR / f"{video_id}_annotated.jpg"
        cv2.imwrite(str(ann_save_path), annotated_img)

        # Record result
        record = {
            "video_id": video_id,
            "channel_name": channel_name,
            "thumbnail_path": str(img_path.relative_to(BASE_DIR)).replace("\\", "/"),
            "image_width": W,
            "image_height": H,
            "csv_face_count": csv_face_count,
            "detected_face_count": detected_face_count,
            "face_count_disagreement": detected_face_count - csv_face_count,
            "f_face": f_face,
            "faces_detected_json": json.dumps(faces_data),
            "largest_face_id": largest_face_id,
            "largest_face_box": largest_face_box,
            "largest_face_area": largest_face_area,
            "p_angry": round(p_angry, 4),
            "p_disgust": round(p_disgust, 4),
            "p_fear": round(p_fear, 4),
            "p_happy": round(p_happy, 4),
            "p_sad": round(p_sad, 4),
            "p_surprise": round(p_surprise, 4),
            "p_neutral": round(p_neutral, 4),
            "f_expr": round(f_expr, 6) if not np.isnan(f_expr) else np.nan,
            "expression_status": expr_status,
            "f_contrast": round(f_contrast, 6),
            "f_saturation": round(f_saturation, 6),
            "annotated_image_path": str(ann_save_path.relative_to(BASE_DIR)).replace("\\", "/"),
        }
        results.append(record)

        if (idx + 1) % 20 == 0 or idx + 1 == total_videos:
            logger.info(f"[{idx+1}/{total_videos}] Processed {video_id} | Faces: {detected_face_count} (CSV: {csv_face_count}) | f_expr: {f_expr if not np.isnan(f_expr) else 'NaN'} | Status: {expr_status}")

    # Clean up detector
    face_detector.close()

    # Save features table
    df_features = pd.DataFrame(results)
    df_features.to_csv(OUTPUT_FEATURES_CSV, index=False, encoding="utf-8")
    logger.info(f"Saved complete thumbnail features to {OUTPUT_FEATURES_CSV} ({len(df_features)} rows)")

    # ------------------------------------------------------------------
    # Generate Contact Sheet (>= 20 Representative Examples)
    # Selected across: zero faces, single face, multiple faces
    # ------------------------------------------------------------------
    zero_face_vids = df_features[df_features["detected_face_count"] == 0]["video_id"].tolist()
    one_face_vids = df_features[df_features["detected_face_count"] == 1]["video_id"].tolist()
    multi_face_vids = df_features[df_features["detected_face_count"] >= 2]["video_id"].tolist()

    selected_zero = zero_face_vids[:8]
    selected_one = one_face_vids[:8]
    selected_multi = multi_face_vids[:8]
    representative_vids = selected_zero + selected_one + selected_multi

    create_contact_sheet(df_features, representative_vids, [CONTACT_SHEET_ANNOTATED, CONTACT_SHEET_ANALYSIS])

    # Print Summary
    valid_expr = df_features["f_expr"].dropna()
    print("\n================== STAGE 5 THUMBNAIL FEATURES SUMMARY ==================")
    print(f"  Total thumbnails analyzed    : {len(df_features)}")
    print(f"  Faces detected               : Total {df_features['detected_face_count'].sum()} faces across {len(df_features)} images")
    print(f"    - Zero-face thumbnails     : {(df_features['detected_face_count'] == 0).sum()} ({(df_features['detected_face_count'] == 0).mean()*100:.1f}%)")
    print(f"    - Single-face thumbnails   : {(df_features['detected_face_count'] == 1).sum()} ({(df_features['detected_face_count'] == 1).mean()*100:.1f}%)")
    print(f"    - Multi-face thumbnails    : {(df_features['detected_face_count'] >= 2).sum()} ({(df_features['detected_face_count'] >= 2).mean()*100:.1f}%)")
    print(f"  Expression detection status:")
    for status, cnt in df_features["expression_status"].value_counts().items():
        print(f"    - {status:<24} : {cnt}")
    print(f"  f_expr (valid)               -> Min: {valid_expr.min():.4f}, Mean: {valid_expr.mean():.4f}, Max: {valid_expr.max():.4f}")
    print(f"  f_contrast                   -> Min: {df_features['f_contrast'].min():.4f}, Mean: {df_features['f_contrast'].mean():.4f}, Max: {df_features['f_contrast'].max():.4f}")
    print(f"  f_saturation                 -> Min: {df_features['f_saturation'].min():.4f}, Mean: {df_features['f_saturation'].mean():.4f}, Max: {df_features['f_saturation'].max():.4f}")
    print(f"  Face-count agreement:")
    exact_match = (df_features["face_count_disagreement"] == 0).sum()
    print(f"    - Exact match with CSV     : {exact_match} ({exact_match / len(df_features) * 100:.1f}%)")
    print(f"    - Mean difference (Det-CSV): {df_features['face_count_disagreement'].mean():.3f}")
    print("========================================================================\n")
    logger.info("Stage 5 completed successfully.")


if __name__ == "__main__":
    extract_thumbnail_features()
