#!/usr/bin/env python3
"""
03_extract_title_features.py
Stage 4: Title Feature Extraction for Click Appeal Score Research Project.

Features:
1. Capitalization Ratio:
   f_caps = (uppercase tokens containing >= 2 letters) / L
2. Trigger-word Ratio:
   f_trigger = (trigger word occurrences) / L
3. Sensitive-word Ratio:
   f_sensitive = (sensitive word occurrences) / L
4. Symbol and Emoji Ratio:
   f_symbol = (! count + ? count + emoji count) / L

Where L is the number of title whitespace tokens (minimum 1).
Documented English and Hindi lexicons are used (labeled researcher-defined baseline).
Saves intermediate counts and raw feature ratios to features/title_features.csv.
"""

import sys
import os
import re
import logging
from pathlib import Path
import pandas as pd
try:
    import emoji
except ImportError:
    emoji = None
    import regex as unicode_regex

# Configure logging
LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "03_extract_title_features.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("title_features")

BASE_DIR = Path(__file__).resolve().parent.parent
CLEANED_CSV_PATH = BASE_DIR / "data" / "processed" / "cleaned_videos.csv"
FEATURES_DIR = BASE_DIR / "features"
FEATURES_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_CSV_PATH = FEATURES_DIR / "title_features.csv"

# ==============================================================================
# RESEARCHER-DEFINED BASELINE LEXICONS (Documented & Non-Scientifically Validated)
# ==============================================================================

ENGLISH_TRIGGER_WORDS = {
    # Superlatives and extreme scale
    "best", "worst", "biggest", "largest", "highest", "hardest", "toughest",
    "fastest", "hottest", "insane", "extreme", "ultimate", "epic", "impossible",
    "unbeatable", "undefeated", "unbeaten", "brutal", "massive", "giant",
    "heaviest", "spiciest", "cheapest", "expensive", "poorest", "record",
    # Competition, outcomes, stakes
    "challenge", "contest", "competition", "tournament", "compete", "vs", "versus",
    "won", "win", "winner", "winning", "lose", "lost", "prize", "jackpot", "cash",
    "standing", "survive", "survived",
    # Urgency, discovery and curiosity hooks
    "viral", "secret", "secretly", "proof", "revealed", "exposed", "truth",
    "shocking", "never", "ever", "finally", "now", "hours", "days", "minutes",
    "free", "worth", "must", "stop", "warning"
}

ENGLISH_TRIGGER_PHRASES = [
    "last to", "first to", "worth it"
]

HINDI_TRIGGER_WORDS = {
    # Scale and impact
    "sabse", "bada", "badi", "maha", "asli", "nakli", "khatarnak",
    "zabardast", "gazab", "dhamaka", "bawal",
    # Monetary and stakes
    "crore", "cr", "lakh", "inam", "jeet", "har", "shart", "muqabla", "khatam"
}

ENGLISH_SENSITIVE_WORDS = {
    # Harm, death and violence
    "die", "died", "death", "dead", "kill", "killed", "murder", "murdered",
    "blood", "bloody", "weapon", "weapons", "fight", "gun", "shooting", "bomb",
    "destroy", "destroyed", "crash", "crashed", "attack", "assault",
    # Illegality, crime, controversy
    "illegal", "crime", "criminal", "scam", "scammer", "scammers", "scammed",
    "cheat", "cheater", "cheating", "jail", "prison", "police", "arrest",
    "arrested", "fired", "steal", "stolen", "pirates", "cult", "racist",
    "racists", "dictatorship", "dictator", "trapped", "roadkill",
    # Physical distress and failure
    "danger", "dangerous", "emergency", "hospital", "failed", "failure",
    "sick", "vomit", "puke", "pee", "poop", "suffering", "heartbreaking",
    "cried", "cry", "pain", "poison", "toxic"
}

ENGLISH_SENSITIVE_PHRASES = [
    "heart attack", "broke into", "rock bottom", "lost mail"
]

HINDI_SENSITIVE_WORDS = {
    "maut", "khoon", "mar", "zeher", "barbaad", "bura", "paap",
    "police", "chor", "dhokha", "danga", "giraftar", "jail"
}


def clean_token(token: str) -> str:
    """Strip leading/trailing punctuation from token."""
    return re.sub(r"^[^\w]+|[^\w]+$", "", token)


def count_emojis(text: str) -> int:
    """Count emoji grapheme clusters; use the emoji package when available.

    The regex fallback counts grapheme clusters containing an extended
    pictographic character or a regional-indicator flag pair.
    """
    if emoji is not None:
        return emoji.emoji_count(text)
    clusters = unicode_regex.findall(r"\X", text)
    return sum(bool(unicode_regex.search(r"\p{Extended_Pictographic}|\p{Regional_Indicator}", c)) for c in clusters)


def match_phrases_and_words(text: str, phrases, words):
    """Match phrases first, then words outside phrase spans (no overlap double-count)."""
    phrase_matches = []
    occupied = []
    # Longest phrases first; case-insensitive and bounded by non-word characters.
    for phrase in sorted(phrases, key=len, reverse=True):
        pattern = re.compile(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", re.IGNORECASE)
        for match in pattern.finditer(text):
            span = match.span()
            if any(span[0] < end and span[1] > start for start, end in occupied):
                continue
            occupied.append(span)
            phrase_matches.append(phrase)

    word_matches = []
    # Preserve the project's original whitespace-tokenization rule. Strip only
    # leading/trailing punctuation as before; do not split internal apostrophes
    # or ellipses into new lexicon hits.
    for match in re.finditer(r"\S+", text, flags=re.UNICODE):
        span = match.span()
        if any(span[0] < end and span[1] > start for start, end in occupied):
            continue
        token = clean_token(match.group(0)).lower()
        if token and token in words:
            word_matches.append(token)
    return phrase_matches + word_matches


def extract_title_features():
    logger.info("Starting Stage 4: Title Feature Extraction...")
    if not CLEANED_CSV_PATH.exists():
        logger.error(f"Cleaned CSV not found at {CLEANED_CSV_PATH}. Run Stage 2 first.")
        sys.exit(1)

    df = pd.read_csv(CLEANED_CSV_PATH)
    logger.info(f"Loaded {len(df)} titles from {CLEANED_CSV_PATH}")

    records = []

    for idx, row in df.iterrows():
        video_id = str(row["video_id"]).strip()
        channel_name = str(row["channel_name"]).strip()
        title = str(row["title"]).strip()

        # Whitespace tokenization
        raw_tokens = title.split()
        L = max(1, len(raw_tokens))

        # 1. Capitalization Feature (f_caps)
        # Count uppercase alphabetic tokens containing at least two letters
        uppercase_tokens = []
        for t in raw_tokens:
            cleaned = clean_token(t)
            letters = [c for c in cleaned if c.isalpha()]
            if len(letters) >= 2 and cleaned.isupper() and all(c.isupper() for c in letters):
                uppercase_tokens.append(cleaned)

        caps_count = len(uppercase_tokens)
        f_caps = caps_count / L

        # 2. Trigger-word Feature (f_trigger)
        trigger_lexicon = ENGLISH_TRIGGER_WORDS | HINDI_TRIGGER_WORDS
        matched_triggers = match_phrases_and_words(
            title, ENGLISH_TRIGGER_PHRASES, trigger_lexicon
        )
        trigger_count = len(matched_triggers)
        f_trigger = trigger_count / L

        # 3. Sensitive-word Feature (f_sensitive)
        sensitive_lexicon = ENGLISH_SENSITIVE_WORDS | HINDI_SENSITIVE_WORDS
        matched_sensitive = match_phrases_and_words(
            title, ENGLISH_SENSITIVE_PHRASES, sensitive_lexicon
        )
        sensitive_count = len(matched_sensitive)
        f_sensitive = sensitive_count / L

        # 4. Symbol and Emoji Feature (f_symbol)
        exclamation_count = title.count("!")
        question_count = title.count("?")
        emoji_count_val = count_emojis(title)
        symbol_total = exclamation_count + question_count + emoji_count_val
        f_symbol = symbol_total / L

        records.append({
            "video_id": video_id,
            "channel_name": channel_name,
            "title": title,
            "token_count_L": L,
            "uppercase_tokens_count": caps_count,
            "uppercase_tokens_list": ";".join(uppercase_tokens),
            "trigger_occurrences_count": trigger_count,
            "trigger_words_matched": ";".join(matched_triggers),
            "sensitive_occurrences_count": sensitive_count,
            "sensitive_words_matched": ";".join(matched_sensitive),
            "exclamation_count": exclamation_count,
            "question_count": question_count,
            "emoji_count": emoji_count_val,
            "symbol_total_count": symbol_total,
            "f_caps": round(f_caps, 6),
            "f_trigger": round(f_trigger, 6),
            "f_sensitive": round(f_sensitive, 6),
            "f_symbol": round(f_symbol, 6)
        })

    df_out = pd.DataFrame(records)
    df_out.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8")
    logger.info(f"Saved title features to {OUTPUT_CSV_PATH} ({len(df_out)} rows)")

    print("\n================== STAGE 4 TITLE FEATURES SUMMARY ==================")
    print(f"  Total titles processed : {len(df_out)}")
    print(f"  f_caps       -> Min: {df_out['f_caps'].min():.4f}, Mean: {df_out['f_caps'].mean():.4f}, Max: {df_out['f_caps'].max():.4f}, >0: {(df_out['f_caps'] > 0).sum()} ({(df_out['f_caps'] > 0).mean()*100:.1f}%)")
    print(f"  f_trigger    -> Min: {df_out['f_trigger'].min():.4f}, Mean: {df_out['f_trigger'].mean():.4f}, Max: {df_out['f_trigger'].max():.4f}, >0: {(df_out['f_trigger'] > 0).sum()} ({(df_out['f_trigger'] > 0).mean()*100:.1f}%)")
    print(f"  f_sensitive  -> Min: {df_out['f_sensitive'].min():.4f}, Mean: {df_out['f_sensitive'].mean():.4f}, Max: {df_out['f_sensitive'].max():.4f}, >0: {(df_out['f_sensitive'] > 0).sum()} ({(df_out['f_sensitive'] > 0).mean()*100:.1f}%)")
    print(f"  f_symbol     -> Min: {df_out['f_symbol'].min():.4f}, Mean: {df_out['f_symbol'].mean():.4f}, Max: {df_out['f_symbol'].max():.4f}, >0: {(df_out['f_symbol'] > 0).sum()} ({(df_out['f_symbol'] > 0).mean()*100:.1f}%)")
    print("====================================================================\n")
    logger.info("Stage 4 completed successfully.")


if __name__ == "__main__":
    extract_title_features()
