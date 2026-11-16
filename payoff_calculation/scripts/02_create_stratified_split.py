#!/usr/bin/env python3
"""
payoff_estimation/scripts/02_create_stratified_split.py
Phase 2 - Step 2: Channel-Level Stratified Partitioning & Peer Pool Isolation.

Requirements:
1. Partition 38 unique channels into:
   - Training: 26 channels (Tier 1: 8, Tier 2: 11, Tier 3: 7)
   - Validation: 6 channels (Tier 1: 2, Tier 2: 3, Tier 3: 1)
   - Test: 6 channels (Tier 1: 2, Tier 2: 2, Tier 3: 2)
   Total: 38 channels (26 / 6 / 6).
2. Reproducible random seed: seed = 42.
3. Channel-level integrity: Each channel belongs to exactly one partition.
4. Peer-pool isolation definition:
   - Training observations: competitor pool = other 25 training channels (Train \\ {i}).
   - Validation observations: competitor pool = 26 training channels.
   - Test observations: competitor pool = 32 pooled channels (Train + Val).
5. Exports:
   - payoff_estimation/data/channel_splits.json
   - payoff_estimation/data/partitioned_payoff_data.csv
   - payoff_estimation/logs/02_create_stratified_split.log
"""

import sys
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd

# Setup directories
SCRIPT_DIR = Path(__file__).resolve().parent
PAYOFF_DIR = SCRIPT_DIR.parent
DATA_DIR = PAYOFF_DIR / "data"
LOGS_DIR = PAYOFF_DIR / "logs"

DATA_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Configure logging
LOG_FILE = LOGS_DIR / "02_create_stratified_split.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("02_create_stratified_split")

RANDOM_SEED = 42


def run_stratified_split():
    logger.info("=" * 70)
    logger.info("STEP 2: CHANNEL-LEVEL STRATIFIED PARTITIONING & PEER ISOLATION")
    logger.info("=" * 70)

    input_csv = DATA_DIR / "processed_payoff_data.csv"
    if not input_csv.exists():
        logger.error(f"Cannot find input dataset: {input_csv}")
        sys.exit(1)

    df = pd.read_csv(input_csv)
    logger.info(f"Loaded processed dataset: {len(df)} rows across {df['channel_id'].nunique()} unique channels.")

    # 1. Channel-level Metadata Aggregation
    channel_df = df.groupby(["channel_id", "channel_name", "subscriber_tier"]).agg(
        video_count=("video_id", "count"),
        subscribers_channel=("subscribers_channel", "first"),
        mean_views=("views", "mean"),
        mean_appeal=("x", "mean")
    ).reset_index()

    n_channels = len(channel_df)
    assert n_channels == 38, f"Expected 38 channels, found {n_channels}"
    logger.info(f"Aggregated channel-level metadata for {n_channels} channels.")

    # 2. Stratified Partitioning Target Specifications
    # Target allocations by tier:
    # Tier 1 (<5M, 12 channels): 8 Train, 2 Val, 2 Test
    # Tier 2 (5M-20M, 16 channels): 11 Train, 3 Val, 2 Test
    # Tier 3 (>20M, 10 channels): 7 Train, 1 Val, 2 Test
    tier_specs = {
        "Tier_1_Low (<5M)": {"train": 8, "validation": 2, "test": 2},
        "Tier_2_Mid (5M-20M)": {"train": 11, "validation": 3, "test": 2},
        "Tier_3_Mega (>20M)": {"train": 7, "validation": 1, "test": 2}
    }

    rng = np.random.RandomState(RANDOM_SEED)
    channel_df["partition"] = ""

    for tier, specs in tier_specs.items():
        sub_df = channel_df[channel_df["subscriber_tier"] == tier].sort_values("channel_id")
        tier_channel_ids = list(sub_df["channel_id"].values)
        
        # Deterministic shuffle with seed
        rng.shuffle(tier_channel_ids)
        
        n_train = specs["train"]
        n_val = specs["validation"]
        n_test = specs["test"]
        
        assert len(tier_channel_ids) == n_train + n_val + n_test, \
            f"Tier {tier} mismatch: {len(tier_channel_ids)} vs {n_train + n_val + n_test}"
        
        train_ids = tier_channel_ids[:n_train]
        val_ids = tier_channel_ids[n_train:n_train + n_val]
        test_ids = tier_channel_ids[n_train + n_val:]
        
        channel_df.loc[channel_df["channel_id"].isin(train_ids), "partition"] = "train"
        channel_df.loc[channel_df["channel_id"].isin(val_ids), "partition"] = "validation"
        channel_df.loc[channel_df["channel_id"].isin(test_ids), "partition"] = "test"
        
        logger.info(f"{tier}: Assigned {len(train_ids)} Train, {len(val_ids)} Val, {len(test_ids)} Test.")

    # 3. Validation of Partition Sizes
    partition_counts = channel_df["partition"].value_counts().to_dict()
    logger.info(f"Channel counts per partition: {partition_counts}")
    assert partition_counts.get("train") == 26, f"Expected 26 train channels, got {partition_counts.get('train')}"
    assert partition_counts.get("validation") == 6, f"Expected 6 val channels, got {partition_counts.get('validation')}"
    assert partition_counts.get("test") == 6, f"Expected 6 test channels, got {partition_counts.get('test')}"

    # 4. Map Partition Assignment back to Video Level
    channel_partition_map = channel_df.set_index("channel_id")["partition"].to_dict()
    df["partition"] = df["channel_id"].map(channel_partition_map)

    # Integrity verification
    assert df["partition"].isna().sum() == 0, "Some videos unassigned!"
    video_partition_counts = df["partition"].value_counts().to_dict()
    logger.info(f"Video counts per partition: {video_partition_counts}")
    assert len(df) == 809, f"Expected 809 total videos, got {len(df)}"

    # 5. Build Comprehensive Channel Split Dictionary
    train_channels = channel_df[channel_df["partition"] == "train"].sort_values("channel_name")
    val_channels = channel_df[channel_df["partition"] == "validation"].sort_values("channel_name")
    test_channels = channel_df[channel_df["partition"] == "test"].sort_values("channel_name")

    peer_pool_rules = {
        "training_observations": {
            "eligible_pool_size": 26,
            "description": "Competitor peer averages for training videos are constructed strictly among the other 25 training channels (Train \\ {i}). Validation and test channels are completely excluded.",
            "eligible_channel_ids": train_channels["channel_id"].tolist()
        },
        "validation_observations": {
            "eligible_pool_size": 26,
            "description": "Competitor peer averages for validation videos are constructed strictly using the 26 training channels as the competitor pool. Validation channels do not compete against each other or test channels.",
            "eligible_channel_ids": train_channels["channel_id"].tolist()
        },
        "test_observations": {
            "eligible_pool_size": 32,
            "description": "Competitor peer averages for held-out test videos are constructed strictly using the pooled 32 training + validation channels (Train ∪ Val). Test channels never compete against each other or enter training pools.",
            "eligible_channel_ids": train_channels["channel_id"].tolist() + val_channels["channel_id"].tolist()
        },
        "post_test_calibration": {
            "eligible_pool_size": 38,
            "description": "For post-test descriptive structural calibration on the entire dataset, competitor peer averages are constructed among all 37 other channels (All \\ {i}).",
            "eligible_channel_ids": channel_df["channel_id"].tolist()
        }
    }

    split_payload = {
        "metadata": {
            "random_seed": RANDOM_SEED,
            "split_methodology": "Subscriber-tier stratified channel-level random sampling",
            "total_channels": 38,
            "total_videos": 809,
            "partition_channel_sizes": {"train": 26, "validation": 6, "test": 6},
            "partition_video_sizes": video_partition_counts,
            "tier_distribution": {
                tier: {
                    "total": int(channel_df[channel_df["subscriber_tier"] == tier].shape[0]),
                    "train": int(channel_df[(channel_df["subscriber_tier"] == tier) & (channel_df["partition"] == "train")].shape[0]),
                    "validation": int(channel_df[(channel_df["subscriber_tier"] == tier) & (channel_df["partition"] == "validation")].shape[0]),
                    "test": int(channel_df[(channel_df["subscriber_tier"] == tier) & (channel_df["partition"] == "test")].shape[0]),
                    "total_videos": int(df[df["subscriber_tier"] == tier].shape[0]),
                    "train_videos": int(df[(df["subscriber_tier"] == tier) & (df["partition"] == "train")].shape[0]),
                    "val_videos": int(df[(df["subscriber_tier"] == tier) & (df["partition"] == "validation")].shape[0]),
                    "test_videos": int(df[(df["subscriber_tier"] == tier) & (df["partition"] == "test")].shape[0])
                }
                for tier in tier_specs.keys()
            }
        },
        "peer_pool_isolation_rules": peer_pool_rules,
        "channels": [
            {
                "channel_id": row["channel_id"],
                "channel_name": row["channel_name"],
                "subscriber_tier": row["subscriber_tier"],
                "subscribers": int(row["subscribers_channel"]),
                "video_count": int(row["video_count"]),
                "partition": row["partition"]
            }
            for _, row in channel_df.sort_values(["partition", "subscriber_tier", "channel_name"]).iterrows()
        ]
    }

    # Export JSON
    splits_json_path = DATA_DIR / "channel_splits.json"
    with open(splits_json_path, "w", encoding="utf-8") as f:
        json.dump(split_payload, f, indent=2)
    logger.info(f"Saved partition mapping: {splits_json_path}")

    # Export Partitioned CSV
    partitioned_csv_path = DATA_DIR / "partitioned_payoff_data.csv"
    df.to_csv(partitioned_csv_path, index=False)
    logger.info(f"Saved partitioned video dataset: {partitioned_csv_path} ({len(df)} rows)")

    # 6. Balance & Integrity Diagnostic Table
    tier_balance_table = channel_df.groupby(["subscriber_tier", "partition"]).agg(
        channel_count=("channel_id", "count"),
        total_videos=("video_count", "sum")
    ).unstack(fill_value=0)
    logger.info("\n" + "=" * 50)
    logger.info("TIER BALANCE & SAMPLE SIZE MATRIX:")
    logger.info("\n" + tier_balance_table.to_string())

    logger.info("=" * 70)
    logger.info("STEP 2 COMPLETED SUCCESSFULLY: STRATIFIED SPLIT & PEER ISOLATION ESTABLISHED")
    logger.info("=" * 70)


if __name__ == "__main__":
    run_stratified_split()
