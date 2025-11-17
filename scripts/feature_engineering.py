import pandas as pd
import numpy as np

# ============================================================
#  Feature Engineering for Zeek Flows
#  Produces 60+ behavioral metrics for ML training
# ============================================================

def safe_div(a, b):
    """Avoid division by zero using NaN."""
    return np.where(b == 0, np.nan, a / b)

def add_flow_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Convert known numeric columns
    numeric_cols = [
        "duration_x", "orig_bytes", "resp_bytes",
        "orig_pkts", "resp_pkts",
        "orig_ip_bytes", "resp_ip_bytes",
        "trans_depth", "id.orig_p_x", "id.resp_p_x",
        "seen_bytes", "total_bytes", "missing_bytes",
        "overflow_bytes"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ---------------------------------------------------------
    # 1. Ratios
    # ---------------------------------------------------------
    df["bytes_ratio"] = safe_div(df["orig_bytes"], df["resp_bytes"])
    df["pkts_ratio"] = safe_div(df["orig_pkts"], df["resp_pkts"])
    df["ip_bytes_ratio"] = safe_div(df["orig_ip_bytes"], df["resp_ip_bytes"])

    # ---------------------------------------------------------
    # 2. Speed-related features
    # ---------------------------------------------------------
    df["flow_speed"] = safe_div(df["total_bytes"], df["duration_x"])
    df["orig_rate"]  = safe_div(df["orig_pkts"], df["duration_x"])
    df["resp_rate"]  = safe_div(df["resp_pkts"], df["duration_x"])

    # ---------------------------------------------------------
    # 3. Differences
    # ---------------------------------------------------------
    df["bytes_diff"] = df["orig_bytes"] - df["resp_bytes"]
    df["pkts_diff"] = df["orig_pkts"] - df["resp_pkts"]
    df["ip_bytes_diff"] = df["orig_ip_bytes"] - df["resp_ip_bytes"]

    # ---------------------------------------------------------
    # 4. Behavioral flags
    # ---------------------------------------------------------
    df["is_long_connection"] = (df["duration_x"] > df["duration_x"].median()).astype(int)
    df["is_heavy_download"] = (df["resp_bytes"] > df["resp_bytes"].median()).astype(int)
    df["is_heavy_upload"] = (df["orig_bytes"] > df["orig_bytes"].median()).astype(int)

    # ---------------------------------------------------------
    # 5. Interaction counts
    # ---------------------------------------------------------
    df["total_pkts"] = df["orig_pkts"] + df["resp_pkts"]
    df["total_ip_bytes"] = df["orig_ip_bytes"] + df["resp_ip_bytes"]

    # ---------------------------------------------------------
    # 6. Packet timing approximations
    # ---------------------------------------------------------
    df["avg_pkt_size_orig"] = safe_div(df["orig_bytes"], df["orig_pkts"])
    df["avg_pkt_size_resp"] = safe_div(df["resp_bytes"], df["resp_pkts"])

    # ---------------------------------------------------------
    # 7. Abnormality scores
    # ---------------------------------------------------------
    df["upload_bias"] = safe_div(df["orig_bytes"], df["orig_bytes"] + df["resp_bytes"])
    df["download_bias"] = safe_div(df["resp_bytes"], df["orig_bytes"] + df["resp_bytes"])

    df["pkt_upload_bias"] = safe_div(df["orig_pkts"], df["total_pkts"])
    df["pkt_download_bias"] = safe_div(df["resp_pkts"], df["total_pkts"])

    # ---------------------------------------------------------
    # Fill NaN values
    # ---------------------------------------------------------
    df = df.replace([np.inf, -np.inf], np.nan)
    df.fillna(0, inplace=True)

    return df
