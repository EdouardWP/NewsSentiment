import json
from typing import Any, Dict, Optional

import pandas as pd


def safe_json_loads(s: str) -> Dict[str, Any]:
    try:
        return json.loads(s) if isinstance(s, str) else {}
    except Exception:
        return {}


def parse_event_time_utc(date_str: str, extra_fields_str: str) -> pd.Timestamp:
    # prefer extra_fields.date_trading if it's there, otherwise just use top-level date
    extra = safe_json_loads(extra_fields_str)
    if "date_trading" in extra and extra["date_trading"]:
        return pd.to_datetime(extra["date_trading"], utc=True, errors="coerce")
    return pd.to_datetime(date_str, utc=True, errors="coerce")


def extract_source(extra_fields_str: str) -> str:
    # grab whatever looks like a publisher name; fallback to dataset
    extra = safe_json_loads(extra_fields_str)
    for k in [
        "publisher", "publication", "source", "source_norm",
        "outlet", "source_domain", "news_outlet",
    ]:
        v = extra.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    ds = extra.get("dataset")
    return ds.strip() if isinstance(ds, str) and ds.strip() else "unknown"


def extract_subset(extra_fields_str: str) -> str:
    extra = safe_json_loads(extra_fields_str)
    ds = extra.get("dataset")
    return ds.strip() if isinstance(ds, str) and ds.strip() else "unknown"


def normalize_text_for_model(text: str, max_chars: int = 4000) -> str:
    # truncate anything massive — FinBERT is fine with short text anyway
    if not isinstance(text, str):
        return ""
    t = " ".join(text.split())
    return t[:max_chars]
