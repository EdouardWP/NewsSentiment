# step 1: stream the 3 HF subsets into a local DuckDB
# supports checkpoint/resume so a Ctrl-C doesn't mean starting over

import json
import duckdb
import pandas as pd
import pyarrow.parquet as pq
from datasets import load_dataset
from huggingface_hub import HfFileSystem
from tqdm import tqdm

from .config import (
    DUCKDB_PATH,
    HF_DATASET,
    HF_DATA_FILES,
    START_DATE,
    END_DATE,
)
from .utils import (
    parse_event_time_utc,
    extract_source,
    extract_subset,
    normalize_text_for_model,
)

# How many accepted rows between each DuckDB flush + checkpoint 
CHECKPOINT_ROWS = 500_000

# Internal write buffer size 
WRITE_BUFFER = 50_000

_CHECKPOINT_FILE = DUCKDB_PATH.parent / ".ingest_checkpoint.json"


def estimate_total_rows() -> int | None:
    # peek at parquet footers for the row count, no data downloaded, just metadata
    try:
        fs = HfFileSystem()
        total = 0
        for pattern in HF_DATA_FILES:
            path_pattern = f"datasets/{HF_DATASET}/{pattern}"
            matched_files = fs.glob(path_pattern)
            for fpath in matched_files:
                with fs.open(fpath) as fobj:
                    meta = pq.read_metadata(fobj)
                    total += meta.num_rows
        return total if total > 0 else None
    except Exception as e:
        print(f"  (could not pre-count rows: {e} — progress bar will have no total)")
        return None


def _load_checkpoint() -> int:
    # how many stream rows did we get through last time?
    try:
        data = json.loads(_CHECKPOINT_FILE.read_text())
        return int(data.get("stream_rows_processed", 0))
    except Exception:
        return 0


def _save_checkpoint(stream_rows_processed: int) -> None:
    _CHECKPOINT_FILE.write_text(
        json.dumps({"stream_rows_processed": stream_rows_processed})
    )


def _clear_checkpoint() -> None:
    _CHECKPOINT_FILE.unlink(missing_ok=True)


def ensure_schema(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("""
        CREATE TABLE IF NOT EXISTS news_raw (
            event_time_utc TIMESTAMP,
            date_utc       VARCHAR,
            text           VARCHAR,
            extra_fields   VARCHAR,
            source         VARCHAR,
            subset         VARCHAR
        );
    """)
    con.execute(
        "CREATE INDEX IF NOT EXISTS idx_news_time ON news_raw(event_time_utc);"
    )


def flush_buffer(
    con: duckdb.DuckDBPyConnection, buffer: list[dict]
) -> None:
    if not buffer:
        return
    df = pd.DataFrame(buffer)
    con.register("tmp_df", df)
    con.execute("INSERT INTO news_raw SELECT * FROM tmp_df;")


def print_summary(con: duckdb.DuckDBPyConnection) -> None:
    summary = con.execute("""
        SELECT
            subset,
            COUNT(*)                            AS row_count,
            MIN(event_time_utc)::DATE           AS earliest_date,
            MAX(event_time_utc)::DATE           AS latest_date,
            COUNT(DISTINCT YEAR(event_time_utc)) AS num_years
        FROM news_raw
        GROUP BY subset
        ORDER BY subset
    """).fetch_df()

    total_row = con.execute("""
        SELECT
            COUNT(*)                  AS row_count,
            MIN(event_time_utc)::DATE AS earliest_date,
            MAX(event_time_utc)::DATE AS latest_date
        FROM news_raw
    """).fetch_df()

    print("\n=== Ingest Summary ===")
    print(
        f"{'Subset':<40} {'Rows':>10}  {'Earliest':>12}  "
        f"{'Latest':>12}  {'Years':>5}"
    )
    print("-" * 85)
    for _, r in summary.iterrows():
        print(
            f"{r['subset']:<40} {r['row_count']:>10,}  "
            f"{r['earliest_date']}  {r['latest_date']}  {r['num_years']:>5}"
        )
    print("-" * 85)
    tr = total_row.iloc[0]
    print(
        f"{'TOTAL':<40} {tr['row_count']:>10,}  "
        f"{tr['earliest_date']}  {tr['latest_date']}"
    )
    print()


def main() -> None:
    con = duckdb.connect(str(DUCKDB_PATH))
    ensure_schema(con)

    print("Counting rows from parquet metadata (headers only)...")
    total_rows = estimate_total_rows()
    if total_rows is not None:
        print(f"Total rows in source parquets: {total_rows:,}")
        print(
            f"(Rows outside {START_DATE}..{END_DATE} will be filtered out)\n"
        )
    else:
        print(
            "Could not determine total — progress bar will show rows "
            "processed.\n"
        )

    skip_rows = _load_checkpoint()
    already_in_db = con.execute("SELECT COUNT(*) FROM news_raw").fetchone()[0]

    if skip_rows > 0 and already_in_db > 0:
        print(
            f"Resuming: skipping first {skip_rows:,} stream rows "
            f"({already_in_db:,} rows already in DB)."
        )
    elif already_in_db > 0 and skip_rows == 0:
        # DB has rows but no checkpoint — wipe and start fresh to avoid dupes
        print(
            f"Found {already_in_db:,} existing rows but no checkpoint. "
            "Clearing table to avoid duplicates..."
        )
        con.execute("DELETE FROM news_raw;")
        skip_rows = 0

    ds = load_dataset(
        HF_DATASET,
        data_files=HF_DATA_FILES,
        split="train",
        streaming=True,
    )

    start_ts = pd.to_datetime(START_DATE, utc=True)
    end_ts = (
        pd.to_datetime(END_DATE, utc=True)
        + pd.Timedelta(days=1)
        - pd.Timedelta(seconds=1)
    )

    buffer: list[dict] = []
    total_written = 0
    total_skipped_filter = 0
    stream_idx = 0  # counts every row from the stream
    rows_since_checkpoint = 0

    pbar = tqdm(ds, total=total_rows, desc="Streaming rows", initial=skip_rows)

    for row in pbar:
        stream_idx += 1

        # Skip rows already processed in a previous run
        if stream_idx <= skip_rows:
            continue

        date_str = row.get("date", "")
        text = row.get("text", "")
        extra = row.get("extra_fields", "")

        event_time = parse_event_time_utc(date_str, extra)
        if pd.isna(event_time):
            total_skipped_filter += 1
            continue
        if event_time < start_ts or event_time > end_ts:
            total_skipped_filter += 1
            continue

        buffer.append(
            {
                "event_time_utc": event_time.to_pydatetime(),
                "date_utc": date_str,
                "text": normalize_text_for_model(text),
                "extra_fields": extra,
                "source": extract_source(extra),
                "subset": extract_subset(extra),
            }
        )

        # Flush write buffer to DuckDB
        if len(buffer) >= WRITE_BUFFER:
            flush_buffer(con, buffer)
            total_written += len(buffer)
            rows_since_checkpoint += len(buffer)
            buffer.clear()

        # Checkpoint every CHECKPOINT_ROWS accepted rows
        if rows_since_checkpoint >= CHECKPOINT_ROWS:
            _save_checkpoint(stream_idx)
            pbar.set_postfix(
                written=f"{total_written + already_in_db:,}",
                ckpt=f"{stream_idx:,}",
            )
            rows_since_checkpoint = 0

    # Final flush
    if buffer:
        flush_buffer(con, buffer)
        total_written += len(buffer)

    _save_checkpoint(stream_idx)

    print(
        f"\nDone. Inserted {total_written:,} new rows "
        f"(skipped {total_skipped_filter:,} outside date range)."
    )
    print(f"Total in DB: {total_written + already_in_db:,} rows.")

    print_summary(con)

    _clear_checkpoint()
    con.close()


if __name__ == "__main__":
    main()
