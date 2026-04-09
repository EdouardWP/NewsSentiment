# step 2: run FinBERT on every unscored row in news_raw
# idempotent via raw_rowid — safe to kill and restart at any point

import time

import duckdb
import pandas as pd
import torch
from tqdm import tqdm
from transformers import pipeline

from .config import (
    DUCKDB_PATH,
    SENTIMENT_MODEL,
    SENT_BATCH_SIZE_GPU,
    SENT_BATCH_SIZE_CPU,
    NYT_RELEVANCE_REGEX,
)

# Max token length for FinBERT (shorter = faster, 256 covers headlines + leads)
MAX_TOKEN_LENGTH = 256

# Rows to accumulate before flushing to DuckDB
# At ~36-72 rows/sec, 10k rows ≈ 2-5 min of work lost on crash (max)
FLUSH_EVERY = 10_000


def _pick_device() -> tuple[int | str, int]:
    # cuda > mps > cpu; returns (device, batch_size)
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(f"Using CUDA GPU: {name}")
        return 0, SENT_BATCH_SIZE_GPU
    if torch.backends.mps.is_available():
        print("Using Apple Silicon MPS GPU")
        return "mps", SENT_BATCH_SIZE_GPU
    print("No GPU detected — using CPU (this will be slower)")
    return -1, SENT_BATCH_SIZE_CPU


def _estimate_time(
    clf, sample_texts: list[str], total_rows: int, batch_size: int
) -> None:
    # quick warm-up so we can give a time estimate before the main loop
    n_warmup = min(batch_size * 3, len(sample_texts))
    if n_warmup == 0:
        return

    warmup_texts = sample_texts[:n_warmup]
    t0 = time.perf_counter()
    for i in range(0, len(warmup_texts), batch_size):
        clf(warmup_texts[i : i + batch_size])
    elapsed = time.perf_counter() - t0

    rows_per_sec = n_warmup / elapsed
    est_seconds = total_rows / rows_per_sec
    est_minutes = est_seconds / 60
    est_hours = est_seconds / 3600

    print(
        f"\nWarm-up: {n_warmup} rows in {elapsed:.1f}s "
        f"({rows_per_sec:.0f} rows/sec)"
    )
    if est_hours >= 1.0:
        print(
            f"Estimated total time: ~{est_hours:.1f} hours "
            f"({est_minutes:.0f} min)"
        )
    else:
        print(f"Estimated total time: ~{est_minutes:.0f} minutes")
    print()


def ensure_schema(con: duckdb.DuckDBPyConnection) -> None:
    # raw_rowid links back to news_raw so re-runs skip already-scored rows
    con.execute("""
        CREATE TABLE IF NOT EXISTS news_scored (
            raw_rowid       BIGINT,
            event_time_utc  TIMESTAMP,
            source          VARCHAR,
            subset          VARCHAR,
            sentiment_label VARCHAR,
            sentiment_score DOUBLE
        );
    """)
    con.execute(
        "CREATE INDEX IF NOT EXISTS idx_scored_time "
        "ON news_scored(event_time_utc);"
    )
    # Unique on raw_rowid — each news_raw row is scored at most once
    try:
        con.execute(
            "CREATE UNIQUE INDEX idx_scored_rowid "
            "ON news_scored(raw_rowid);"
        )
    except duckdb.CatalogException:
        pass  # index already exists


def label_to_signed_score(label: str, score: float) -> float:
    # positive -> +score, negative -> -score, neutral -> 0
    l = (label or "").lower()
    if "pos" in l:
        return float(score)
    if "neg" in l:
        return -float(score)
    return 0.0


def flush_to_db(
    con: duckdb.DuckDBPyConnection, out_rows: list[dict]
) -> None:
    if not out_rows:
        return
    df_out = pd.DataFrame(out_rows)
    con.register("tmp_scored", df_out)
    con.execute("INSERT INTO news_scored SELECT * FROM tmp_scored;")


def main() -> None:
    con = duckdb.connect(str(DUCKDB_PATH))
    ensure_schema(con)

    # NYT gets filtered to finance-relevant only; the other two are already domain-specific
    relevance_sql = f"""
        SELECT r.rowid AS raw_rowid,
               r.event_time_utc, r.source, r.subset, r.text
        FROM news_raw r
        WHERE r.rowid NOT IN (SELECT raw_rowid FROM news_scored)
          AND (
              r.subset != 'nyt_articles_2000_present'
              OR regexp_matches(lower(r.text), '{NYT_RELEVANCE_REGEX}')
          )
    """

    todo = con.execute(relevance_sql).fetch_df()

    if todo.empty:
        print("No rows to score. news_scored already up to date.")
        con.close()
        return

    already_scored = con.execute(
        "SELECT COUNT(*) FROM news_scored"
    ).fetchone()[0]

    # Show filter impact
    total_raw = con.execute("SELECT COUNT(*) FROM news_raw").fetchone()[0]
    nyt_total = con.execute(
        "SELECT COUNT(*) FROM news_raw "
        "WHERE subset = 'nyt_articles_2000_present'"
    ).fetchone()[0]
    nyt_kept = len(todo[todo["subset"] == "nyt_articles_2000_present"])

    print(
        f"Relevance filter: {nyt_kept:,} / {nyt_total:,} NYT articles "
        f"kept ({100 * nyt_kept / nyt_total:.1f}%)"
    )
    print("Other subsets:    kept all (already finance-domain)")
    print(
        f"Rows to score:    {len(todo):,}  "
        f"(was {total_raw:,} before filter)"
    )
    print(f"Already scored:   {already_scored:,}")
    print(f"Total when done:  {already_scored + len(todo):,}")

    device, batch_size = _pick_device()
    print(f"Batch size:       {batch_size}")

    clf = pipeline(
        "text-classification",
        model=SENTIMENT_MODEL,
        tokenizer=SENTIMENT_MODEL,
        truncation=True,
        max_length=MAX_TOKEN_LENGTH,
        device=device,
    )

    sample_texts = (
        todo["text"].fillna("").head(batch_size * 3).tolist()
    )
    _estimate_time(
        clf, sample_texts, total_rows=len(todo), batch_size=batch_size
    )

    out_rows: list[dict] = []
    total_flushed = 0
    t_start = time.perf_counter()

    num_batches = (len(todo) + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc="Scoring sentiment"):
        i = batch_idx * batch_size
        batch = todo.iloc[i : i + batch_size]
        texts = batch["text"].fillna("").tolist()

        preds = clf(texts)

        for idx, pred in enumerate(preds):
            label = pred.get("label", "")
            score = float(pred.get("score", 0.0))
            signed = label_to_signed_score(label, score)

            out_rows.append(
                {
                    "raw_rowid": int(batch.iloc[idx]["raw_rowid"]),
                    "event_time_utc": batch.iloc[idx]["event_time_utc"],
                    "source": batch.iloc[idx]["source"],
                    "subset": batch.iloc[idx]["subset"],
                    "sentiment_label": label,
                    "sentiment_score": signed,
                }
            )

        # Flush periodically
        if len(out_rows) >= FLUSH_EVERY:
            flush_to_db(con, out_rows)
            total_flushed += len(out_rows)
            out_rows.clear()

    # Final flush
    if out_rows:
        flush_to_db(con, out_rows)
        total_flushed += len(out_rows)

    elapsed = time.perf_counter() - t_start
    rate = total_flushed / elapsed if elapsed > 0 else 0

    print(
        f"\nDone. Scored {total_flushed:,} rows in "
        f"{elapsed / 60:.1f} minutes"
    )
    print(f"  Throughput: {rate:.0f} rows/sec")
    print(f"  Total in news_scored: {already_scored + total_flushed:,}")

    con.close()


if __name__ == "__main__":
    main()
