# step 3: aggregate scores into daily features, one row per NYSE trading session
# 16:00 ET cutoff prevents look-ahead bias; source z-scoring reduces outlet tone differences

import duckdb
import numpy as np
import pandas as pd
import exchange_calendars as ecals

from .config import DUCKDB_PATH, START_DATE, END_DATE, DAILY_SENTIMENT_CSV


def to_nyse_trading_day(event_time_utc: pd.Series) -> pd.Series:
    # before 16:00 ET -> same session, at/after 16:00 ET -> next session
    # snap weekends/holidays forward to the next valid NYSE open
    cal = ecals.get_calendar("XNYS")

    et = event_time_utc.dt.tz_convert("America/New_York")
    close_time = pd.Timestamp("16:00:00").time()

    # Build target date: same day if before close, next day if at/after close
    dates = et.dt.normalize().dt.tz_localize(None)  # midnight, tz-naive
    after_close = pd.Series(
        [t >= close_time if t is not None else False for t in et.dt.time],
        index=event_time_utc.index,
    )
    dates = dates.where(~after_close, dates + pd.Timedelta(days=1))

    # Snap every target date to the next valid NYSE session
    snapped = []
    for d in dates:
        try:
            snapped.append(
                pd.Timestamp(cal.date_to_session(d, direction="next"))
            )
        except Exception:
            snapped.append(pd.NaT)
    return pd.to_datetime(snapped)


def main() -> None:
    con = duckdb.connect(str(DUCKDB_PATH))

    df = con.execute("""
        SELECT event_time_utc, source, sentiment_score
        FROM news_scored
        WHERE event_time_utc IS NOT NULL
    """).fetch_df()

    con.close()

    if df.empty:
        raise RuntimeError(
            "news_scored is empty. Run 02_score_sentiment.py first."
        )

    df["event_time_utc"] = pd.to_datetime(
        df["event_time_utc"], utc=True, errors="coerce"
    )
    df = df.dropna(subset=["event_time_utc"])

    # Map each item to a trading day (session date)
    print("Mapping timestamps to NYSE trading sessions...")
    df["session"] = to_nyse_trading_day(df["event_time_utc"])
    df = df.dropna(subset=["session"])

    # Source-normalize sentiment to reduce outlet tone bias
    g = df.groupby("source")["sentiment_score"]
    df["sent_z"] = (df["sentiment_score"] - g.transform("mean")) / (
        g.transform("std").replace(0, np.nan)
    )
    df["sent_z"] = df["sent_z"].fillna(0.0)

    # Daily aggregation
    daily = (
        df.groupby("session")
        .agg(
            news_count=("sent_z", "size"),
            sent_mean=("sent_z", "mean"),
            sent_std=("sent_z", "std"),
            sent_sum=("sent_z", "sum"),
        )
        .reset_index()
    )

    daily["sent_std"] = daily["sent_std"].fillna(0.0)

    daily = daily.sort_values("session").reset_index(drop=True)

    start = pd.to_datetime(START_DATE)
    end = pd.to_datetime(END_DATE)
    daily = daily[
        (daily["session"] >= start) & (daily["session"] <= end)
    ].copy()

    daily[["news_count", "sent_mean", "sent_std", "sent_sum"]] = daily[
        ["news_count", "sent_mean", "sent_std", "sent_sum"]
    ].fillna(0.0)

    daily.to_csv(DAILY_SENTIMENT_CSV, index=False)
    print(f"Wrote {len(daily):,} rows -> {DAILY_SENTIMENT_CSV}")


if __name__ == "__main__":
    main()
