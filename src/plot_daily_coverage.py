# stacked bar chart of weekly article counts by source (resampled to weekly so 15 years is readable)
# run: python -m src.plot_daily_coverage

import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker

from .config import DUCKDB_PATH, NYT_RELEVANCE_REGEX, OUT_DIR


def main() -> None:
    con = duckdb.connect(str(DUCKDB_PATH), read_only=True)

    # Pull daily counts per subset (applying the same NYT relevance filter)
    df = con.execute(f"""
        SELECT
            CAST(event_time_utc AS DATE) AS day,
            subset,
            COUNT(*) AS n
        FROM news_raw
        WHERE event_time_utc IS NOT NULL
          AND (
              subset != 'nyt_articles_2000_present'
              OR regexp_matches(lower(text), '{NYT_RELEVANCE_REGEX}')
          )
        GROUP BY day, subset
        ORDER BY day
    """).fetch_df()

    con.close()

    df["day"] = pd.to_datetime(df["day"])

    # Pivot to wide format: one column per subset
    pivot = df.pivot_table(
        index="day", columns="subset", values="n", fill_value=0
    )

    label_map = {
        "sp500_daily_headlines": "S&P 500 Daily Headlines",
        "nyt_articles_2000_present": "New York Times Articles",
        "reddit_finance_sp500": "Reddit Finance SP500",
    }
    colour_map = {
        "sp500_daily_headlines": "#1f77b4",
        "nyt_articles_2000_present": "#d62728",
        "reddit_finance_sp500": "#ff7f0e",
    }

    col_order = [
        c for c in [
            "sp500_daily_headlines",
            "nyt_articles_2000_present",
            "reddit_finance_sp500",
        ]
        if c in pivot.columns
    ]
    pivot = pivot[col_order]

    weekly = pivot.resample("W").sum()

    fig, ax = plt.subplots(figsize=(20, 6))

    bottom = None
    for col in col_order:
        label = label_map.get(col, col)
        colour = colour_map.get(col, None)
        vals = weekly[col].values
        if bottom is None:
            ax.bar(
                weekly.index, vals, width=6,
                label=label, color=colour, alpha=0.85,
            )
            bottom = vals.copy()
        else:
            ax.bar(
                weekly.index, vals, width=6,
                bottom=bottom, label=label, color=colour, alpha=0.85,
            )
            bottom += vals

    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Articles per Week (finance-relevant)", fontsize=12)

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{int(x):,}")
    )

    ax.legend(loc="upper left", fontsize=11, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_xlim(pd.Timestamp("2010-01-01"), pd.Timestamp("2025-07-31"))

    plt.tight_layout()

    out_path = OUT_DIR / "daily_news_coverage.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved chart -> {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
