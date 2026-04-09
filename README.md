# News Sentiment Pipeline

Builds a daily sentiment feature table from multi-source financial news, for use as an input to a PPO-based SPY trading agent.

**Output:** `out/daily_sentiment.csv` — one row per NYSE trading session with columns: `session`, `news_count`, `sent_mean`, `sent_std`, `sent_sum`.

## Data Sources

Three subsets from the [Multi-Source Financial News](https://huggingface.co/datasets/Brianferrell787/financial-news-multisource) dataset on Hugging Face:

- **S&P 500 Daily Headlines** — market recap headlines
- **New York Times Articles** — filtered to finance-relevant articles using keyword matching
- **Reddit Finance SP500** — posts from financial subreddits

## Setup

```bash
pip install -r requirements.txt
huggingface-cli login   # required for dataset access
```

## Pipeline (run in order)

### Step 1: Ingest news to DuckDB

Streams the 3 subsets from Hugging Face, filters to 2010–2025, and writes to a local DuckDB database. Includes progress bar with total row count and checkpoint/resume support.

```bash
python -m src.01_ingest_to_duckdb
```

### Step 2: Score sentiment with FinBERT

Scores each article using FinBERT (`ProsusAI/finbert`). NYT articles are filtered to finance-relevant content using keyword matching (reducing ~999K articles to ~102K). Auto-detects GPU (MPS/CUDA). Idempotent — safe to interrupt and resume.

```bash
python -m src.02_score_sentiment
```

### Step 3: Build daily sentiment features

Aggregates scored articles into daily features per NYSE trading session. Applies source-level z-score normalisation (reduces cross-outlet tone bias), then computes daily mean, std, sum, and count. Uses a 16:00 ET cutoff to prevent look-ahead bias.

```bash
python -m src.03_build_daily_sentiment
```

## Optional: Plot daily news coverage

Generates a stacked bar chart showing weekly article counts by source (saved to `out/daily_news_coverage.png`).

```bash
python -m src.plot_daily_coverage
```

## Key Design Decisions

- **No look-ahead bias**: the 16:00 ET session cutoff ensures day-t features only contain news available before market close on day t.
- **Source z-scoring**: per-source normalisation reduces systematic tone differences (e.g., Reddit skewing more negative than NYT).
- **Relevance filtering**: NYT articles are keyword-filtered to finance/market topics; the other two subsets are kept in full as they are already domain-specific.
