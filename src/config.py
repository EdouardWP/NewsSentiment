from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUT_DIR = PROJECT_ROOT / "out"
OUT_DIR.mkdir(exist_ok=True, parents=True)

DUCKDB_PATH = OUT_DIR / "news.duckdb"

HF_DATASET = "Brianferrell787/financial-news-multisource"

HF_DATA_FILES = [
    "data/sp500_daily_headlines/*.parquet",
    "data/nyt_articles_2000_present/*.parquet",
    "data/reddit_finance_sp500/*.parquet",
]

# Time window for paper
START_DATE = "2010-01-01"
END_DATE = "2025-12-31"

# Ingest batching (rows per write chunk)
INGEST_BUFFER_ROWS = 50_000

# Sentiment batching (GPU batch size; CPU fallback uses 32)
SENT_BATCH_SIZE_GPU = 128
SENT_BATCH_SIZE_CPU = 32

# Model: FinBERT-style sentiment classifier
SENTIMENT_MODEL = "ProsusAI/finbert"

# Relevance filter: applied to NYT articles only 
NYT_RELEVANCE_REGEX = (
    r"(stock|market|s&p|s\.p\.|spy|nasdaq|dow jones|nyse|wall street|"
    r"investor|trading|trader|earnings|revenue|profit|dividend|"
    r"merger|acquisition|ipo|bond|treasury|yield|"
    r"inflation|interest rate|federal reserve|the fed |"
    r"gdp|recession|economic|economy|fiscal|monetary|"
    r"bank|banking|hedge fund|private equity|venture capital|"
    r"bull market|bear market|rally|selloff|sell-off|"
    r"oil price|crude oil|commodit|bitcoin|crypto|"
    r"quarter|quarterly|fiscal year|annual report|"
    r"ceo |cfo |chief executive|chief financial|"
    r"share price|market cap|valuation|"
    r"dow |ftse|index fund|etf |mutual fund)"
)

# Output files
DAILY_SENTIMENT_CSV = OUT_DIR / "daily_sentiment.csv"
