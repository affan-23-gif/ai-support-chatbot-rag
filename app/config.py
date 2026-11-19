"""App settings, read from environment variables or a .env file."""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    # "gemini" uses the real API. "fake" runs fully offline (tests, demos without a key).
    llm_provider: str = "gemini"
    gemini_api_key: str = ""
    chat_model: str = "gemini-3.5-flash-lite"
    # Model that grades answers in the eval. Each Gemini model has its own free quota,
    # so a different model here halves the load on CHAT_MODEL. Empty = same as CHAT_MODEL.
    judge_model: str = ""
    embedding_model: str = "gemini-embedding-001"
    embedding_dim: int = 768
    # Max chat-model requests per minute. Free tier allows 15 for gemini-3.5-flash-lite;
    # 12 leaves headroom. Raise it if you enable billing.
    gemini_rpm: int = 12

    docs_dir: Path = ROOT / "data" / "docs"
    orders_csv: Path = ROOT / "data" / "orders.csv"
    index_dir: Path = ROOT / "data" / "index"
    db_path: Path = ROOT / "data" / "supportbot.db"

    chunk_size: int = 900          # characters per chunk
    chunk_overlap: int = 150
    top_k: int = 5                 # chunks sent to the LLM
    retrieval_mode: str = "hybrid" # "hybrid" or "vector"
    # Below this cosine similarity (and with no keyword match) we refuse instead of guessing.
    min_similarity: float = 0.30
    history_turns: int = 6

    # ---- production settings ----
    # Password for /admin and the admin API (user name "admin"). Empty = admin is switched off.
    admin_password: str = ""
    # Max chat messages per client IP per minute. Protects your Gemini quota from spam.
    chat_rate_limit: int = 20
    # Behind a proxy (Render, Railway, nginx) the real client IP is in X-Forwarded-For.
    trust_proxy: bool = False
    session_ttl_hours: int = 24
    # Wrong order-number/email combinations allowed per session before we stop and hand off.
    max_order_lookup_failures: int = 3
    max_upload_mb: int = 10
    support_email: str = "support@pinecrest-demo.example"
    # Optional: Slack (or any) incoming-webhook URL. Each new ticket is posted there.
    ticket_webhook_url: str = ""


settings = Settings()
