import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
EXPORTS_DIR = DATA_DIR / "exports"
UPLOADS_DIR = DATA_DIR / "uploads"
PROJECTS_DIR = DATA_DIR / "projects"
PARQUET_DIR = DATA_DIR / "parquet"
CACHE_DIR = DATA_DIR / "cache"
ANALYTICS_RESULTS_DIR = DATA_DIR / "analytics_results"

for directory in [DATA_DIR, EXPORTS_DIR, UPLOADS_DIR, PROJECTS_DIR, PARQUET_DIR, CACHE_DIR, ANALYTICS_RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "heuristic-offline")
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "4096"))
DUCKDB_PATH = str(DATA_DIR / "analytics.duckdb")
AUTH_DB_PATH = DATA_DIR / "auth.db"

# Database Backend Configuration (sqlite, mysql, postgres)
DEFAULT_NEON_URL = "postgresql://neondb_owner:npg_O2UpcDyJLb9W@ep-holy-breeze-b4do39st.c-6.us-east-2.aws.neon.tech/neondb?sslmode=require&channel_binding=require"
NEON_DATABASE_URL = os.getenv("NEON_DATABASE_URL") or os.getenv("DATABASE_URL") or DEFAULT_NEON_URL
_default_db = "postgres" if NEON_DATABASE_URL else "sqlite"
DB_TYPE = os.getenv("DB_TYPE", _default_db).lower()

MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "analytics_platform")

PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = int(os.getenv("PG_PORT", "5432"))
PG_USER = os.getenv("PG_USER", "postgres")
PG_PASSWORD = os.getenv("PG_PASSWORD", "")
PG_DATABASE = os.getenv("PG_DATABASE", "analytics_platform")

# Notification & Dispatch Configuration (Email & Phone / SMS)
SMTP_SERVER = os.getenv("SMTP_SERVER") or "smtp.gmail.com"
SMTP_PORT = int(os.getenv("SMTP_PORT") or 587)
SMTP_USERNAME = os.getenv("SMTP_USERNAME") or "chintukumaredu00@gmail.com"
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD") or "bcrskjjmnkgfyqol"
SMTP_FROM = os.getenv("SMTP_FROM") or os.getenv("SMTP_USERNAME") or "chintukumaredu00@gmail.com"
SMTP_USE_TLS = os.getenv("SMTP_USE_TLS", "true").lower() in ("true", "1", "yes")

# HTTPS Transactional Email APIs (Recommended for Cloud Deployments like Render Free Tier)
RESEND_API_KEY = os.getenv("RESEND_API_KEY", "")
BREVO_API_KEY = os.getenv("BREVO_API_KEY", "")
DEFAULT_EMAIL_RELAY = "https://earlier-reduces-rebates-winning.trycloudflare.com/dispatch_email"
EMAIL_WEBHOOK_URL = os.getenv("EMAIL_WEBHOOK_URL") or DEFAULT_EMAIL_RELAY

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER", "")

# Base URL for Verification Links (auto-detects Render URL if deployed)
_render_url = os.getenv("RENDER_EXTERNAL_URL") or "https://ai-data-analyst-platform-7rhs.onrender.com"
APP_BASE_URL = os.getenv("APP_BASE_URL") or _render_url
ENABLE_TEST_OTP_PREVIEW = os.getenv("ENABLE_TEST_OTP_PREVIEW", "false").lower() in ("true", "1", "yes")
