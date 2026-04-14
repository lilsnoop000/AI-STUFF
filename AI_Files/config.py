import os
import sys

# ==============================
# KOGNAC AI CONFIGURATION v10
# ==============================

# --- Network ---
VPS_IP = os.environ.get("AI_VPS_IP", "178.104.172.121")
VPS_PORT = int(os.environ.get("AI_VPS_PORT", 11434))
VPS_BASE_URL = f"http://{VPS_IP}:{VPS_PORT}"
VPS_FALLBACK_URL = os.environ.get("AI_VPS_FALLBACK", "")
VPS_CACHE_TTL_SECONDS = 60

# --- Auth ---
API_KEY = os.environ.get("AI_API_KEY", "")

# --- Models ---
MODEL_FAST = os.environ.get("AI_MODEL_FAST", "phi3:mini")
MODEL_DEEP = os.environ.get("AI_MODEL_DEEP", "mistral:7b-instruct-q4_0")

# --- Paths ---
BASE_DIR = os.path.realpath(os.path.dirname(os.path.abspath(__file__)))
KEY_PATH = os.path.join(BASE_DIR, "key.txt")
MEMORY_DIR = os.path.normpath(os.path.join(BASE_DIR, "../memory"))
DB_PATH = os.path.join(MEMORY_DIR, "memory.db")
MODEL_PATH = os.path.normpath(os.path.join(BASE_DIR, "../models/all-MiniLM-L6-v2"))
LOG_PATH = os.path.normpath(os.path.join(BASE_DIR, "../logs/ai_system.log"))
EXPORT_DIR = os.path.normpath(os.path.join(BASE_DIR, "../exports"))
SANDBOX_DIR = os.path.normpath(os.path.join(BASE_DIR, "../sandbox_tmp"))
CONTAINER_ROOT = os.path.dirname(BASE_DIR)

# --- Query ---
QUERY_TIMEOUT_SECONDS = 120
QUERY_RETRIES = 2
AGENT_FUTURE_TIMEOUT = 130
STREAM_ENABLED = True

# --- Memory ---
MAX_MEMORIES_PER_PROJECT = 10000
MAX_DB_SIZE_MB = 500
COMPRESS_CLUSTER_THRESHOLD = 5
COMPRESS_KEEP_TOP_N = 2
SEARCH_TOP_K = 6
MAX_CONTEXT_CHARS = 6000
REINDEX_BATCH_SIZE = 100

# --- RAG ---
SIMILARITY_THRESHOLD = 0.35
RELEVANCE_FILTER_ENABLED = True
DEDUP_ENABLED = True
DEDUP_THRESHOLD = 0.92
MEMORY_TYPES_ENABLED = True

# --- Sessions ---
MAX_CONVERSATION_TURNS = 10
MAX_SESSIONS = 20

# --- Ingest ---
INGEST_CHUNK_SIZE = 500
MAX_INGEST_FILE_SIZE_MB = 10
SMART_CHUNKING_ENABLED = True
INGESTDIR_MAX_FILES = 50
INGEST_EXCLUDE_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv",
                       ".tox", ".mypy_cache", "build", "dist", ".egg-info"}

# --- Input ---
MAX_PROMPT_LENGTH = 4000

# --- Background ---
BACKGROUND_INTERVAL_MINUTES = 5
KNOWLEDGE_REVIEW_INTERVAL_HOURS = 24
KNOWLEDGE_REVIEW_BATCH_SIZE = 10
KNOWLEDGE_REVIEW_DELETE = False

# --- Summarisation ---
SUMMARISE_MIN_WORDS = 8
SUMMARISE_MIN_RESPONSE_WORDS = 50
SUMMARISE_MODEL = "fast"

# --- Self-eval (opt-in) ---
SELF_EVAL_ENABLED = False
SELF_EVAL_MIN_WORDS = 15

# --- Routing ---
AUTO_ROUTE_ENABLED = True

# --- Sandbox ---
SANDBOX_ENABLED = True
SANDBOX_TIMEOUT_SECONDS = 10
SANDBOX_MAX_OUTPUT_CHARS = 10000

# --- Preferences ---
PREFERENCES_ENABLED = True
PREFERENCE_EXTRACT_EVERY_N = 5

# --- Hallucination ---
HALLUCINATION_THRESHOLD = 2

# --- Teaching ---
EVOLVE_KEYFILE_EVERY_N = 10
MAX_KEYFILE_SIZE_KB = 50

# --- Logging ---
MAX_LOG_SIZE_MB = 10
LOG_BACKUP_COUNT = 3

# --- Notifications ---
NOTIFICATIONS_ENABLED = True
NOTIFICATION_WEBHOOK_URL = ""
NOTIFICATION_RETENTION_DAYS = 7

# --- Undo ---
UNDO_RETENTION_HOURS = 24

# --- Server ---
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 5000
RATE_LIMIT_DEFAULT = "60 per minute"
RATE_LIMIT_ASK = "30 per minute"
RATE_LIMIT_STREAM = "20 per minute"
RATE_LIMIT_STORAGE = "memory://"

# --- FAISS ---
FAISS_ENABLED = True


# ==============================
# PATH VALIDATION
# ==============================
_validated = False


def validate_paths():
    global _validated
    if _validated:
        return
    _validated = True

    errors = []
    warnings = []

    if not os.path.isdir(BASE_DIR):
        errors.append(f"BASE_DIR missing: {BASE_DIR}")
    if not os.path.isdir(MODEL_PATH):
        warnings.append(f"Model path missing: {MODEL_PATH}")

    critical = {"DB_PATH": DB_PATH, "LOG_PATH": LOG_PATH, "KEY_PATH": KEY_PATH,
                "EXPORT_DIR": EXPORT_DIR, "SANDBOX_DIR": SANDBOX_DIR}
    for name, path in critical.items():
        resolved = os.path.realpath(path) if os.path.exists(path) else os.path.abspath(path)
        if not resolved.startswith(CONTAINER_ROOT):
            errors.append(f"SECURITY: {name} outside container: {resolved}")

    for w in warnings:
        print(f"[CONFIG WARNING] {w}", file=sys.stderr)
    if errors:
        print("\n[CONFIG] STARTUP ABORTED:\n", file=sys.stderr)
        for e in errors:
            print(f"  ❌ {e}\n", file=sys.stderr)
        sys.exit(1)

    for d in [EXPORT_DIR, SANDBOX_DIR, os.path.dirname(LOG_PATH), MEMORY_DIR]:
        os.makedirs(d, exist_ok=True)

    print("[CONFIG] Path validation passed.")
