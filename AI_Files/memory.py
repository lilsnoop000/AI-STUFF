import os
import sqlite3
import numpy as np
import json
import time
import threading
import schedule
from sentence_transformers import SentenceTransformer

from config import (
    DB_PATH, MODEL_PATH, EXPORT_DIR,
    MAX_MEMORIES_PER_PROJECT, MAX_DB_SIZE_MB,
    COMPRESS_CLUSTER_THRESHOLD, COMPRESS_KEEP_TOP_N,
    SEARCH_TOP_K, BACKGROUND_INTERVAL_MINUTES,
    MAX_CONTEXT_CHARS, FAISS_ENABLED, CONTAINER_ROOT,
    SIMILARITY_THRESHOLD, RELEVANCE_FILTER_ENABLED,
    DEDUP_ENABLED, DEDUP_THRESHOLD, MEMORY_TYPES_ENABLED,
    KNOWLEDGE_REVIEW_INTERVAL_HOURS, KNOWLEDGE_REVIEW_BATCH_SIZE,
    KNOWLEDGE_REVIEW_DELETE, PREFERENCES_ENABLED,
    REINDEX_BATCH_SIZE, NOTIFICATION_RETENTION_DAYS,
    UNDO_RETENTION_HOURS, MAX_CONVERSATION_TURNS
)
from logger import get_logger

log = get_logger("memory")

MEMORY_DIR = os.path.dirname(DB_PATH)
os.makedirs(MEMORY_DIR, exist_ok=True)

model = SentenceTransformer(MODEL_PATH, local_files_only=True)
EMBEDDING_DIM = model.get_sentence_embedding_dimension()
log.info(f"Embedding model loaded. dim={EMBEDDING_DIM}")


def encode_text(text):
    emb = model.encode(text).astype(np.float32)
    n = np.linalg.norm(emb)
    if n > 0:
        emb /= n
    return emb


# ==============================
# MEMORY TYPES
# ==============================
VALID_TYPES = {"fact", "procedure", "preference", "episodic", "ingested",
               "compressed", "query", "general"}


def classify_memory_type(content):
    if not MEMORY_TYPES_ENABLED:
        return "general"
    lo = content.lower()
    if lo.startswith("[ingested_source"): return "ingested"
    if lo.startswith("[compressed:"): return "compressed"
    if lo.startswith("[user_query]"): return "query"
    if lo.startswith("[preference]") or lo.startswith("[user_pref]"): return "preference"
    if lo.startswith("[exchange_summary]"): return "episodic"
    if lo.startswith("[project_created]"): return "general"
    proc = ["step 1", "first,", "to do this", "how to", "install ", "configure ", "set up "]
    if any(s in lo for s in proc): return "procedure"
    return "general"


# ==============================
# FAISS — generation counter eliminates all race conditions
# ==============================
_faiss_available = False
_faiss = None

if FAISS_ENABLED:
    try:
        import faiss as _faiss
        _faiss_available = True
        log.info("FAISS loaded")
    except ImportError:
        log.warning("FAISS not installed — numpy fallback")

_faiss_indexes = {}       # project -> (index, ids_list)
_faiss_generations = {}   # project -> int
_faiss_lock = threading.Lock()  # protects _faiss_indexes and _faiss_generations dicts ONLY


def _invalidate_faiss(project):
    """Increment generation counter. Next search will rebuild. No DB I/O."""
    with _faiss_lock:
        _faiss_generations[project] = _faiss_generations.get(project, 0) + 1
        _faiss_indexes.pop(project, None)


def _build_faiss_index(project):
    """Build FAISS index from DB. Discards result if generation changed during build."""
    if not _faiss_available:
        return

    # Read generation BEFORE building (under lock — fast)
    with _faiss_lock:
        gen_before = _faiss_generations.get(project, 0)

    # DB read WITHOUT lock — allows concurrent searches/writes
    conn = get_connection()
    rows = conn.execute("SELECT id, embedding FROM memories WHERE project=?", (project,)).fetchall()
    index = _faiss.IndexFlatIP(EMBEDDING_DIM)
    ids = []
    for mid, blob in rows:
        emb = np.frombuffer(blob, dtype=np.float32).copy()
        if emb.shape[0] != EMBEDDING_DIM:
            continue
        n = np.linalg.norm(emb)
        if n > 0:
            emb /= n
        index.add(emb.reshape(1, -1))
        ids.append(mid)

    # Install ONLY if generation hasn't changed (under lock — fast)
    with _faiss_lock:
        gen_after = _faiss_generations.get(project, 0)
        if gen_after == gen_before:
            _faiss_indexes[project] = (index, ids)
            log.debug(f"FAISS built: '{project}' — {len(ids)} vectors (gen={gen_after})")
        else:
            log.debug(f"FAISS build discarded: '{project}' gen {gen_before}→{gen_after}")


def _get_faiss_index(project):
    """Get current FAISS index, building if needed. Returns (index, ids) or None."""
    with _faiss_lock:
        entry = _faiss_indexes.get(project)
        if entry is not None:
            return entry

    _build_faiss_index(project)

    with _faiss_lock:
        return _faiss_indexes.get(project)


# ==============================
# DB CONNECTION — thread-local, WAL mode
# ==============================
_local = threading.local()


def get_connection():
    if not hasattr(_local, "conn") or _local.conn is None:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        _local.conn = conn
    return _local.conn


# ==============================
# SCHEMA — all tables, single DB, single source of truth
# ==============================
_initialized = False


def init():
    global _initialized
    if _initialized:
        return
    _initialized = True
    conn = get_connection()

    conn.executescript("""
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY, project TEXT, cluster TEXT, content TEXT,
            embedding BLOB, importance REAL, access_count INTEGER,
            last_access REAL, timestamp REAL, locked INTEGER DEFAULT 0,
            memory_type TEXT DEFAULT 'general', pinned INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS deleted_memories (
            id INTEGER PRIMARY KEY, orig_id INTEGER, project TEXT, cluster TEXT,
            content TEXT, embedding BLOB, importance REAL, access_count INTEGER,
            last_access REAL, timestamp REAL, locked INTEGER, memory_type TEXT,
            pinned INTEGER, deleted_at REAL
        );

        CREATE TABLE IF NOT EXISTS preferences (
            id INTEGER PRIMARY KEY, key TEXT UNIQUE, value TEXT,
            source TEXT DEFAULT 'auto', status TEXT DEFAULT 'active',
            updated_at REAL
        );

        CREATE TABLE IF NOT EXISTS relationships (
            id INTEGER PRIMARY KEY, memory_id_a INTEGER, memory_id_b INTEGER,
            rel_type TEXT DEFAULT 'related', created_at REAL,
            UNIQUE(memory_id_a, memory_id_b)
        );

        CREATE TABLE IF NOT EXISTS session_messages (
            id INTEGER PRIMARY KEY, session_name TEXT, role TEXT,
            content TEXT, timestamp REAL
        );

        CREATE TABLE IF NOT EXISTS config_state (
            key TEXT PRIMARY KEY, value TEXT
        );

        CREATE TABLE IF NOT EXISTS notifications (
            id INTEGER PRIMARY KEY, timestamp REAL, level TEXT,
            source TEXT, message TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_mem_project ON memories(project);
        CREATE INDEX IF NOT EXISTS idx_mem_cluster ON memories(cluster);
        CREATE INDEX IF NOT EXISTS idx_mem_type ON memories(memory_type);
        CREATE INDEX IF NOT EXISTS idx_mem_pinned ON memories(pinned);
        CREATE INDEX IF NOT EXISTS idx_rel_a ON relationships(memory_id_a);
        CREATE INDEX IF NOT EXISTS idx_rel_b ON relationships(memory_id_b);
        CREATE INDEX IF NOT EXISTS idx_sess ON session_messages(session_name);
        CREATE INDEX IF NOT EXISTS idx_notif_ts ON notifications(timestamp);
        CREATE INDEX IF NOT EXISTS idx_del_ts ON deleted_memories(deleted_at);
    """)

    # Migrate: add columns if missing (for upgrades from older schemas)
    for col, typ, default in [
        ("memory_type", "TEXT", "'general'"),
        ("pinned", "INTEGER", "0"),
    ]:
        try:
            conn.execute(f"SELECT {col} FROM memories LIMIT 1")
        except sqlite3.OperationalError:
            conn.execute(f"ALTER TABLE memories ADD COLUMN {col} {typ} DEFAULT {default}")
            log.info(f"Migrated: added {col}")

    # Migrate preferences: add status column
    try:
        conn.execute("SELECT status FROM preferences LIMIT 1")
    except sqlite3.OperationalError:
        try:
            conn.execute("ALTER TABLE preferences ADD COLUMN status TEXT DEFAULT 'active'")
            log.info("Migrated: added preferences.status")
        except sqlite3.OperationalError:
            pass

    # Migrate tutor_state.json → config_state table (one-time)
    _migrate_json_state(conn)

    conn.commit()
    log.info("DB initialized — all tables ready")


def _migrate_json_state(conn):
    """One-time migration from tutor_state.json to config_state table."""
    json_path = os.path.join(os.path.dirname(DB_PATH), "..", "AI_Files", "tutor_state.json")
    # Also check BASE_DIR relative path
    from config import BASE_DIR
    alt_path = os.path.join(BASE_DIR, "tutor_state.json")

    for path in [alt_path, json_path]:
        if os.path.exists(path):
            # Only migrate if config_state is empty
            existing = conn.execute("SELECT COUNT(*) FROM config_state").fetchone()[0]
            if existing > 0:
                return
            try:
                with open(path, "r") as f:
                    state = json.load(f)
                for k, v in state.items():
                    val = json.dumps(v) if isinstance(v, (dict, list, bool)) else str(v)
                    conn.execute("INSERT OR IGNORE INTO config_state (key, value) VALUES (?, ?)", (k, val))
                conn.commit()
                log.info(f"Migrated state from {path} to config_state table")
                # Rename old file so migration doesn't repeat
                os.rename(path, path + ".migrated")
            except Exception as e:
                log.warning(f"State migration failed: {e}")
            return


# ==============================
# CONFIG STATE (replaces tutor_state.json — no filelock needed)
# ==============================
_state_defaults = {
    "confusion_topics": "{}",
    "teaching_style": "balanced",
    "teaching_enabled": "true",
    "total_interactions": "0",
    "model_mode": "hybrid",
    "active_model": "fast",
}


def load_config_state():
    conn = get_connection()
    rows = conn.execute("SELECT key, value FROM config_state").fetchall()
    state = dict(_state_defaults)
    for k, v in rows:
        state[k] = v
    # Parse types
    result = {}
    for k, v in state.items():
        if k == "confusion_topics":
            try:
                result[k] = json.loads(v)
            except (json.JSONDecodeError, TypeError):
                result[k] = {}
        elif k in ("teaching_enabled",):
            result[k] = v.lower() in ("true", "1", "yes") if isinstance(v, str) else bool(v)
        elif k == "total_interactions":
            try:
                result[k] = int(v)
            except (ValueError, TypeError):
                result[k] = 0
        else:
            result[k] = v
    return result


def save_config_state(state):
    conn = get_connection()
    for k, v in state.items():
        if isinstance(v, (dict, list)):
            val = json.dumps(v)
        elif isinstance(v, bool):
            val = "true" if v else "false"
        else:
            val = str(v)
        conn.execute("INSERT INTO config_state (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=?",
                     (k, val, val))
    conn.commit()


def increment_interactions():
    """Atomic increment — no read-modify-write race."""
    conn = get_connection()
    conn.execute("""
        INSERT INTO config_state (key, value) VALUES ('total_interactions', '1')
        ON CONFLICT(key) DO UPDATE SET value = CAST(CAST(value AS INTEGER) + 1 AS TEXT)
    """)
    conn.commit()


# ==============================
# NOTIFICATIONS (in SQLite, not JSONL)
# ==============================
def add_notification(message, level="info", source="system"):
    conn = get_connection()
    conn.execute("INSERT INTO notifications (timestamp, level, source, message) VALUES (?,?,?,?)",
                 (time.time(), level, source, message))
    conn.commit()

    # Optional webhook
    from config import NOTIFICATION_WEBHOOK_URL
    if NOTIFICATION_WEBHOOK_URL:
        import requests as _req
        threading.Thread(target=_post_webhook,
                         args=(NOTIFICATION_WEBHOOK_URL, message, level, source), daemon=True).start()


def _post_webhook(url, message, level, source):
    import requests as _req
    try:
        r = _req.post(url, json={"level": level, "source": source, "message": message,
                                  "time": time.strftime("%Y-%m-%d %H:%M:%S")}, timeout=5)
        if r.status_code >= 400:
            log.warning(f"Webhook HTTP {r.status_code}")
    except _req.exceptions.ConnectionError:
        log.warning(f"Webhook unreachable: {url}")
    except Exception as e:
        log.warning(f"Webhook error: {e}")


def get_notifications(since=0, limit=20):
    conn = get_connection()
    rows = conn.execute("""
        SELECT timestamp, level, source, message FROM notifications
        WHERE timestamp > ? ORDER BY timestamp DESC LIMIT ?
    """, (since, limit)).fetchall()
    return [{"time": r[0], "time_str": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r[0])),
             "level": r[1], "source": r[2], "message": r[3]} for r in reversed(rows)]


def get_unread_count(since=0):
    conn = get_connection()
    return conn.execute("SELECT COUNT(*) FROM notifications WHERE timestamp > ?", (since,)).fetchone()[0]


def purge_old_notifications():
    cutoff = time.time() - NOTIFICATION_RETENTION_DAYS * 86400
    conn = get_connection()
    conn.execute("DELETE FROM notifications WHERE timestamp < ?", (cutoff,))
    conn.commit()


# ==============================
# SESSION MESSAGES (append-only, replaces JSON blob)
# ==============================
def append_session_message(session_name, role, content):
    conn = get_connection()
    conn.execute("INSERT INTO session_messages (session_name, role, content, timestamp) VALUES (?,?,?,?)",
                 (session_name, role, content[:2000], time.time()))
    conn.commit()


def load_session_messages(session_name, limit=None):
    if limit is None:
        limit = MAX_CONVERSATION_TURNS * 2
    conn = get_connection()
    rows = conn.execute("""
        SELECT role, content, timestamp FROM session_messages
        WHERE session_name=? ORDER BY id DESC LIMIT ?
    """, (session_name, limit)).fetchall()
    return [{"role": r[0], "content": r[1], "time": r[2]} for r in reversed(rows)]


def clear_session_messages(session_name):
    conn = get_connection()
    conn.execute("DELETE FROM session_messages WHERE session_name=?", (session_name,))
    conn.commit()


def delete_session_data(session_name):
    conn = get_connection()
    conn.execute("DELETE FROM session_messages WHERE session_name=?", (session_name,))
    conn.commit()


def list_session_names():
    conn = get_connection()
    rows = conn.execute("""
        SELECT session_name, COUNT(*), MAX(timestamp) FROM session_messages
        GROUP BY session_name ORDER BY MAX(timestamp) DESC
    """).fetchall()
    return [{"name": r[0], "count": r[1], "last_active": r[2]} for r in rows]


def purge_old_session_messages():
    """Keep only the last MAX_CONVERSATION_TURNS*2 messages per session."""
    conn = get_connection()
    sessions = conn.execute("SELECT DISTINCT session_name FROM session_messages").fetchall()
    limit = MAX_CONVERSATION_TURNS * 2
    for (name,) in sessions:
        conn.execute("""
            DELETE FROM session_messages WHERE session_name=? AND id NOT IN (
                SELECT id FROM session_messages WHERE session_name=? ORDER BY id DESC LIMIT ?
            )
        """, (name, name, limit))
    conn.commit()


def export_session_messages(session_name):
    conn = get_connection()
    rows = conn.execute("""
        SELECT role, content, timestamp FROM session_messages
        WHERE session_name=? ORDER BY id
    """, (session_name,)).fetchall()
    return [{"role": r[0], "content": r[1], "time": r[2]} for r in rows]


# ==============================
# RESOURCE CHECKS
# ==============================
def _check_db_size():
    try:
        s = os.path.getsize(DB_PATH) / (1024 * 1024)
        if s >= MAX_DB_SIZE_MB:
            return False
        if s >= MAX_DB_SIZE_MB * 0.85:
            log.warning(f"DB approaching limit: {s:.1f}MB")
    except FileNotFoundError:
        pass
    return True


def _check_memory_count(project):
    return get_memory_count(project) < MAX_MEMORIES_PER_PROJECT


# ==============================
# DEDUPLICATION
# ==============================
def _find_duplicate(emb, project):
    if not DEDUP_ENABLED:
        return None, 0
    if _faiss_available:
        entry = _get_faiss_index(project)
        if entry and entry[0].ntotal > 0:
            sims, idxs = entry[0].search(emb.reshape(1, -1), 1)
            if idxs[0][0] >= 0 and sims[0][0] >= DEDUP_THRESHOLD:
                return entry[1][idxs[0][0]], float(sims[0][0])
        return None, 0
    # Numpy fallback
    conn = get_connection()
    rows = conn.execute("SELECT id, embedding FROM memories WHERE project=? ORDER BY timestamp DESC LIMIT 500",
                        (project,)).fetchall()
    for mid, blob in rows:
        e = np.frombuffer(blob, dtype=np.float32)
        if e.shape[0] != EMBEDDING_DIM:
            continue
        n = np.linalg.norm(e)
        if n == 0:
            continue
        if float(np.dot(emb, e / n)) >= DEDUP_THRESHOLD:
            return mid, float(np.dot(emb, e / n))
    return None, 0


# ==============================
# ADD MEMORY
# ==============================
def add_memory(content, project="default", importance=1.0, cluster="general",
               locked=0, memory_type=None):
    if not content or not content.strip():
        return None
    if not _check_db_size() or not _check_memory_count(project):
        return None
    emb = encode_text(content)
    if emb.shape[0] != EMBEDDING_DIM:
        return None

    dup_id, _ = _find_duplicate(emb, project)
    if dup_id is not None:
        conn = get_connection()
        conn.execute("UPDATE memories SET importance=MIN(importance+0.5,5.0), last_access=?, access_count=access_count+1 WHERE id=?",
                     (time.time(), dup_id))
        conn.commit()
        return dup_id

    if memory_type is None:
        memory_type = classify_memory_type(content)
    if memory_type not in VALID_TYPES:
        memory_type = "general"

    # Invalidate FAISS BEFORE insert — any concurrent search rebuilds from DB
    _invalidate_faiss(project)

    now = time.time()
    conn = get_connection()
    cursor = conn.execute("""
        INSERT INTO memories (project, cluster, content, embedding, importance,
            access_count, last_access, timestamp, locked, memory_type, pinned)
        VALUES (?,?,?,?,?,1,?,?,?,?,0)
    """, (project, cluster, content, emb.tobytes(), importance, now, now, locked, memory_type))
    conn.commit()
    return cursor.lastrowid


# ==============================
# CLUSTERING
# ==============================
def auto_cluster(text):
    kw = {
        "cyber": ["exploit","vulnerability","attack","payload","injection","malware","firewall","encryption"],
        "system": ["linux","kernel","process","cpu","disk","filesystem","systemd","cron","bash"],
        "ai": ["model","prompt","llm","embedding","token","neural","training","inference","transformer"],
        "network": ["tcp","udp","port","packet","dns","http","socket","proxy","vpn","routing"],
        "code": ["function","class","variable","loop","python","compile","debug","git","api","database"],
    }
    t = text.lower()
    scores = {c: sum(1 for k in keys if k in t) for c, keys in kw.items()}
    scores = {c: s for c, s in scores.items() if s > 0}
    return max(scores, key=scores.get) if scores else "general"


# ==============================
# SEARCH — relevance filtering + type-aware + pinned injection
# ==============================
def search(query_text, project="default", top_k=SEARCH_TOP_K, prefer_types=None):
    q_emb = encode_text(query_text)
    if _faiss_available:
        results = _search_faiss(q_emb, project, top_k, prefer_types)
    else:
        results = _search_numpy(q_emb, project, top_k, prefer_types)

    # Inject pinned memories at front
    pinned = get_connection().execute(
        "SELECT id, content, cluster, memory_type FROM memories WHERE project=? AND pinned=1",
        (project,)).fetchall()
    pinned_set = set()
    merged = []
    tc = 0
    for _, content, cl, mt in pinned:
        if content not in pinned_set and tc + len(content) <= MAX_CONTEXT_CHARS:
            merged.append((content, cl, mt or "general"))
            pinned_set.add(content)
            tc += len(content)
    for content, cl, mt in results:
        if content not in pinned_set and tc + len(content) <= MAX_CONTEXT_CHARS:
            merged.append((content, cl, mt))
            tc += len(content)
    return merged


def _score(sim, imp, ac, mt, prefer):
    base = sim * imp * (1 + ac * 0.1)
    bonus = 1.0
    if mt == "preference": bonus = 1.8
    elif prefer and mt in prefer: bonus = 1.3
    elif mt == "fact": bonus = 1.1
    return base * bonus


def _search_faiss(q_emb, project, top_k, prefer):
    entry = _get_faiss_index(project)
    if not entry or entry[0].ntotal == 0:
        return []
    index, ids = entry
    k = min(top_k * 4, index.ntotal)

    with _faiss_lock:
        sims, idxs = index.search(q_emb.reshape(1, -1), k)

    conn = get_connection()
    now = time.time()
    cands = []
    for rank in range(idxs.shape[1]):
        fidx = idxs[0][rank]
        if fidx < 0 or fidx >= len(ids):
            continue
        sim = float(sims[0][rank])
        if RELEVANCE_FILTER_ENABLED and sim < SIMILARITY_THRESHOLD:
            continue
        mid = ids[fidx]
        row = conn.execute("SELECT content, importance, access_count, cluster, memory_type FROM memories WHERE id=?",
                           (mid,)).fetchone()
        if not row:
            continue
        c, imp, ac, cl, mt = row
        cands.append((_score(sim, imp, ac, mt or "general", prefer), mid, c, cl, mt or "general"))
    cands.sort(reverse=True)
    top = cands[:top_k]
    for _, mid, _, _, _ in top:
        conn.execute("UPDATE memories SET access_count=access_count+1, last_access=? WHERE id=?", (now, mid))
    conn.commit()
    results = []
    tc = 0
    for _, _, c, cl, mt in top:
        if tc + len(c) > MAX_CONTEXT_CHARS:
            break
        results.append((c, cl, mt))
        tc += len(c)
    return results


def _search_numpy(q_emb, project, top_k, prefer):
    conn = get_connection()
    rows = conn.execute("SELECT id, content, embedding, importance, access_count, cluster, memory_type FROM memories WHERE project=?",
                        (project,)).fetchall()
    now = time.time()
    cands = []
    for mid, c, blob, imp, ac, cl, mt in rows:
        e = np.frombuffer(blob, dtype=np.float32)
        if e.shape[0] != EMBEDDING_DIM:
            continue
        n = np.linalg.norm(e)
        if n == 0:
            continue
        sim = float(np.dot(q_emb, e / n))
        if RELEVANCE_FILTER_ENABLED and sim < SIMILARITY_THRESHOLD:
            continue
        cands.append((_score(sim, imp, ac, mt or "general", prefer), mid, c, cl, mt or "general"))
    cands.sort(reverse=True)
    top = cands[:top_k]
    for _, mid, _, _, _ in top:
        conn.execute("UPDATE memories SET access_count=access_count+1, last_access=? WHERE id=?", (now, mid))
    conn.commit()
    results = []
    tc = 0
    for _, _, c, cl, mt in top:
        if tc + len(c) > MAX_CONTEXT_CHARS:
            break
        results.append((c, cl, mt))
        tc += len(c)
    return results


# ==============================
# PINNED MEMORIES
# ==============================
def pin_memory(mid):
    conn = get_connection()
    if not conn.execute("SELECT id FROM memories WHERE id=?", (mid,)).fetchone():
        return False, "Not found"
    conn.execute("UPDATE memories SET pinned=1 WHERE id=?", (mid,))
    conn.commit()
    return True, "pinned"


def unpin_memory(mid):
    conn = get_connection()
    if not conn.execute("SELECT id FROM memories WHERE id=?", (mid,)).fetchone():
        return False, "Not found"
    conn.execute("UPDATE memories SET pinned=0 WHERE id=?", (mid,))
    conn.commit()
    return True, "unpinned"


# ==============================
# RELATIONSHIPS
# ==============================
def add_relationship(id_a, id_b, rel_type="related"):
    a, b = min(id_a, id_b), max(id_a, id_b)
    conn = get_connection()
    try:
        conn.execute("INSERT OR IGNORE INTO relationships (memory_id_a, memory_id_b, rel_type, created_at) VALUES (?,?,?,?)",
                     (a, b, rel_type, time.time()))
        conn.commit()
        return True
    except Exception:
        return False


def get_related(mid, limit=10):
    conn = get_connection()
    return conn.execute("""
        SELECT m.id, m.content, m.cluster, r.rel_type FROM relationships r
        JOIN memories m ON m.id = CASE WHEN r.memory_id_a=? THEN r.memory_id_b ELSE r.memory_id_a END
        WHERE r.memory_id_a=? OR r.memory_id_b=? LIMIT ?
    """, (mid, mid, mid, limit)).fetchall()


# ==============================
# PREFERENCES (with status: active/blocked)
# ==============================
def set_preference(key, value, source="auto"):
    conn = get_connection()
    # If source is auto and the key is blocked, skip
    if source == "auto":
        row = conn.execute("SELECT status FROM preferences WHERE key=?", (key,)).fetchone()
        if row and row[0] == "blocked":
            return
    conn.execute("""
        INSERT INTO preferences (key, value, source, status, updated_at) VALUES (?,?,?,'active',?)
        ON CONFLICT(key) DO UPDATE SET value=?, source=?, status='active', updated_at=?
    """, (key, value, source, time.time(), value, source, time.time()))
    conn.commit()


def delete_preference(key):
    """Block a preference — won't be re-extracted by auto."""
    conn = get_connection()
    row = conn.execute("SELECT id FROM preferences WHERE key=?", (key,)).fetchone()
    if not row:
        return False, "Not found"
    conn.execute("UPDATE preferences SET status='blocked' WHERE key=?", (key,))
    conn.commit()
    return True, "blocked"


def get_all_preferences():
    conn = get_connection()
    return [{"key": r[0], "value": r[1], "source": r[2]}
            for r in conn.execute("SELECT key, value, source FROM preferences WHERE status='active' ORDER BY key").fetchall()]


def get_preferences_context():
    if not PREFERENCES_ENABLED:
        return ""
    prefs = get_all_preferences()
    if not prefs:
        return ""
    lines = ["USER PREFERENCES (always respect these):"]
    lines.extend(f"- {p['key']}: {p['value']}" for p in prefs)
    return "\n".join(lines)


# ==============================
# COMPRESSION — correct project propagation
# ==============================
def compress_cluster(cluster_name):
    conn = get_connection()
    now = time.time()
    rows = conn.execute("SELECT id, content, importance, last_access, project FROM memories WHERE cluster=? AND locked=0 AND pinned=0",
                        (cluster_name,)).fetchall()
    if len(rows) < COMPRESS_CLUSTER_THRESHOLD:
        return

    def sc(r):
        return r[2] / (1.0 + (now - r[3]) / 86400.0)

    sr = sorted(rows, key=sc, reverse=True)
    to_compress = sr[COMPRESS_KEEP_TOP_N:]
    if not to_compress:
        return

    by_proj = {}
    for r in to_compress:
        by_proj.setdefault(r[4], []).append(r)

    for proj, group in by_proj.items():
        bullets = [f"- {r[1].split(chr(10))[0].strip()[:200]}" for r in group if r[1].strip()][:20]
        summary = f"[COMPRESSED:{cluster_name}] {len(group)} memories:\n" + "\n".join(bullets)
        for r in group:
            conn.execute("DELETE FROM memories WHERE id=?", (r[0],))
        _invalidate_faiss(proj)
        conn.commit()
        add_memory(summary, project=proj, cluster=cluster_name, importance=2.0, memory_type="compressed")


# ==============================
# KNOWLEDGE REVIEW
# ==============================
def knowledge_review():
    conn = get_connection()
    rows = conn.execute("""
        SELECT id, content, importance, last_access, access_count, project FROM memories
        WHERE locked=0 AND pinned=0 AND memory_type NOT IN ('preference','compressed')
        ORDER BY last_access ASC LIMIT ?
    """, (KNOWLEDGE_REVIEW_BATCH_SIZE,)).fetchall()
    now = time.time()
    flagged = []
    for mid, content, imp, la, ac, proj in rows:
        days = (now - la) / 86400.0
        if days > 90 and imp <= 1.0 and ac <= 1:
            preview = content[:80].replace("\n", " ")
            if KNOWLEDGE_REVIEW_DELETE:
                add_notification(f"Auto-deleting stale memory {mid}: {preview}", "warning", "memory")
                conn.execute("DELETE FROM memories WHERE id=?", (mid,))
                _invalidate_faiss(proj)
                flagged.append(("deleted", mid))
            else:
                add_notification(f"Stale memory {mid} ({days:.0f}d, imp={imp:.1f}, used={ac}x): {preview}", "info", "review")
                flagged.append(("flagged", mid))
        elif days > 30 and imp <= 1.5 and ac <= 2:
            conn.execute("UPDATE memories SET importance=? WHERE id=?", (max(imp * 0.85, 0.3), mid))
            flagged.append(("decayed", mid))
    conn.commit()
    return flagged


# ==============================
# REINDEX — batched to avoid holding write lock
# ==============================
def reindex_all():
    conn = get_connection()
    rows = conn.execute("SELECT id, content FROM memories").fetchall()
    count = 0
    batch = []
    for mid, content in rows:
        emb = encode_text(content)
        if emb.shape[0] != EMBEDDING_DIM:
            continue
        batch.append((emb.tobytes(), mid))
        count += 1
        if len(batch) >= REINDEX_BATCH_SIZE:
            for b_emb, b_id in batch:
                conn.execute("UPDATE memories SET embedding=? WHERE id=?", (b_emb, b_id))
            conn.commit()
            batch = []
    if batch:
        for b_emb, b_id in batch:
            conn.execute("UPDATE memories SET embedding=? WHERE id=?", (b_emb, b_id))
        conn.commit()
    with _faiss_lock:
        _faiss_indexes.clear()
        _faiss_generations.clear()
    log.info(f"Re-indexed {count} memories in batches of {REINDEX_BATCH_SIZE}")
    return count


# ==============================
# DELETE / UNDO (soft-delete via deleted_memories table)
# ==============================
def delete_memory(mid):
    conn = get_connection()
    row = conn.execute("SELECT * FROM memories WHERE id=?", (mid,)).fetchone()
    if not row:
        return False, "Not found"
    if row[9] == 1:  # locked column
        return False, "Locked"
    # Copy to deleted_memories
    conn.execute("""
        INSERT INTO deleted_memories (orig_id, project, cluster, content, embedding,
            importance, access_count, last_access, timestamp, locked, memory_type, pinned, deleted_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9], row[10], row[11], time.time()))
    conn.execute("DELETE FROM memories WHERE id=?", (mid,))
    conn.commit()
    _invalidate_faiss(row[1])  # project
    return True, "deleted"


def undo_delete():
    """Restore the most recently deleted memory."""
    conn = get_connection()
    row = conn.execute("SELECT * FROM deleted_memories ORDER BY deleted_at DESC LIMIT 1").fetchone()
    if not row:
        return False, "Nothing to undo"
    # row: id, orig_id, project, cluster, content, embedding, importance, access_count,
    #      last_access, timestamp, locked, memory_type, pinned, deleted_at
    conn.execute("""
        INSERT INTO memories (project, cluster, content, embedding, importance,
            access_count, last_access, timestamp, locked, memory_type, pinned)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
    """, (row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9], row[10], row[11], row[12]))
    conn.execute("DELETE FROM deleted_memories WHERE id=?", (row[0],))
    conn.commit()
    _invalidate_faiss(row[2])
    return True, f"Restored memory (was id {row[1]})"


def purge_deleted():
    cutoff = time.time() - UNDO_RETENTION_HOURS * 3600
    conn = get_connection()
    conn.execute("DELETE FROM deleted_memories WHERE deleted_at < ?", (cutoff,))
    conn.commit()


# ==============================
# UPDATE / LOCK / UNLOCK / FIND / VIEW / LIST
# ==============================
def update_memory(mid, new_content):
    if not new_content or not new_content.strip():
        return False, "Empty"
    conn = get_connection()
    row = conn.execute("SELECT locked, project FROM memories WHERE id=?", (mid,)).fetchone()
    if not row:
        return False, "Not found"
    if row[0] == 1:
        return False, "Locked"
    emb = encode_text(new_content)
    conn.execute("UPDATE memories SET content=?, embedding=?, last_access=? WHERE id=?",
                 (new_content, emb.tobytes(), time.time(), mid))
    conn.commit()
    _invalidate_faiss(row[1])
    return True, "updated"


def lock_memory(mid):
    conn = get_connection()
    if not conn.execute("SELECT id FROM memories WHERE id=?", (mid,)).fetchone():
        return False, "Not found"
    conn.execute("UPDATE memories SET locked=1 WHERE id=?", (mid,))
    conn.commit()
    return True, "locked"


def unlock_memory(mid):
    conn = get_connection()
    if not conn.execute("SELECT id FROM memories WHERE id=?", (mid,)).fetchone():
        return False, "Not found"
    conn.execute("UPDATE memories SET locked=0 WHERE id=?", (mid,))
    conn.commit()
    return True, "unlocked"


def find_memories(keyword, project="default"):
    conn = get_connection()
    return conn.execute("SELECT id, content, cluster, importance FROM memories WHERE project=? AND content LIKE ? ORDER BY importance DESC LIMIT 10",
                        (project, f"%{keyword}%")).fetchall()


def get_memory_by_id(mid):
    conn = get_connection()
    return conn.execute("SELECT id, project, cluster, content, importance, access_count, last_access, timestamp, locked, memory_type, pinned FROM memories WHERE id=?",
                        (mid,)).fetchone()


def list_memories(project="default", cluster=None, memory_type=None, page=0, per_page=20):
    conn = get_connection()
    q = "SELECT id, content, cluster, importance, memory_type, pinned FROM memories WHERE project=?"
    p = [project]
    if cluster:
        q += " AND cluster=?"; p.append(cluster)
    if memory_type:
        q += " AND memory_type=?"; p.append(memory_type)
    q += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
    p.extend([per_page, page * per_page])
    return conn.execute(q, p).fetchall()


# ==============================
# STATS / PROJECTS
# ==============================
def get_memory_count(project="default"):
    return get_connection().execute("SELECT COUNT(*) FROM memories WHERE project=?", (project,)).fetchone()[0]


def get_db_stats():
    try:
        conn = get_connection()
        total = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        locked = conn.execute("SELECT COUNT(*) FROM memories WHERE locked=1").fetchone()[0]
        pinned = conn.execute("SELECT COUNT(*) FROM memories WHERE pinned=1").fetchone()[0]
        clusters = conn.execute("SELECT COUNT(DISTINCT cluster) FROM memories").fetchone()[0]
        types = dict(conn.execute("SELECT memory_type, COUNT(*) FROM memories GROUP BY memory_type").fetchall())
        prefs = conn.execute("SELECT COUNT(*) FROM preferences WHERE status='active'").fetchone()[0]
        rels = conn.execute("SELECT COUNT(*) FROM relationships").fetchone()[0]
        deleted = conn.execute("SELECT COUNT(*) FROM deleted_memories").fetchone()[0]
        size = os.path.getsize(DB_PATH) / (1024 * 1024) if os.path.exists(DB_PATH) else 0.0
        return {"total": total, "locked": locked, "pinned": pinned, "cluster_count": clusters,
                "size_mb": size, "faiss": "active" if _faiss_available else "numpy",
                "types": types, "preferences": prefs, "relationships": rels, "undo_available": deleted}
    except Exception as e:
        log.error(f"Stats error: {e}")
        return {"total": 0, "locked": 0, "pinned": 0, "cluster_count": 0, "size_mb": 0,
                "faiss": "error", "types": {}, "preferences": 0, "relationships": 0, "undo_available": 0}


def get_projects():
    return [{"name": r[0], "count": r[1]}
            for r in get_connection().execute("SELECT project, COUNT(*) FROM memories GROUP BY project ORDER BY COUNT(*) DESC").fetchall()]


# ==============================
# EXPORT / IMPORT
# ==============================
def export_memories(project=None):
    conn = get_connection()
    q = "SELECT id, project, cluster, content, importance, access_count, last_access, timestamp, locked, memory_type, pinned FROM memories"
    rows = conn.execute(q + (" WHERE project=? ORDER BY id" if project else " ORDER BY id"),
                        (project,) if project else ()).fetchall()
    records = [{"id": r[0], "project": r[1], "cluster": r[2], "content": r[3], "importance": r[4],
                "access_count": r[5], "last_access": r[6], "timestamp": r[7], "locked": r[8],
                "memory_type": r[9] or "general", "pinned": r[10] or 0} for r in rows]
    prefs = get_all_preferences()
    rels = [{"a": r[0], "b": r[1], "type": r[2]}
            for r in conn.execute("SELECT memory_id_a, memory_id_b, rel_type FROM relationships").fetchall()]

    os.makedirs(EXPORT_DIR, exist_ok=True)
    fp = os.path.join(EXPORT_DIR, f"export_{project or 'all'}_{time.strftime('%Y%m%d_%H%M%S')}.json")
    resolved = os.path.realpath(os.path.abspath(fp))
    if not resolved.startswith(CONTAINER_ROOT):
        return False, "Security: outside container"
    with open(fp, "w", encoding="utf-8") as f:
        json.dump({"count": len(records), "memories": records, "preferences": prefs, "relationships": rels},
                  f, indent=2, ensure_ascii=False)
    return True, fp


def import_memories(filepath, target_project=None):
    resolved = os.path.realpath(os.path.abspath(filepath))
    if not resolved.startswith(CONTAINER_ROOT):
        return False, "Security: outside container"
    if not os.path.isfile(resolved):
        return False, "Not found"
    try:
        with open(resolved, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return False, str(e)

    imported = skipped = 0
    for rec in data.get("memories", []):
        content = rec.get("content", "").strip()
        if not content:
            skipped += 1; continue
        proj = target_project or rec.get("project", "default")
        if not _check_memory_count(proj):
            skipped += 1; continue
        add_memory(content, project=proj, importance=rec.get("importance", 1.0),
                   cluster=rec.get("cluster", "general"), locked=rec.get("locked", 0),
                   memory_type=rec.get("memory_type"))
        imported += 1

    for p in data.get("preferences", []):
        if p.get("key") and p.get("value"):
            set_preference(p["key"], p["value"], source="import")
    for r in data.get("relationships", []):
        add_relationship(r.get("a", 0), r.get("b", 0), r.get("type", "related"))
    return True, f"Imported {imported}, skipped {skipped}"


# ==============================
# BACKGROUND MAINTENANCE
# ==============================
_stop_event = threading.Event()
_last_review = 0
_dirty_clusters = set()
_dirty_lock = threading.Lock()


def _mark_dirty(cluster):
    with _dirty_lock:
        _dirty_clusters.add(cluster)


def background():
    global _last_review
    try:
        with _dirty_lock:
            dirty = _dirty_clusters.copy()
            _dirty_clusters.clear()
        if dirty:
            for cl in dirty:
                compress_cluster(cl)

        now = time.time()
        if now - _last_review > KNOWLEDGE_REVIEW_INTERVAL_HOURS * 3600:
            knowledge_review()
            _last_review = now

        purge_old_notifications()
        purge_deleted()
        purge_old_session_messages()
    except Exception as e:
        log.error(f"Background error: {e}")


schedule.every(BACKGROUND_INTERVAL_MINUTES).minutes.do(background)


def start_background():
    def loop():
        while not _stop_event.is_set():
            schedule.run_pending()
            _stop_event.wait(timeout=10)
    threading.Thread(target=loop, daemon=True).start()


def stop_background():
    _stop_event.set()


def startup():
    init()
    start_background()
