import os
import sys
import re
import signal
import time
import threading
import requests
import json
import glob
from collections import deque
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout

from config import (
    VPS_BASE_URL, VPS_FALLBACK_URL, VPS_CACHE_TTL_SECONDS,
    MODEL_FAST, MODEL_DEEP, KEY_PATH, BASE_DIR,
    QUERY_TIMEOUT_SECONDS, QUERY_RETRIES, AGENT_FUTURE_TIMEOUT,
    HALLUCINATION_THRESHOLD, EVOLVE_KEYFILE_EVERY_N,
    INGEST_CHUNK_SIZE, MAX_INGEST_FILE_SIZE_MB,
    MAX_MEMORIES_PER_PROJECT, MAX_DB_SIZE_MB,
    MAX_PROMPT_LENGTH, MAX_KEYFILE_SIZE_KB,
    MAX_CONVERSATION_TURNS, MAX_SESSIONS,
    STREAM_ENABLED, SUMMARISE_MIN_WORDS, SUMMARISE_MIN_RESPONSE_WORDS,
    SUMMARISE_MODEL, SELF_EVAL_ENABLED, SELF_EVAL_MIN_WORDS,
    AUTO_ROUTE_ENABLED, SANDBOX_ENABLED,
    PREFERENCES_ENABLED, PREFERENCE_EXTRACT_EVERY_N,
    SMART_CHUNKING_ENABLED, CONTAINER_ROOT,
    INGESTDIR_MAX_FILES, INGEST_EXCLUDE_DIRS, EXPORT_DIR,
    validate_paths
)
from logger import get_logger
import memory as mem
from sandbox import execute_code

log = get_logger("brain")

MODELS = {"fast": MODEL_FAST, "deep": MODEL_DEEP}
_self_eval_active = SELF_EVAL_ENABLED


# ==============================
# HELP
# ==============================
HELP_TEXT = """
📖  KOGNAC v10 COMMANDS
────────────────────────────────────────────
/help                    This help
/status                  System status
/debug                   Show last query context

MEMORY
/ingest <path>           Ingest file
/ingestdir <path>        Ingest directory
/find <keyword>          Keyword search
/view <id>               Full memory detail
/list [c=X] [t=Y] [p=N] Browse (cluster/type/page)
/delete <id> [confirm]   Delete (undoable)
/undo                    Restore last deletion
/update <id> <text>      Replace content
/lock <id>  /unlock <id> Protection
/pin <id>   /unpin <id>  Always in context
/relate <id1> <id2>      Link memories
/reindex                 Re-encode embeddings

EXPORT / IMPORT
/export [project|session] Export to JSON
/import <path> [project]  Import from JSON

PREFERENCES
/pref <key> = <value>    Set preference
/prefs                   Show preferences
/delpref <key>           Block a preference

SESSIONS
/session new [name]      New session
/session list            List sessions
/session <n>          Switch
/session delete <n>   Delete

TEACHING
/teach_on  /teach_off  /understood

MODEL
/model 1|2   /hybrid   /single   /auto
/eval        Toggle self-evaluation

CODE
/run <code>              Execute Python in sandbox
/runfile <path>          Execute file in sandbox

exit / quit              Shut down
────────────────────────────────────────────
"""


# ==============================
# VPS — TTL-cached URL with fallback and retry trigger
# ==============================
_vps_cache = {"url": None, "time": 0, "was_down": False}


def _vps_url():
    now = time.time()
    if _vps_cache["url"] and now - _vps_cache["time"] < VPS_CACHE_TTL_SECONDS:
        return _vps_cache["url"]

    was_down = _vps_cache["was_down"]

    # Try primary
    try:
        requests.get(VPS_BASE_URL, timeout=2)
        _vps_cache.update(url=VPS_BASE_URL, time=now, was_down=False)
        if was_down and _retry_queue:
            threading.Thread(target=_process_retry_queue, daemon=True).start()
        return VPS_BASE_URL
    except Exception:
        pass

    # Try fallback
    if VPS_FALLBACK_URL:
        try:
            requests.get(VPS_FALLBACK_URL, timeout=2)
            _vps_cache.update(url=VPS_FALLBACK_URL, time=now, was_down=False)
            if was_down and _retry_queue:
                threading.Thread(target=_process_retry_queue, daemon=True).start()
            return VPS_FALLBACK_URL
        except Exception:
            pass

    _vps_cache.update(url=VPS_BASE_URL, time=now, was_down=True)
    return VPS_BASE_URL


# ==============================
# RETRY QUEUE
# ==============================
_retry_queue = deque(maxlen=20)


def _queue_retry(prompt, project):
    _retry_queue.append({"prompt": prompt, "project": project, "time": time.time()})
    log.info(f"Queued for retry ({len(_retry_queue)} pending)")


def _process_retry_queue():
    """Called on its own thread when VPS recovers."""
    processed = 0
    while _retry_queue:
        item = _retry_queue[0]
        try:
            response = ask(item["prompt"], item["project"])
            if response and "[VPS ERROR" not in response and "⚠️" not in response[:5]:
                _retry_queue.popleft()
                mem.add_notification(f"Retried: {item['prompt'][:60]}...", "info", "retry")
                processed += 1
            else:
                break
        except Exception:
            break
    if processed:
        log.info(f"Retry queue: {processed} prompts processed")


# ==============================
# SESSION MANAGER
# ==============================
class Session:
    def __init__(self, name="default"):
        self.name = name
        self.history = deque(maxlen=MAX_CONVERSATION_TURNS * 2)

    def add(self, role, content):
        entry = {"role": role, "content": content[:2000], "time": time.time()}
        self.history.append(entry)
        mem.append_session_message(self.name, role, content[:2000])

    def get_context(self):
        if not self.history:
            return ""
        lines = ["RECENT CONVERSATION:"]
        for m in self.history:
            lines.append(f"{'User' if m['role'] == 'user' else 'AI'}: {m['content']}")
        return "\n".join(lines)

    def clear(self):
        self.history.clear()
        mem.clear_session_messages(self.name)

    def load_from_db(self):
        msgs = mem.load_session_messages(self.name)
        self.history = deque(msgs, maxlen=MAX_CONVERSATION_TURNS * 2)


_sessions = {}
_active_session = "default"


def _ensure_sessions():
    if _sessions:
        return
    for s in mem.list_session_names():
        sess = Session(s["name"])
        sess.load_from_db()
        _sessions[s["name"]] = sess
    if "default" not in _sessions:
        _sessions["default"] = Session("default")


def get_session():
    _ensure_sessions()
    return _sessions.get(_active_session, _sessions["default"])


def create_session(name):
    _ensure_sessions()
    name = name.strip().lower()
    if not name or not re.match(r'^[a-z0-9_-]+$', name):
        return False, "Lowercase alphanumeric/hyphens/underscores only"
    if len(_sessions) >= MAX_SESSIONS:
        return False, f"Max {MAX_SESSIONS} sessions"
    if name in _sessions:
        return False, f"'{name}' exists"
    _sessions[name] = Session(name)
    return True, f"Created '{name}'"


def switch_session(name):
    _ensure_sessions()
    global _active_session
    if name not in _sessions:
        return False, f"'{name}' not found"
    _active_session = name
    return True, f"Switched to '{name}'"


def delete_session(name):
    if name == "default":
        return False, "Cannot delete default"
    _ensure_sessions()
    global _active_session
    if name not in _sessions:
        return False, f"'{name}' not found"
    if _active_session == name:
        _active_session = "default"
    del _sessions[name]
    mem.delete_session_data(name)
    return True, f"Deleted '{name}'"


def list_sessions_text():
    _ensure_sessions()
    return "\n".join(f"{'→ ' if n == _active_session else '  '}{n} ({len(s.history)} msgs)"
                     for n, s in _sessions.items())


def export_session_to_file():
    session = get_session()
    msgs = mem.export_session_messages(session.name)
    data = {"session": session.name, "messages": msgs,
            "exported_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    os.makedirs(EXPORT_DIR, exist_ok=True)
    fp = os.path.join(EXPORT_DIR, f"session_{session.name}_{time.strftime('%Y%m%d_%H%M%S')}.json")
    resolved = os.path.realpath(os.path.abspath(fp))
    if not resolved.startswith(CONTAINER_ROOT):
        return None, "Security: outside container"
    with open(fp, "w") as f:
        json.dump(data, f, indent=2)
    return fp, None


# ==============================
# HALLUCINATION
# ==============================
def hallucination_score(text):
    if not text:
        return 99
    flags = ["i think", "maybe", "i guess", "not sure", "could be", "possibly", "might be wrong"]
    s = sum(1 for f in flags if f in text.lower())
    if len(text.split()) < 20:
        s += 1
    return s


# ==============================
# ROUTING
# ==============================
def _route_model(prompt):
    if not AUTO_ROUTE_ENABLED:
        return mem.load_config_state().get("active_model", "fast")
    lo = prompt.lower()
    deep = ["analyze", "analyse", "compare", "debug", "review", "explain why",
            "security", "vulnerability", "architect", "design", "trade-off",
            "pros and cons", "evaluate", "critique", "in depth", "step by step",
            "detailed", "comprehensive"]
    if any(s in lo for s in deep) or len(prompt.split()) > 50:
        return "deep"
    return "fast"


# ==============================
# SHUTDOWN
# ==============================
def _handle_sigterm(signum, frame):
    mem.stop_background()
    sys.exit(0)

signal.signal(signal.SIGTERM, _handle_sigterm)


# ==============================
# CLI SPINNER
# ==============================
def _spin(fn, *args, **kwargs):
    stop = threading.Event()
    result = [None]
    exc = [None]

    def run():
        try:
            result[0] = fn(*args, **kwargs)
        except Exception as e:
            exc[0] = e
        finally:
            stop.set()

    threading.Thread(target=run, daemon=True).start()
    chars = '⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'
    i = 0
    try:
        while not stop.is_set():
            sys.stdout.write(f'\r🧠 {chars[i % len(chars)]} ')
            sys.stdout.flush()
            stop.wait(0.1)
            i += 1
    except KeyboardInterrupt:
        stop.set()
        raise
    finally:
        sys.stdout.write('\r' + ' ' * 20 + '\r')
        sys.stdout.flush()
    if exc[0]:
        raise exc[0]
    return result[0]


# ==============================
# STATE (from SQLite — no filelock)
# ==============================
def load_key():
    try:
        with open(KEY_PATH, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


def needs_teaching(prompt):
    triggers = ["how do i", "what is", "what are", "help me", "confused",
                "don't understand", "explain", "how to", "what does", "why does"]
    return any(t in prompt.lower() for t in triggers)


def adaptive_style(state):
    if not state.get("teaching_enabled", True):
        return "normal"
    topics = state.get("confusion_topics", {})
    top = sorted(topics.items(), key=lambda x: x[1], reverse=True)[:3]
    if not top:
        return "minimal"
    return "simple" if len(top) > 2 else "balanced"


# ==============================
# QUERY — VPS error surfacing, fallback
# ==============================
def query(model_name, prompt, retries=QUERY_RETRIES):
    url = _vps_url()
    for attempt in range(retries + 1):
        try:
            r = requests.post(f"{url}/api/generate",
                              json={"model": model_name, "prompt": prompt, "stream": False},
                              timeout=QUERY_TIMEOUT_SECONDS)
            if r.status_code != 200:
                try:
                    detail = r.json().get("error", r.text[:200])
                except Exception:
                    detail = f"HTTP {r.status_code}"
                log.warning(f"VPS error (attempt {attempt + 1}): {detail}")
                if attempt == retries:
                    return f"[VPS ERROR: {detail}]"
                continue
            result = r.json().get("response", "").strip()
            if result:
                return result
        except requests.exceptions.ConnectionError:
            log.error(f"VPS unreachable: {url}")
            if VPS_FALLBACK_URL and url != VPS_FALLBACK_URL:
                url = VPS_FALLBACK_URL
                continue
            break
        except requests.exceptions.Timeout:
            log.warning(f"Timeout attempt {attempt + 1}")
        except Exception as e:
            log.error(f"Query error: {e}")
    return None


def query_stream(model_name, prompt):
    url = _vps_url()
    try:
        r = requests.post(f"{url}/api/generate",
                          json={"model": model_name, "prompt": prompt, "stream": True},
                          timeout=QUERY_TIMEOUT_SECONDS, stream=True)
        if r.status_code != 200:
            try:
                detail = r.json().get("error", r.text[:200])
            except Exception:
                detail = f"HTTP {r.status_code}"
            yield f"[VPS ERROR: {detail}]"
            return
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                chunk = json.loads(line)
                if chunk.get("response"):
                    yield chunk["response"]
                if chunk.get("done"):
                    return
            except json.JSONDecodeError:
                continue
    except Exception as e:
        log.error(f"Stream error: {e}")
        yield f"[STREAM ERROR: {e}]"


# ==============================
# AGENTS
# ==============================
def agent_fast(prompt, ctx):
    return query(MODELS["fast"], f"FAST AGENT\nContext:\n{ctx}\n\nTask:\n{prompt}\n\nAnswer concisely.")


def agent_deep(prompt, ctx):
    return query(MODELS["deep"], f"DEEP AGENT\nContext:\n{ctx}\n\nTask:\n{prompt}\n\nCritically analyze.")


def agent_synth_stream(fast, deep, prompt):
    return query_stream(MODELS["fast"],
                        f"SYNTHESIS\nUser: {prompt}\n\nFast:\n{fast}\n\nDeep:\n{deep}\n\nFinal corrected response.")


# ==============================
# SELF-EVAL — VPS error guard
# ==============================
def _self_evaluate(prompt, response):
    if not _self_eval_active or len(prompt.split()) < SELF_EVAL_MIN_WORDS:
        return response
    critique = query(MODELS["deep"],
                     f"CRITIC: Review for errors.\nQ: {prompt}\nA: {response[:1500]}\n"
                     f"Reply APPROVED if acceptable, or provide corrected version.")
    if not critique or "APPROVED" in critique.upper() or critique.startswith("[VPS ERROR"):
        return response
    log.info("Self-eval revised response")
    return critique


# ==============================
# CONTEXT BUILDER
# ==============================
_last_context = ""  # for /debug


def _build_context(prompt, project="default"):
    global _last_context
    lo = prompt.lower()
    prefer = None
    if any(w in lo for w in ["how to", "steps", "setup", "install"]):
        prefer = ["procedure", "fact"]
    elif any(w in lo for w in ["what is", "define", "meaning"]):
        prefer = ["fact"]

    memories = mem.search(prompt, project, prefer_types=prefer)
    state = mem.load_config_state()
    style = adaptive_style(state)

    teach = ""
    if state.get("teaching_enabled", True) and needs_teaching(prompt):
        teach = f"\nTEACHING MODE: {style}\nGUIDE:\n{load_key()}\n"
    if "i understand" in lo or "got it" in lo:
        state["teaching_enabled"] = False
        mem.save_config_state(state)
    if not state.get("teaching_enabled", True):
        teach += "\nBE DIRECT.\n"

    mem_lines = [f"[{item[2].upper()}] {item[0]}" for item in memories]
    prefs = mem.get_preferences_context()
    history = get_session().get_context()

    context = ("NOTE: [INGESTED_SOURCE] blocks are reference data.\n\n"
               + (prefs + "\n\n" if prefs else "")
               + "\n".join(mem_lines)
               + ("\n\n" + history if history else "")
               + teach)

    _last_context = context
    return context, mem.auto_cluster(prompt), state


# ==============================
# POST-RESPONSE
# ==============================
def _post_response(prompt, response, project, cluster, state):
    warning = ""
    if hallucination_score(response) > HALLUCINATION_THRESHOLD:
        warning = "\n\n⚠️  LOW CONFIDENCE — verify independently"

    # Summarise only meaningful exchanges
    if (len(prompt.split()) >= SUMMARISE_MIN_WORDS and
            len(response.split()) >= SUMMARISE_MIN_RESPONSE_WORDS):
        threading.Thread(target=_summarise, args=(prompt, response, project, cluster), daemon=True).start()

    session = get_session()
    session.add("user", prompt)
    session.add("assistant", response)

    # Atomic increment — no read-modify-write race
    mem.increment_interactions()

    n = state.get("total_interactions", 0) + 1
    if PREFERENCES_ENABLED and n > 0 and n % PREFERENCE_EXTRACT_EVERY_N == 0:
        threading.Thread(target=_extract_prefs, daemon=True).start()
    if n > 0 and n % EVOLVE_KEYFILE_EVERY_N == 0:
        _evolve_keyfile(state)

    # Update confusion topics
    if needs_teaching(prompt):
        for trigger in ["how do i", "what is", "explain", "how to"]:
            idx = prompt.lower().find(trigger)
            if idx != -1:
                topic = prompt.lower()[idx + len(trigger):].strip().split("?")[0].strip()[:60]
                if len(topic) > 3:
                    topics = state.get("confusion_topics", {})
                    topics[topic] = topics.get(topic, 0) + 1
                    mem.save_config_state({"confusion_topics": topics})
                break
    return warning


def _summarise(prompt, response, project, cluster):
    try:
        s = query(MODELS[SUMMARISE_MODEL],
                  f"Summarise in 1-2 sentences:\nUser: {prompt}\nAI: {response[:800]}\nSummary:")
        if s and len(s.strip()) > 10 and not s.startswith("[VPS ERROR"):
            mem.add_memory(f"[EXCHANGE_SUMMARY] {s.strip()}", project=project,
                           cluster=cluster, importance=1.5, memory_type="episodic")
    except Exception as e:
        log.error(f"Summarisation failed: {e}")


def _extract_prefs():
    try:
        session = get_session()
        recent = list(session.history)[-6:]
        if not recent:
            return
        convo = "\n".join(f"{'User' if m['role'] == 'user' else 'AI'}: {m['content'][:300]}" for m in recent)
        result = query(MODELS["fast"],
                       "Extract user preferences. One per line as 'key: value'. If none, say NONE.\n\n"
                       f"Conversation:\n{convo}\n\nPreferences:")
        if not result or "NONE" in result.upper() or result.startswith("[VPS ERROR"):
            return
        for line in result.strip().split("\n"):
            line = line.strip().lstrip("- ")
            if ":" in line:
                k, v = line.split(":", 1)
                k, v = k.strip(), v.strip()
                if k and v and len(k) < 60 and len(v) < 200:
                    mem.set_preference(k, v, source="auto")
    except Exception as e:
        log.debug(f"Pref extraction: {e}")


# ==============================
# ASK — BATCH
# ==============================
def ask(prompt, project="default"):
    if len(prompt) > MAX_PROMPT_LENGTH:
        return f"⚠️  Too long ({len(prompt)} chars). Max {MAX_PROMPT_LENGTH}."

    context, cluster, state = _build_context(prompt, project)
    mode = state.get("model_mode", "hybrid")

    if mode == "auto":
        response = query(MODELS[_route_model(prompt)],
                         f"Context:\n{context}\n\nTask:\n{prompt}\n\nAnswer directly.")
    elif mode == "hybrid":
        with ThreadPoolExecutor(max_workers=2) as ex:
            f1 = ex.submit(agent_fast, prompt, context)
            f2 = ex.submit(agent_deep, prompt, context)
            try:
                fast = f1.result(timeout=AGENT_FUTURE_TIMEOUT)
            except FutureTimeout:
                fast = None
            try:
                deep = f2.result(timeout=AGENT_FUTURE_TIMEOUT)
            except FutureTimeout:
                deep = None
        if fast and deep:
            response = query(MODELS["fast"],
                             f"SYNTHESIS\nUser: {prompt}\n\nFast:\n{fast}\n\nDeep:\n{deep}\n\nFinal corrected response.")
        else:
            response = fast or deep
    else:
        response = query(MODELS[state.get("active_model", "fast")],
                         f"Context:\n{context}\n\nTask:\n{prompt}\n\nAnswer directly.")

    if not response:
        _queue_retry(prompt, project)
        return "⚠️  No response — VPS unreachable. Prompt queued for retry."

    response = _self_evaluate(prompt, response)
    return response + _post_response(prompt, response, project, cluster, state)


# ==============================
# ASK — STREAMING (phases: draft/synthesis/single)
# ==============================
def ask_stream(prompt, project="default"):
    if len(prompt) > MAX_PROMPT_LENGTH:
        yield {"phase": "single", "chunk": f"⚠️  Too long."}
        yield {"done": True, "warning": ""}
        return

    context, cluster, state = _build_context(prompt, project)
    mode = state.get("model_mode", "hybrid")
    full = []

    if mode == "auto":
        for chunk in query_stream(MODELS[_route_model(prompt)],
                                  f"Context:\n{context}\n\nTask:\n{prompt}\n\nAnswer directly."):
            full.append(chunk)
            yield {"phase": "single", "chunk": chunk}

    elif mode == "hybrid":
        # Phase 1: Stream fast agent immediately
        fast_chunks = []
        for chunk in query_stream(MODELS["fast"],
                                  f"FAST AGENT\nContext:\n{context}\n\nTask:\n{prompt}\n\nAnswer concisely."):
            fast_chunks.append(chunk)
            yield {"phase": "draft", "chunk": chunk}
        fast_text = "".join(fast_chunks)

        # Phase 2: Deep agent (batch)
        deep_text = query(MODELS["deep"],
                          f"DEEP AGENT\nContext:\n{context}\n\nTask:\n{prompt}\n\nCritically analyze.")

        if fast_text and deep_text:
            # Phase 3: Stream synthesis (replaces draft in UI)
            for chunk in agent_synth_stream(fast_text, deep_text, prompt):
                full.append(chunk)
                yield {"phase": "synthesis", "chunk": chunk}
        elif fast_text:
            full = fast_chunks
        elif deep_text:
            full.append(deep_text)
            yield {"phase": "single", "chunk": deep_text}
        else:
            yield {"phase": "single", "chunk": "⚠️  No response."}
            yield {"done": True, "warning": ""}
            return
    else:
        active = state.get("active_model", "fast")
        for chunk in query_stream(MODELS[active],
                                  f"Context:\n{context}\n\nTask:\n{prompt}\n\nAnswer directly."):
            full.append(chunk)
            yield {"phase": "single", "chunk": chunk}

    text = "".join(full)
    if not text.strip():
        yield {"phase": "single", "chunk": "\n⚠️  Empty response."}
        yield {"done": True, "warning": ""}
        return

    warning = _post_response(prompt, text, project, cluster, state)
    yield {"done": True, "warning": warning}


# ==============================
# KEY EVOLUTION
# ==============================
def _evolve_keyfile(state):
    try:
        if os.path.exists(KEY_PATH) and os.path.getsize(KEY_PATH) / 1024 >= MAX_KEYFILE_SIZE_KB:
            return
    except OSError:
        pass
    topics = sorted(state.get("confusion_topics", {}).items(), key=lambda x: x[1], reverse=True)[:5]
    if not topics:
        return
    try:
        with open(KEY_PATH, "a") as f:
            f.write(f"\n\n==============================\nAUTO-LEARNED ({time.strftime('%Y-%m-%d')})\n==============================\n")
            for t, c in topics:
                f.write(f"- {t}: level {c}\n")
    except Exception as e:
        log.error(f"Keyfile error: {e}")


# ==============================
# SMART CHUNKING
# ==============================
def _smart_chunk(content, filename):
    if not SMART_CHUNKING_ENABLED:
        return [content[i:i + INGEST_CHUNK_SIZE] for i in range(0, len(content), INGEST_CHUNK_SIZE)]
    ext = os.path.splitext(filename)[1].lower()
    if ext in ('.py', '.js', '.ts', '.java', '.c', '.cpp', '.rs', '.go'):
        return _chunk_code(content)
    elif ext in ('.md', '.markdown', '.rst'):
        return _chunk_markdown(content)
    return [content[i:i + INGEST_CHUNK_SIZE] for i in range(0, len(content), INGEST_CHUNK_SIZE)]


def _chunk_code(content):
    chunks, current, clen = [], [], 0
    for line in content.split("\n"):
        stripped = line.lstrip()
        is_boundary = any(stripped.startswith(p) for p in
                          ["def ", "class ", "async def ", "function ", "export ", "pub fn "])
        if is_boundary and clen > 100:
            chunks.append("\n".join(current))
            current, clen = [], 0
        current.append(line)
        clen += len(line)
        if clen > INGEST_CHUNK_SIZE * 2:
            chunks.append("\n".join(current))
            current, clen = [], 0
    if current:
        chunks.append("\n".join(current))
    return [c for c in chunks if c.strip()]


def _chunk_markdown(content):
    chunks, current, clen = [], [], 0
    for line in content.split("\n"):
        if line.startswith("#") and clen > 100:
            chunks.append("\n".join(current))
            current, clen = [], 0
        current.append(line)
        clen += len(line)
        if clen > INGEST_CHUNK_SIZE * 2:
            chunks.append("\n".join(current))
            current, clen = [], 0
    if current:
        chunks.append("\n".join(current))
    return [c for c in chunks if c.strip()]


# ==============================
# HEALTH CHECK
# ==============================
def _check_model_health():
    url = _vps_url()
    try:
        r = requests.get(f"{url}/api/tags", timeout=5)
        if r.status_code == 200:
            models = [m.get("name", "") for m in r.json().get("models", [])]
            for needed in [MODEL_FAST, MODEL_DEEP]:
                if not any(needed in m for m in models):
                    mem.add_notification(f"Model '{needed}' not found. Run: ollama pull {needed}",
                                         "warning", "startup")
            return True
    except Exception:
        pass
    mem.add_notification("VPS unreachable at startup", "warning", "startup")
    return False


# ==============================
# COMMAND HELPERS
# ==============================
def _cmd_status():
    state = mem.load_config_state()
    stats = mem.get_db_stats()
    projects = mem.get_projects()
    proj_list = ", ".join(f"{p['name']}({p['count']})" for p in projects) or "none"
    type_str = ", ".join(f"{k}:{v}" for k, v in stats.get("types", {}).items()) or "none"

    cached_vps = _vps_cache.get("url", "unknown")
    vps_down = _vps_cache.get("was_down", True)
    vps_status = "❌ Unreachable" if vps_down else "✅ Online"

    return "\n".join([
        "", "⚙️   KOGNAC v10", "─" * 44,
        f"  VPS:           {vps_status} ({cached_vps})",
        f"  Mode:          {state.get('model_mode', 'hybrid')}",
        f"  Model:         {state.get('active_model', 'fast')} ({MODELS.get(state.get('active_model', 'fast'), '?')})",
        f"  Auto-route:    {'ON' if AUTO_ROUTE_ENABLED else 'OFF'}",
        f"  Self-eval:     {'ON' if _self_eval_active else 'OFF'}",
        f"  Teaching:      {'ON' if state.get('teaching_enabled') else 'OFF'}",
        f"  Session:       {_active_session} ({len(get_session().history)} msgs)",
        f"  Retry queue:   {len(_retry_queue)}",
        "─" * 44,
        f"  DB:            {stats['size_mb']:.1f}MB / {MAX_DB_SIZE_MB}MB",
        f"  Memories:      {stats['total']} (🔒{stats['locked']} 📌{stats['pinned']})",
        f"  Types:         {type_str}",
        f"  Clusters:      {stats['cluster_count']}",
        f"  Search:        {stats['faiss']}",
        f"  Prefs:         {stats['preferences']}",
        f"  Relations:     {stats['relationships']}",
        f"  Undo avail:    {stats['undo_available']}",
        f"  Projects:      {proj_list}",
        "─" * 44, ""
    ])


def _is_binary(fp):
    try:
        with open(fp, "rb") as f:
            return b"\x00" in f.read(1024)
    except Exception:
        return True


def _cmd_ingest(filepath, project):
    root = os.path.dirname(BASE_DIR)
    resolved = os.path.realpath(os.path.abspath(filepath))
    if not resolved.startswith(root):
        return "❌ Outside container"
    if not os.path.isfile(resolved):
        return f"❌ Not found: {resolved}"
    if _is_binary(resolved):
        return "❌ Binary file"
    size = os.path.getsize(resolved) / (1024 * 1024)
    if size > MAX_INGEST_FILE_SIZE_MB:
        return f"❌ Too large ({size:.1f}MB)"
    try:
        with open(resolved, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except Exception as e:
        return f"❌ Read error: {e}"
    filename = os.path.basename(resolved)
    chunks = _smart_chunk(content, filename)
    current = mem.get_memory_count(project)
    if current + len(chunks) > MAX_MEMORIES_PER_PROJECT:
        return f"❌ Limit exceeded ({current}+{len(chunks)} > {MAX_MEMORIES_PER_PROJECT})"
    for i, chunk in enumerate(chunks):
        wrapped = f"[INGESTED_SOURCE file={filename} chunk={i + 1}/{len(chunks)}]\n{chunk}\n[/INGESTED_SOURCE]"
        mem.add_memory(wrapped, project=project, cluster="ingested", importance=1.0, memory_type="ingested")
    return f"✅ '{filename}' — {len(chunks)} chunks"


def _should_exclude(path):
    parts = path.replace("\\", "/").split("/")
    return any(p in INGEST_EXCLUDE_DIRS for p in parts)


def _cmd_ingestdir(dirpath, project):
    root = os.path.dirname(BASE_DIR)
    resolved = os.path.realpath(os.path.abspath(dirpath))
    if not resolved.startswith(root):
        return "❌ Outside container"
    if not os.path.isdir(resolved):
        return "❌ Not a directory"
    files = []
    for ext in ("*.py", "*.js", "*.md", "*.txt", "*.json", "*.yaml", "*.yml", "*.toml",
                "*.sh", "*.c", "*.cpp", "*.h", "*.rs", "*.go", "*.java", "*.ts", "*.html", "*.css"):
        for fp in glob.glob(os.path.join(resolved, "**", ext), recursive=True):
            rel = os.path.relpath(fp, resolved)
            if not _should_exclude(rel):
                files.append(fp)
    files = sorted(set(files))
    if not files:
        return "❌ No text files found"
    total = len(files)
    capped = files[:INGESTDIR_MAX_FILES]
    cap_warn = f"\n⚠️  Found {total} — ingesting first {INGESTDIR_MAX_FILES}." if total > INGESTDIR_MAX_FILES else ""
    ok = fail = 0
    details = []
    for fp in capped:
        r = _cmd_ingest(fp, project)
        details.append(f"  {os.path.relpath(fp, resolved)}: {r}")
        if r.startswith("✅"):
            ok += 1
        else:
            fail += 1
    return f"📁 {ok} ok, {fail} failed ({len(capped)} files){cap_warn}\n" + "\n".join(details)


def _cmd_view(id_str):
    try:
        mid = int(id_str)
    except ValueError:
        return "❌ Invalid ID"
    row = mem.get_memory_by_id(mid)
    if not row:
        return f"❌ Not found: {mid}"
    mid, proj, cl, content, imp, ac, la, ts, locked, mtype, pinned = row
    lines = [
        f"─── MEMORY {mid} ───",
        f"Project: {proj}  Cluster: {cl}  Type: {mtype}",
        f"Importance: {imp:.1f}  Accessed: {ac}x  {'🔒' if locked else '🔓'}  {'📌' if pinned else ''}",
        f"Created: {time.strftime('%Y-%m-%d %H:%M', time.localtime(ts))}",
        f"Last access: {time.strftime('%Y-%m-%d %H:%M', time.localtime(la))}",
        "─" * 40, content, "─" * 40
    ]
    related = mem.get_related(mid)
    if related:
        lines.append(f"\nRelated ({len(related)}):")
        for rid, rc, rcl, rtype in related:
            lines.append(f"  [{rid}] ({rtype}) {rc[:80]}...")
    return "\n".join(lines)


def _cmd_list(args, project):
    cluster = mtype = None
    page = 0
    for a in args:
        if a.startswith("c=") or a.startswith("cluster="):
            cluster = a.split("=", 1)[1]
        elif a.startswith("t=") or a.startswith("type="):
            mtype = a.split("=", 1)[1]
        elif a.startswith("p=") or a.startswith("page="):
            try:
                page = int(a.split("=", 1)[1])
            except ValueError:
                pass
    rows = mem.list_memories(project, cluster=cluster, memory_type=mtype, page=page)
    if not rows:
        return "No memories found"
    lines = [f"{'ID':>5} {'Type':<12} {'Cluster':<10} {'Imp':>4} {'📌':>2} Content"]
    for mid, content, cl, imp, mt, pinned in rows:
        preview = content[:55].replace("\n", " ")
        lines.append(f"{mid:>5} {(mt or 'general'):<12} {cl:<10} {imp:>4.1f} {'📌' if pinned else '  '} {preview}...")
    lines.append(f"\n  Page {page} (20/page). Use /list p={page + 1} for next.")
    return "\n".join(lines)


# ==============================
# COMMAND KERNEL
# ==============================
def cmd(input_str, project="default"):
    parts = input_str.strip().split()
    text_parts = input_str.strip().split(None, 2)
    if not parts:
        return "empty"
    c = parts[0].lower()

    if c == "/help":
        return HELP_TEXT
    if c == "/status":
        return _cmd_status()
    if c == "/debug":
        return f"─── LAST CONTEXT ({len(_last_context)} chars) ───\n{_last_context[:3000]}\n───"
    if c == "/clear":
        get_session().clear()
        return "✅ Cleared"

    # Sessions
    if c == "/session":
        if len(parts) < 2:
            return "Usage: /session new|list|delete|<name>"
        sub = parts[1].lower()
        if sub == "list":
            return list_sessions_text()
        if sub == "new":
            name = parts[2] if len(parts) > 2 else f"s_{int(time.time())}"
            ok, msg = create_session(name)
            if ok:
                switch_session(name)
            return msg
        if sub == "delete":
            return delete_session(parts[2])[1] if len(parts) > 2 else "Usage: /session delete <name>"
        return switch_session(sub)[1]

    # Preferences
    if c == "/pref":
        rest = input_str[len("/pref"):].strip()
        if "=" not in rest:
            return "Usage: /pref <key> = <value>"
        k, v = rest.split("=", 1)
        mem.set_preference(k.strip(), v.strip(), source="manual")
        return f"✅ {k.strip()} = {v.strip()}"
    if c == "/prefs":
        prefs = mem.get_all_preferences()
        return "\n".join(f"  {p['key']} = {p['value']} ({p['source']})" for p in prefs) if prefs else "No preferences"
    if c == "/delpref":
        if len(parts) < 2:
            return "Usage: /delpref <key>"
        ok, msg = mem.delete_preference(parts[1])
        return f"✅ Blocked '{parts[1]}'" if ok else f"❌ {msg}"

    # Memory
    if c == "/ingest":
        return _cmd_ingest(parts[1], project) if len(parts) >= 2 else "Usage: /ingest <path>"
    if c == "/ingestdir":
        return _cmd_ingestdir(parts[1], project) if len(parts) >= 2 else "Usage: /ingestdir <path>"
    if c == "/find":
        if len(parts) < 2:
            return "Usage: /find <keyword>"
        results = mem.find_memories(parts[1], project)
        if not results:
            return f"🔍 No results for '{parts[1]}'"
        return "\n".join(f"  [{mid}] ({cl}, imp={imp:.1f}) {content[:100]}..."
                         for mid, content, cl, imp in results)
    if c == "/view":
        return _cmd_view(parts[1]) if len(parts) >= 2 else "Usage: /view <id>"
    if c == "/list":
        return _cmd_list(parts[1:], project)
    if c == "/delete":
        if len(parts) < 2:
            return "Usage: /delete <id> [confirm]"
        try:
            mid = int(parts[1])
        except ValueError:
            return "❌ Invalid ID"
        confirmed = len(parts) >= 3 and parts[2].lower() == "confirm"
        if not confirmed:
            try:
                if input(f"Delete {mid}? (y/N): ").strip().lower() != "y":
                    return "Cancelled"
            except (EOFError, RuntimeError):
                return "❌ Use: /delete <id> confirm"
        ok, msg = mem.delete_memory(mid)
        return f"✅ Deleted {mid} (use /undo to restore)" if ok else f"❌ {msg}"
    if c == "/undo":
        ok, msg = mem.undo_delete()
        return f"✅ {msg}" if ok else f"❌ {msg}"
    if c == "/update":
        if len(text_parts) < 3:
            return "Usage: /update <id> <text>"
        try:
            mid = int(text_parts[1])
        except ValueError:
            return "❌ Invalid ID"
        ok, msg = mem.update_memory(mid, text_parts[2])
        return f"✅ Updated {mid}" if ok else f"❌ {msg}"
    if c == "/lock":
        if len(parts) < 2: return "Usage: /lock <id>"
        try: ok, msg = mem.lock_memory(int(parts[1]))
        except ValueError: return "❌ Invalid ID"
        return f"🔒 Locked {parts[1]}" if ok else f"❌ {msg}"
    if c == "/unlock":
        if len(parts) < 2: return "Usage: /unlock <id>"
        try: ok, msg = mem.unlock_memory(int(parts[1]))
        except ValueError: return "❌ Invalid ID"
        return f"🔓 Unlocked {parts[1]}" if ok else f"❌ {msg}"
    if c == "/pin":
        if len(parts) < 2: return "Usage: /pin <id>"
        try: ok, msg = mem.pin_memory(int(parts[1]))
        except ValueError: return "❌ Invalid ID"
        return f"📌 Pinned {parts[1]}" if ok else f"❌ {msg}"
    if c == "/unpin":
        if len(parts) < 2: return "Usage: /unpin <id>"
        try: ok, msg = mem.unpin_memory(int(parts[1]))
        except ValueError: return "❌ Invalid ID"
        return f"Unpinned {parts[1]}" if ok else f"❌ {msg}"
    if c == "/relate":
        if len(parts) < 3: return "Usage: /relate <id1> <id2>"
        try: a, b = int(parts[1]), int(parts[2]); mem.add_relationship(a, b); return f"🔗 {a} ↔ {b}"
        except ValueError: return "❌ Invalid IDs"
    if c == "/reindex":
        return f"✅ Re-indexed {mem.reindex_all()} memories"

    # Export/Import
    if c == "/export":
        if len(parts) >= 2 and parts[1].lower() == "session":
            fp, err = export_session_to_file()
            return f"✅ {fp}" if fp else f"❌ {err}"
        ok, r = mem.export_memories(parts[1] if len(parts) >= 2 else None)
        return f"✅ {r}" if ok else f"❌ {r}"
    if c == "/import":
        if len(parts) < 2: return "Usage: /import <path> [project]"
        ok, r = mem.import_memories(parts[1], target_project=parts[2] if len(parts) >= 3 else None)
        return f"✅ {r}" if ok else f"❌ {r}"

    # Code
    if c in ("/run", "/exec"):
        code = input_str[len(c):].strip()
        if not code: return "Usage: /run <code>"
        r = execute_code(code)
        if r["success"]:
            return f"✅ {r['duration']}s\n{r['output'] or '(no output)'}" + (f"\nStderr: {r['error']}" if r['error'] else "")
        return f"❌ {r['error']}\n{r['output'] or ''}"
    if c == "/runfile":
        if len(parts) < 2: return "Usage: /runfile <path>"
        resolved = os.path.realpath(os.path.abspath(parts[1]))
        if not resolved.startswith(CONTAINER_ROOT): return "❌ Outside container"
        if not os.path.isfile(resolved): return f"❌ Not found"
        try:
            with open(resolved, "r") as f: code = f.read()
            r = execute_code(code)
            if r["success"]:
                return f"✅ {r['duration']}s\n{r['output'] or '(no output)'}"
            return f"❌ {r['error']}\n{r['output'] or ''}"
        except Exception as e: return f"❌ {e}"

    # Model
    if c == "/model":
        if len(parts) < 2 or parts[1] not in ("1", "2"): return "Usage: /model 1|2"
        mem.save_config_state({"active_model": "fast" if parts[1] == "1" else "deep", "model_mode": "single"})
        return f"✅ {MODELS['fast' if parts[1] == '1' else 'deep']}"
    if c == "/hybrid":
        mem.save_config_state({"model_mode": "hybrid"}); return "✅ Hybrid"
    if c == "/single":
        mem.save_config_state({"model_mode": "single"}); return "✅ Single"
    if c == "/auto":
        mem.save_config_state({"model_mode": "auto"}); return "✅ Auto-route"

    if c == "/eval":
        global _self_eval_active
        _self_eval_active = not _self_eval_active
        return f"🔍 Self-eval {'ON' if _self_eval_active else 'OFF'}"

    if c == "/teach_on":
        mem.save_config_state({"teaching_enabled": True}); return "🧠 Teaching ON"
    if c == "/teach_off":
        mem.save_config_state({"teaching_enabled": False}); return "🧘 Teaching OFF"
    if c == "/understood":
        mem.save_config_state({"teaching_enabled": False}); return "✔ Understood"

    return f"❓ Unknown '{c}'. /help"


# ==============================
# STARTUP
# ==============================
def startup():
    validate_paths()
    mem.startup()
    _ensure_sessions()
    _check_model_health()
    mem.add_notification("KOGNAC v10 started", "info", "brain")
    log.info("Brain startup complete")


# ==============================
# CLI
# ==============================
def _cli_prompt():
    state = mem.load_config_state()
    return f"[{state.get('model_mode', 'hybrid')} | {_active_session}] > "


if __name__ == "__main__":
    startup()
    print("⚙️  KOGNAC v10  |  /help for commands")
    while True:
        try:
            q = input(_cli_prompt()).strip()
            if not q:
                continue
            if q.lower() in ("exit", "quit"):
                mem.stop_background()
                print("👋 Done.")
                break
            if q.startswith("/"):
                print(cmd(q))
                continue
            print(f"\n{_spin(ask, q)}\n")
        except KeyboardInterrupt:
            mem.stop_background()
            print("\n👋 Done.")
            break
