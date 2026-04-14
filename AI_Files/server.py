import os
import json
import functools
from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from brain import (
    ask, ask_stream, cmd, startup,
    create_session, switch_session, delete_session, list_sessions_text,
    get_session, _sessions, _active_session, export_session_to_file,
    _self_eval_active, _retry_queue, _vps_cache, MODELS
)
from config import (
    SERVER_HOST, SERVER_PORT, MAX_PROMPT_LENGTH, BASE_DIR, API_KEY,
    RATE_LIMIT_DEFAULT, RATE_LIMIT_ASK, RATE_LIMIT_STREAM,
    RATE_LIMIT_STORAGE, STREAM_ENABLED, SANDBOX_ENABLED,
    AUTO_ROUTE_ENABLED, MAX_DB_SIZE_MB
)
from logger import get_logger
from sandbox import execute_code
import memory as mem

log = get_logger("server")
startup()

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})
limiter = Limiter(app=app, key_func=get_remote_address,
                  default_limits=[RATE_LIMIT_DEFAULT], storage_uri=RATE_LIMIT_STORAGE)
UI_DIR = os.path.join(BASE_DIR, "ui")

ALLOWED = {"/help", "/status", "/debug", "/find", "/view", "/list", "/clear",
            "/model", "/hybrid", "/single", "/auto", "/teach_on", "/teach_off",
            "/understood", "/lock", "/unlock", "/pin", "/unpin", "/delete", "/undo",
            "/update", "/export", "/import", "/session", "/pref", "/prefs", "/delpref",
            "/reindex", "/run", "/exec", "/runfile", "/ingest", "/ingestdir", "/relate", "/eval"}


def auth(f):
    @functools.wraps(f)
    def wrap(*a, **kw):
        if API_KEY and request.headers.get("X-API-Key") != API_KEY:
            return jsonify({"error": "Unauthorized"}), 401
        return f(*a, **kw)
    return wrap


# ==============================
# STREAMING (phase-aware SSE)
# ==============================
@app.route("/api/ask/stream", methods=["POST"])
@auth
@limiter.limit(RATE_LIMIT_STREAM)
def handle_stream():
    if not STREAM_ENABLED:
        return jsonify({"error": "Streaming disabled"}), 400
    data = request.json or {}
    prompt = data.get("prompt", "").strip()
    project = data.get("project", "default").strip()
    if not prompt:
        return jsonify({"error": "Empty"}), 400
    if len(prompt) > MAX_PROMPT_LENGTH:
        return jsonify({"error": "Too long"}), 400
    if prompt.startswith("/"):
        if prompt.startswith("/delete ") and "confirm" not in prompt:
            prompt = prompt.rstrip() + " confirm"
        return _cmd_route(prompt, project)

    def gen():
        try:
            for item in ask_stream(prompt, project=project):
                yield f"data: {json.dumps(item)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return Response(gen(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ==============================
# BATCH
# ==============================
@app.route("/api/ask", methods=["POST"])
@auth
@limiter.limit(RATE_LIMIT_ASK)
def handle_ask():
    data = request.json or {}
    prompt = data.get("prompt", "").strip()
    project = data.get("project", "default").strip()
    if not prompt:
        return jsonify({"error": "Empty"}), 400
    if len(prompt) > MAX_PROMPT_LENGTH:
        return jsonify({"error": "Too long"}), 400
    if prompt.startswith("/"):
        if prompt.startswith("/delete ") and "confirm" not in prompt:
            prompt = prompt.rstrip() + " confirm"
        return _cmd_route(prompt, project)
    try:
        return jsonify({"response": ask(prompt, project=project)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _cmd_route(prompt, project):
    word = prompt.split()[0].lower()
    if word not in ALLOWED:
        return jsonify({"response": f"❌ '{word}' unavailable in UI"})
    try:
        return jsonify({"response": cmd(prompt, project=project)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==============================
# STATUS (uses cached VPS — no blocking request)
# ==============================
@app.route("/api/status", methods=["GET"])
@auth
def handle_status():
    try:
        state = mem.load_config_state()
        stats = mem.get_db_stats()
        return jsonify({
            "status": "ok",
            "details": cmd("/status"),
            "summary": {
                "vps_url": _vps_cache.get("url", "unknown"),
                "vps_online": not _vps_cache.get("was_down", True),
                "model_mode": state.get("model_mode", "hybrid"),
                "active_model": state.get("active_model", "fast"),
                "auto_route": AUTO_ROUTE_ENABLED,
                "self_eval": _self_eval_active,
                "streaming": STREAM_ENABLED,
                "teaching": state.get("teaching_enabled", True),
                "session": _active_session,
                "session_msgs": len(get_session().history),
                "retry_queue": len(_retry_queue),
                "db_size_mb": round(stats["size_mb"], 1),
                "db_max_mb": MAX_DB_SIZE_MB,
                "total_memories": stats["total"],
                "pinned": stats["pinned"],
                "undo_available": stats["undo_available"],
                "interactions": state.get("total_interactions", 0),
            }
        })
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


# ==============================
# PROJECTS
# ==============================
@app.route("/api/projects", methods=["GET"])
@auth
def get_projects():
    projects = mem.get_projects()
    if not any(p["name"] == "default" for p in projects):
        projects.insert(0, {"name": "default", "count": 0})
    return jsonify({"projects": projects})


@app.route("/api/projects", methods=["POST"])
@auth
def create_project():
    name = (request.json or {}).get("name", "").strip().lower()
    if not name or not name.isalnum():
        return jsonify({"error": "Alphanumeric required"}), 400
    mem.add_memory(f"[PROJECT_CREATED] Project '{name}' initialized",
                   project=name, cluster="system", importance=0.5, memory_type="general")
    return jsonify({"status": "ok", "project": name})


# ==============================
# SESSIONS
# ==============================
@app.route("/api/sessions", methods=["GET"])
@auth
def get_sessions():
    from brain import _ensure_sessions
    _ensure_sessions()
    return jsonify({
        "sessions": [{"name": n, "messages": len(s.history), "active": n == _active_session}
                     for n, s in _sessions.items()],
        "active": _active_session
    })


@app.route("/api/sessions", methods=["POST"])
@auth
def new_session():
    name = (request.json or {}).get("name", "")
    ok, msg = create_session(name)
    if ok:
        switch_session(name)
    return jsonify({"success": ok, "message": msg})


@app.route("/api/sessions/switch", methods=["POST"])
@auth
def switch_sess():
    ok, msg = switch_session((request.json or {}).get("name", ""))
    return jsonify({"success": ok, "message": msg})


@app.route("/api/sessions/delete", methods=["POST"])
@auth
def del_sess():
    ok, msg = delete_session((request.json or {}).get("name", ""))
    return jsonify({"success": ok, "message": msg})


@app.route("/api/sessions/export", methods=["POST"])
@auth
def export_sess():
    fp, err = export_session_to_file()
    return jsonify({"status": "ok", "filepath": fp} if fp else {"error": err})


# ==============================
# CODE / PREFS / NOTIFICATIONS / EXPORT / IMPORT
# ==============================
@app.route("/api/execute", methods=["POST"])
@auth
@limiter.limit("10 per minute")
def run_code():
    if not SANDBOX_ENABLED:
        return jsonify({"error": "Disabled"}), 400
    code = (request.json or {}).get("code", "").strip()
    if not code:
        return jsonify({"error": "No code"}), 400
    return jsonify(execute_code(code))


@app.route("/api/preferences", methods=["GET"])
@auth
def prefs_get():
    return jsonify({"preferences": mem.get_all_preferences()})


@app.route("/api/preferences", methods=["POST"])
@auth
def prefs_set():
    d = request.json or {}
    k, v = d.get("key", "").strip(), d.get("value", "").strip()
    if not k or not v:
        return jsonify({"error": "key and value required"}), 400
    mem.set_preference(k, v, source="manual")
    return jsonify({"status": "ok"})


@app.route("/api/notifications", methods=["GET"])
@auth
def notifs():
    since = float(request.args.get("since", 0))
    return jsonify({"notifications": mem.get_notifications(since=since),
                    "unread": mem.get_unread_count(since=since)})


@app.route("/api/export", methods=["POST"])
@auth
def export():
    ok, r = mem.export_memories((request.json or {}).get("project"))
    return jsonify({"status": "ok", "filepath": r} if ok else {"error": r})


@app.route("/api/import", methods=["POST"])
@auth
def import_():
    d = request.json or {}
    ok, r = mem.import_memories(d.get("filepath", ""), target_project=d.get("project"))
    return jsonify({"status": "ok", "message": r} if ok else {"error": r})


# ==============================
# STATIC UI
# ==============================
@app.route("/")
def ui():
    return send_from_directory(UI_DIR, "index.html")


@app.route("/<path:f>")
def static_(f):
    return send_from_directory(UI_DIR, f)


if __name__ == "__main__":
    print(f"🚀 http://{SERVER_HOST}:{SERVER_PORT}")
    app.run(host=SERVER_HOST, port=SERVER_PORT, debug=False)
