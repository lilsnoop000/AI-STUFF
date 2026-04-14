#!/usr/bin/env bash
set -euo pipefail

VAULT="/run/media/chadwickboseman/KOGNAC VAULT/Artificial Intelligence"
CONTAINER="$VAULT/encrypted_ai.img"
MOUNT="$VAULT/mount"
AI_DIR="$MOUNT/AI_Files"
MAPPER_NAME="ai_container"
MODE="${1:-server}"
CONTAINER_OPEN=0; MOUNTED=0; AI_PID=""

cleanup() {
    echo "🛑 Cleanup..."
    if [[ -n "$AI_PID" ]] && kill -0 "$AI_PID" 2>/dev/null; then
        kill -15 "$AI_PID" 2>/dev/null || true
        for i in {1..30}; do kill -0 "$AI_PID" 2>/dev/null || break; sleep 1; done
        kill -0 "$AI_PID" 2>/dev/null && kill -9 "$AI_PID" 2>/dev/null || true
    fi
    [[ "$MOUNTED" -eq 1 ]] && mountpoint -q "$MOUNT" && { sync; sudo umount "$MOUNT" || true; }
    [[ "$CONTAINER_OPEN" -eq 1 ]] && [[ -b "/dev/mapper/$MAPPER_NAME" ]] && { sudo cryptsetup luksClose "$MAPPER_NAME" || true; }
    echo "✅ Secured."
}
trap cleanup SIGINT SIGTERM EXIT

[[ ! -f "$CONTAINER" ]] && { echo "❌ Container not found"; exit 1; }
[[ ! -d "$MOUNT" ]] && mkdir -p "$MOUNT"
command -v cryptsetup &>/dev/null || { echo "❌ cryptsetup missing"; exit 1; }
command -v python3 &>/dev/null || { echo "❌ python3 missing"; exit 1; }
FREE_KB=$(df --output=avail "$VAULT" | tail -1)
[[ "$FREE_KB" -lt 512000 ]] && { echo "❌ <500MB free"; exit 1; }

if [[ -b "/dev/mapper/$MAPPER_NAME" ]]; then CONTAINER_OPEN=1
else sudo cryptsetup luksOpen "$CONTAINER" "$MAPPER_NAME"; CONTAINER_OPEN=1; fi

if mountpoint -q "$MOUNT"; then MOUNTED=1
else
    sudo mount "/dev/mapper/$MAPPER_NAME" "$MOUNT"
    if ! mountpoint -q "$MOUNT"; then
        echo "❌ Mount failed — filesystem may be corrupt"
        exit 1
    fi
    MOUNTED=1
fi

for DIR in "$AI_DIR" "$AI_DIR/ui" "$MOUNT/exports" "$MOUNT/logs" "$MOUNT/memory" "$MOUNT/sandbox_tmp"; do
    [[ ! -d "$DIR" ]] && mkdir -p "$DIR"
done

[[ "$(stat -c '%U' "$MOUNT")" != "$USER" ]] && sudo chown -R "$USER":"$USER" "$MOUNT"

cd "$AI_DIR"
if [[ "$MODE" == "cli" ]]; then
    echo "🚀 CLI mode"; python3 brain.py & AI_PID=$!
else
    echo "🚀 Server mode"; python3 server.py & AI_PID=$!
    if command -v curl &>/dev/null; then
        for i in {1..20}; do
            curl -s http://127.0.0.1:5000/api/status 2>/dev/null | grep -q ok && { echo "✅ http://127.0.0.1:5000"; break; }
            kill -0 "$AI_PID" 2>/dev/null || { echo "❌ Server died"; exit 1; }; sleep 1
        done
    else echo "⚠️  curl not found — try http://127.0.0.1:5000"; sleep 3; fi
fi
echo "💡 Ctrl+C to stop"; wait "$AI_PID"
