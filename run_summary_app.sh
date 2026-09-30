#!/usr/bin/env bash
#
# Serve the paper-summary app and open it in the browser.
#
# The app fetches papers/topics.json and the summary markdown at runtime, which
# a browser refuses to do from a file:// page. So it needs a real HTTP server,
# rooted at the repository so that papers/ resolves.
#
# Override anything from the environment, e.g.
#   PORT=9000 NO_BROWSER=1 ./run_summary_app.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3.10}"
PORT="${PORT:-8765}"
PORT_TRIES="${PORT_TRIES:-20}"
HOST="${HOST:-127.0.0.1}"
NO_BROWSER="${NO_BROWSER:-}"
APP_PATH="papers/app/index.html"

server_pid=""

cleanup() {
  if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
    echo "[stop] shutting down the server (pid $server_pid)"
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

# --- checks ----------------------------------------------------------------

if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "[error] $PYTHON not found. This project needs python3.10 — plain python3" >&2
  echo "[error] has no xarray and is not the project interpreter." >&2
  exit 1
fi

for required in "papers/topics.json" "$APP_PATH" "papers/app/app.js" "papers/app/vendor/marked.min.js"; do
  if [[ ! -f "$PROJECT_ROOT/$required" ]]; then
    echo "[error] missing $required — the app is incomplete" >&2
    exit 1
  fi
done

if ! "$PYTHON" -c "import json,sys; json.load(open(sys.argv[1]))" \
     "$PROJECT_ROOT/papers/topics.json" 2>/dev/null; then
  echo "[error] papers/topics.json is not valid JSON" >&2
  exit 1
fi

n_papers="$("$PYTHON" - "$PROJECT_ROOT/papers/topics.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print(sum(len(t.get("papers", [])) for t in d.get("topics", [])))
PY
)"
echo "[info] $n_papers papers indexed in papers/topics.json"

# --- find a free port ------------------------------------------------------

free_port() {
  local start="$1" tries="$2" p
  for (( p = start; p < start + tries; p++ )); do
    if "$PYTHON" - "$p" <<'PY' 2>/dev/null
import socket, sys
s = socket.socket()
try:
    s.bind(("127.0.0.1", int(sys.argv[1])))
except OSError:
    sys.exit(1)
finally:
    s.close()
PY
    then
      echo "$p"
      return 0
    fi
  done
  return 1
}

if ! port="$(free_port "$PORT" "$PORT_TRIES")"; then
  echo "[error] no free port in $PORT..$((PORT + PORT_TRIES - 1))" >&2
  exit 1
fi
[[ "$port" != "$PORT" ]] && echo "[note] port $PORT was busy, using $port"

# --- serve -----------------------------------------------------------------

url="http://${HOST}:${port}/${APP_PATH}"

cd "$PROJECT_ROOT"
"$PYTHON" -m http.server "$port" --bind "$HOST" >/dev/null 2>&1 &
server_pid=$!

for _ in $(seq 1 40); do
  if ! kill -0 "$server_pid" 2>/dev/null; then
    echo "[error] the server exited immediately" >&2
    exit 1
  fi
  if "$PYTHON" - "$HOST" "$port" <<'PY' 2>/dev/null
import socket, sys
s = socket.socket(); s.settimeout(0.3)
try:
    s.connect((sys.argv[1], int(sys.argv[2])))
except OSError:
    sys.exit(1)
finally:
    s.close()
PY
  then
    break
  fi
  sleep 0.1
done

echo "[serve] $url"
echo "[serve] root $PROJECT_ROOT"

if [[ -z "$NO_BROWSER" ]]; then
  if command -v open >/dev/null 2>&1; then
    open "$url"
  elif command -v xdg-open >/dev/null 2>&1; then
    xdg-open "$url" >/dev/null 2>&1 || true
  else
    echo "[note] could not find a browser opener — open the URL above yourself"
  fi
fi

echo "[done] press Ctrl-C to stop"
wait "$server_pid"
