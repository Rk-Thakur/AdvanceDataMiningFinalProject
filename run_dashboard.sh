#!/usr/bin/env bash
# Builds the Docker image, starts the dashboard container and opens it in the browser.
# Usage: ./run_dashboard.sh            (stop it later with: docker stop wine-dashboard)
#        PORT=8080 ./run_dashboard.sh  (use another port)
set -euo pipefail
cd "$(dirname "$0")"

IMAGE=wine-dashboard
CONTAINER=wine-dashboard
PORT="${PORT:-8000}"

echo "Building image (trains the models; first build takes a few minutes)..."
docker build -t "$IMAGE" .

docker rm -f "$CONTAINER" >/dev/null 2>&1 || true

# use the next free port if the requested one is already taken
port_busy() { (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; }
while port_busy "$PORT"; do
  echo "Port ${PORT} is in use, trying $((PORT + 1))"
  PORT=$((PORT + 1))
done
URL="http://localhost:${PORT}/"

docker run -d --rm --name "$CONTAINER" -p "${PORT}:8000" "$IMAGE" >/dev/null

echo "Waiting for the dashboard at ${URL} ..."
for _ in $(seq 1 30); do
  if curl -fs "$URL" >/dev/null 2>&1; then break; fi
  sleep 1
done

case "$(uname -s)" in
  Darwin) open "$URL" ;;
  Linux) xdg-open "$URL" >/dev/null 2>&1 || true ;;
  MINGW*|MSYS*|CYGWIN*) start "$URL" ;;
esac

echo "Dashboard running at ${URL}"
echo "Stop it with: docker stop ${CONTAINER}"
