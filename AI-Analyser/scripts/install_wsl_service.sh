#!/usr/bin/env bash
# Run the analyser as a systemd service inside WSL, so it survives terminal closes,
# restarts on failure, and comes back when the distro starts.
set -e
cat > /etc/systemd/system/wes.service <<'UNIT'
[Unit]
Description=WES AI Analyser
After=network-online.target ollama.service
Wants=ollama.service

[Service]
ExecStart=/root/wes/start.sh
Restart=always
RestartSec=5
WorkingDirectory=/c/Users/Anderson/Desktop/Projects/LLM_development/AI-Analyser

[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
systemctl enable --now wes
for i in $(seq 40); do
  curl -s -m 2 http://127.0.0.1:8000/system >/dev/null && break
  sleep 1
done
if curl -s -m 3 http://127.0.0.1:8000/system >/dev/null; then
  echo "wes.service: up"
  systemctl is-enabled wes | sed 's/^/  enabled: /'
else
  echo "wes.service FAILED"; systemctl status wes --no-pager | tail -20; exit 1
fi
