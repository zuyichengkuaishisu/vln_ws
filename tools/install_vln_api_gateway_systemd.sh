#!/usr/bin/env bash
set -euo pipefail

sudo install -m 600 /home/wzy/.config/vln-api-gateway/gateway.env /etc/vln-api-gateway.env
sudo tee /etc/systemd/system/vln-api-gateway.service >/dev/null <<'EOF'
[Unit]
Description=VLN Local API Gateway
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=wzy
WorkingDirectory=/home/wzy/vln_ws
EnvironmentFile=/etc/vln-api-gateway.env
ExecStart=/home/wzy/anaconda3/envs/vln/bin/python -m uvicorn tools.api_gateway:app --host ${HOST} --port ${PORT}
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now vln-api-gateway.service
