#!/usr/bin/env bash
# Installs the paper-trading bot on a fresh Ubuntu server and starts it as a
# service that survives crashes and reboots. Run from the unzipped folder:
#   bash setup.sh
set -euo pipefail
cd "$(dirname "$0")"
APP="$PWD"

echo "== Checking that Binance futures data is reachable from this server"
code=$(curl -s -o /dev/null -w "%{http_code}" "https://fapi.binance.com/fapi/v1/ping" || true)
if [ "$code" != "200" ]; then
  echo "Binance answered HTTP $code. This server's region is blocked (US regions are)."
  echo "Create the server in a non-US region (e.g. Frankfurt) and try again."
  exit 1
fi

echo "== Installing Python"
sudo apt-get update -qq
sudo apt-get install -y -qq python3-venv python3-pip unzip >/dev/null

echo "== Installing Python packages (a few minutes)"
python3 -m venv "$APP/.venv"
"$APP/.venv/bin/pip" install -q --upgrade pip
"$APP/.venv/bin/pip" install -q -r "$APP/requirements-paper.txt"

echo "== Self-checks"
cd "$APP/backend"
"$APP/.venv/bin/python" -m forecast.test_forecast

echo "== Installing the service"
sed "s|__APP__|$APP|g; s|__USER__|$USER|g" "$APP/paperbot.service" | sudo tee /etc/systemd/system/paperbot.service >/dev/null
sudo systemctl daemon-reload
sudo systemctl enable --now paperbot
sleep 5
sudo systemctl --no-pager status paperbot | head -5

cat > "$HOME/status.sh" <<EOF
#!/usr/bin/env bash
cd "$APP/backend" && "$APP/.venv/bin/python" -m forecast.paper --status
EOF
chmod +x "$HOME/status.sh"
echo
echo "Done. Check the score any time with:  ~/status.sh"
