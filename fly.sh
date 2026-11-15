#!/usr/bin/env bash
# SkyHand: fly a drone with a Rokoko glove. One script for setup and flight (Linux / macOS).
#
#   ./fly.sh                 fly the real drone (MQTT). First run creates .venv automatically.
#   ./fly.sh --ai            + Nemotron VLA obstacle avoidance (needs VERDA_INFERENCE_KEY)
#   ./fly.sh --dry-run       flight window, nothing sent to the drone
#   ./fly.sh --sim           simulator, no drone
#   ./fly.sh --fake          simulator, no glove
#   ./fly.sh --train         record your gestures, train the model (Device SDK only)
#   ./fly.sh --check         check the glove gestures, no drone
#   ./fly.sh --setup         (re)create .venv
#
# Options: --mqtt-host IP (default 192.168.4.18)  --vlm-source URL  --vlm-interval S (0.1..5)
#          --ai-hover-only  --studio (use Rokoko Studio instead of the Device SDK)
# Logs: /tmp/skyhand-*.log
set -eo pipefail
cd "$(dirname "$0")"

case " $* " in *" --help "*|*" -h "*) sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;; esac

setup_venv() {
  local py=${PYTHON:-python3}
  "$py" -c 'import sys; assert sys.version_info >= (3, 9)' 2>/dev/null || { echo "[fly] need Python 3.9+"; exit 1; }
  "$py" -c 'import tkinter' 2>/dev/null || {
    echo "[fly] tkinter missing. mac: brew install python-tk   linux: sudo apt install python3-tk"; exit 1; }
  echo "[fly] creating .venv ..."
  "$py" -m venv .venv
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -q -r requirements.txt
  echo "[fly] .venv ready"
}
if [ ! -x .venv/bin/python ] || [[ " $* " == *" --setup "* ]]; then
  setup_venv
  [[ " $* " == *" --setup "* ]] && exit 0
fi
export PATH="$PWD/.venv/bin:$PATH"

mode=safe; studio=0; pass=(); targs=()
ai=0; vlm_source="rtmp://localhost:1935/live/drone"; vlm_interval=0.2; ai_slide=1; mqtt_host=192.168.4.18
while [ $# -gt 0 ]; do
  a=$1; shift
  case "$a" in
    --vlm-source) vlm_source=$1; shift; continue ;;
    --vlm-interval) vlm_interval=$1; shift; continue ;;
    --ai-slide) ai_slide=1; continue ;;
    --ai-hover-only) ai_slide=0; continue ;;
    --mqtt-host|--host) mqtt_host=$1; shift; continue ;;
  esac
  case "$a" in
    --safe) mode=safe ;;
    --ai) mode=safe; ai=1; pass+=(--ai) ;;
    --sim) mode=sim ;;
    --fake) mode=fake ;;
    --train) mode=train ;;
    --check) mode=check ;;
    --studio) studio=1 ;;
    --quick|--add|--text|--gestures=*) targs+=("$a") ;;
    *) pass+=("$a") ;;
  esac
done

SDK_HOME="${ROKOKO_SDK_HOME:-$HOME/.local/share/rokoko-device-sdk}"
# The Device SDK also runs on macOS (install-device-sdk.sh); use Studio only when the SDK isn't installed.
if [ "$studio" = 0 ] && [ ! -f "$SDK_HOME/env" ]; then
  studio=1
  [ "$mode" = fake ] || echo "[fly] no Rokoko Device SDK here: expecting Rokoko Studio streaming to port 14043"
fi
if [ "$studio" = 1 ] && [ "$mode" = train ]; then
  echo "[fly] --train needs the Device SDK (raw IMU data); with Studio the gesture rules are used instead"; exit 1
fi

pids=()
cleanup() { for p in ${pids[@]+"${pids[@]}"}; do kill "$p" 2>/dev/null || true; done; wait 2>/dev/null || true; }
trap cleanup EXIT INT TERM
SETSID=""; command -v setsid >/dev/null && SETSID=setsid  # macOS has no setsid

# close helpers left over from a previous run
for pat in "python3? -m skyhand" "scripts/fake_glove.py" "scripts/sdk_bridge.py" "rkk-hand-solver" \
           "scripts/vlm_service.py" "scripts/ai_mux.py"; do
  pkill -u "$(id -u)" -f "$pat" 2>/dev/null || true
done
sleep 0.5

gargs=()  # glove settings for the flying programs
if [ "$mode" = fake ]; then
  $SETSID python3 scripts/fake_glove.py demo > /tmp/skyhand-fake.log 2>&1 &
  pids+=($!)
  echo "[fly] fake glove demo running (log /tmp/skyhand-fake.log)"
elif [ "$studio" = 1 ]; then
  gargs=(--finger-range 3.4)
  [ "$mode" = safe ] && gargs+=(--gesture-model none)  # rules only: the trained model needs SDK raw IMU
else
  # shellcheck disable=SC1091
  . "$SDK_HOME/env"
  if pgrep -x rokoko-sdk >/dev/null; then
    echo "[fly] SDK driver already running"
  else
    $SETSID rokoko-sdk --auto-stream-usb > /tmp/skyhand-driver.log 2>&1 &
    p=$!; pids+=($p); sleep 2
    kill -0 "$p" 2>/dev/null || { tail -3 /tmp/skyhand-driver.log
      echo "[fly] driver failed; if it says 'not activated' run: rokoko signin && rokoko activate"; exit 1; }
    echo "[fly] SDK driver started (log /tmp/skyhand-driver.log)"
  fi
  $SETSID python3 scripts/sdk_bridge.py > /tmp/skyhand-bridge.log 2>&1 &
  p=$!; pids+=($p); sleep 2
  kill -0 "$p" 2>/dev/null || { tail -5 /tmp/skyhand-bridge.log; exit 1; }
  echo "[fly] glove bridge running (log /tmp/skyhand-bridge.log)"
  gargs=(--finger-range 1.5)
  [ -f models/gestures.npz ] && gargs+=(--gesture-model models/gestures.npz)
fi
[ "$mode" = train ] || [ "$mode" = fake ] || echo "[fly] hold your hand FLAT, palm down, fingers forward for 2 s to calibrate"

port_up() { (exec 3<>"/dev/tcp/$1/$2") 2>/dev/null; }
is_local() {
  case "$1" in localhost|127.*|0.0.0.0) return 0 ;; esac
  ifconfig 2>/dev/null | grep -q "inet $1 " || ip -4 addr 2>/dev/null | grep -q "inet $1/"
}

if [ "$ai" = 1 ]; then
  # 1) MQTT broker (only if it should be on this machine and isn't running)
  if ! port_up "$mqtt_host" 1883; then
    if is_local "$mqtt_host" && command -v mosquitto >/dev/null; then
      printf 'listener 1883\nallow_anonymous true\n' > /tmp/skyhand-mosquitto.conf
      mosquitto -c /tmp/skyhand-mosquitto.conf > /tmp/skyhand-mosquitto.log 2>&1 &
      pids+=($!); sleep 1
      echo "[fly] started MQTT broker on port 1883 (log /tmp/skyhand-mosquitto.log)"
    else
      echo "[fly] warning: no MQTT broker at $mqtt_host:1883 (start mosquitto there, or pass --mqtt-host)"
    fi
  fi
  # 2) video server the drone pushes RTMP to
  case "$vlm_source" in rtmp://localhost*|rtmp://127.*|rtsp://localhost*|rtsp://127.*)
    if ! port_up 127.0.0.1 1935 && command -v mediamtx >/dev/null; then
      cfg=""; for c in /opt/homebrew/etc/mediamtx/mediamtx.yml /usr/local/etc/mediamtx/mediamtx.yml /etc/mediamtx/mediamtx.yml; do
        [ -f "$c" ] && cfg=$c && break; done
      mediamtx $cfg > /tmp/skyhand-mediamtx.log 2>&1 &
      pids+=($!); sleep 1
      echo "[fly] started video server mediamtx (log /tmp/skyhand-mediamtx.log)"
    fi ;;
  esac
  # 3) VLM service (needs the API key)
  if [ -z "${VERDA_INFERENCE_KEY:-}" ] && [ -f "$HOME/.verda_env" ]; then
    # shellcheck disable=SC1091
    . "$HOME/.verda_env"
  fi
  awk -v v="$vlm_interval" 'BEGIN { exit !(v >= 0.1 && v <= 5) }' || {
    echo "[fly] --vlm-interval must be 0.1..5 s (AI messages further apart can't agree, see skyhand/ai_mix.py)"; exit 1; }
  if [ -n "${VERDA_INFERENCE_KEY:-}" ]; then
    $SETSID python3 scripts/vlm_service.py --source "$vlm_source" --mqtt-host "$mqtt_host" \
      --min-interval "$vlm_interval" --workers 2 > /tmp/skyhand-vlm.log 2>&1 &
    pids+=($!)
    echo "[fly] VLM running on $vlm_source, every ${vlm_interval}s (log /tmp/skyhand-vlm.log)"
  else
    echo "[fly] warning: VERDA_INFERENCE_KEY not set (no ~/.verda_env): no VLM, the glove drives the drone alone"
  fi
  # 4) mixer: glove (skyhand/pilot_vel) + AI -> drone/cmd_vel. Without it nothing reaches the drone in --ai mode.
  margs=(--local-host "$mqtt_host" --cloud-host "$mqtt_host")
  [ "$ai_slide" = 1 ] || margs+=(--stop-only)
  [[ " ${pass[*]} " == *" --dry-run "* ]] && margs+=(--out-topic skyhand/shadow_vel)  # dry run: never drive the drone
  $SETSID python3 scripts/ai_mux.py "${margs[@]}" > /tmp/skyhand-mux.log 2>&1 &
  p=$!; pids+=($p); sleep 1
  kill -0 "$p" 2>/dev/null || { tail -5 /tmp/skyhand-mux.log; exit 1; }
  echo "[fly] AI mixer running: on an obstacle the AI backs off, then $([ "$ai_slide" = 1 ] && echo "slides to the free side / hovers" || echo "hovers") (log /tmp/skyhand-mux.log)"
fi

all=(${gargs[@]+"${gargs[@]}"} ${pass[@]+"${pass[@]}"} --mqtt-host "$mqtt_host")
case "$mode" in
  safe)  python3 scripts/safe_fly_v2.py ${all[@]+"${all[@]}"} ;;
  sim)   python3 -m skyhand --drone sim --viz --simple ${all[@]+"${all[@]}"} ;;
  fake)  python3 -m skyhand --drone sim --viz --simple ${pass[@]+"${pass[@]}"} ;;
  train) python3 scripts/train_gestures.py ${targs[@]+"${targs[@]}"} ;;
  check) python3 scripts/glove_check.py ;;
esac
