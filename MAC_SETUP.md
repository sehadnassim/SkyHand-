# SkyHand setup on a new machine (Mac or Linux)

Instructions for whoever (person or AI assistant) sets up this repo on another computer.
Goal: the pilot laptop shows the **flight window** (big status pill, TAKE OFF / LAND buttons, hand-gesture
cards that light up for the current gesture) and flies the drone over MQTT with the Rokoko glove.

## Rules
- The only flight program is `scripts/safe_fly_v2.py`, started by `./fly.sh`. Old modes (`--mqtt`, `--gesture`,
  `--llm`, `skyhand/gui.py`, `safe_gesture.py`) were removed on purpose. Do not bring them back.
- Do not change speed caps, gesture mappings or yaw sign. They were tuned on the real drone.
- Never commit API keys. `VERDA_INFERENCE_KEY` is only exported in the shell.
- Before flying, always test with `--dry-run` (prints what would be sent, sends nothing).

## 1. Install (once)
```bash
git clone https://github.com/hakim-amer/hackathon && cd hackathon/glove-drone-teleop
# mac: Python 3.9+ with tkinter.  brew install python python-tk   (or the python.org installer)
./setup_venv.sh          # creates .venv; fly.sh uses it automatically
```
If `setup_venv.sh` says tkinter is missing, install it as printed and run it again. No tkinter = no window.

## 2. Glove data
- **Linux with the Rokoko Device SDK** (`~/.local/share/rokoko-device-sdk`): `fly.sh` starts the driver and
  `scripts/sdk_bridge.py` itself and uses the trained gesture model `models/gestures.npz`.
- **macOS**: use **Rokoko Studio**. Livestreaming -> Custom -> enable, data format **JSON v3**,
  IP `127.0.0.1`, port **14043**. Glove connected and visible in Studio. `fly.sh` detects macOS and uses the
  finger-curl rules for gestures (the trained model needs SDK raw IMU data that Studio does not send).
- Check that data arrives: `python3 scripts/glove_check.py`.

## 3. Run
```bash
./fly.sh --dry-run                         # window opens, nothing is sent; check gestures light up the cards
./fly.sh --mqtt-host <broker-ip>           # real drone (broker the drone listens to, default 192.168.4.2)
./fly.sh --ai --mqtt-host <broker-ip>      # + AI assist panel (needs ai_mux on the drone, see below)
./fly.sh --sim                             # simulator with 3D view, no drone
```
Hold the hand flat, palm down, fingers forward for 2 s to calibrate.

| Gesture (right hand, palm down) | Action |
|---|---|
| thumbs up, hold 1 s (on the ground) | take off |
| open hand, tilt fingers down / up | forward / back (max 0.4 m/s) |
| peace sign, roll hand right / left | turn right / left |
| index only, pointed up / down | up / down a bit (0.15 m/s, max 2 s) |
| fist or pinky, hold 1 s | land |
| Esc key or LAND button | land |

Hand control switches on by itself 5 s after take-off. Keep the DJI remote in hand as the emergency stop.

## 4. Troubleshooting
| Symptom | Fix |
|---|---|
| no window / `No module named _tkinter` | install python-tk (see step 1), rerun `./setup_venv.sh` |
| window but cards stay dark, "no glove" | Studio streaming not on, wrong port (14043) or format (must be JSON v3) |
| gestures not recognised on Mac | recalibrate (button in the window), hand flat for 2 s; Studio needs `--finger-range 3.4` (fly.sh sets it) |
| "NOT CONNECTED" | wrong `--mqtt-host`, broker not running (`brew install mosquitto`), or not on the drone's Wi-Fi |
| `--train` fails on Mac | expected: training needs the Device SDK (Linux) |

## 5. AI assist (optional, VLM obstacle avoidance)
Three parts, all in this repo:
- **cloud**: MQTT broker + `python3 scripts/vlm_service.py --source rtsp://localhost:8554/drone --mqtt-host localhost`
  (camera stream relayed by mediamtx; `export VERDA_INFERENCE_KEY=...`)
- **drone onboard computer**: `pip install "paho-mqtt>=2"` then
  `python3 scripts/ai_mux.py --local-host <broker-ip> --cloud-host <cloud-ip>`
  (add `--out-topic skyhand/shadow_vel` for a first test that does not move the drone)
- **pilot laptop**: `./fly.sh --ai --mqtt-host <broker-ip>`, tick "AI assist" in the window.
Check the model on photos of the obstacle first: `python3 scripts/vlm_avoid_check.py --images <dir>` with files
named `avoid_left_*.jpg`, `avoid_right_*.jpg`, `continue_*.jpg`, `stop_*.jpg`.
