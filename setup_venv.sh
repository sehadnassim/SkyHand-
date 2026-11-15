#!/usr/bin/env bash
# Create .venv with everything SkyHand needs (macOS or Linux). Run once:  ./setup_venv.sh
# Then:  source .venv/bin/activate
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
"$PY" -c 'import sys; assert sys.version_info >= (3, 9), "need Python 3.9+ (mac: brew install python)"'
"$PY" -c 'import tkinter' 2>/dev/null || { echo "tkinter missing (the flight window needs it). mac: brew install python-tk@$("$PY" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')  linux: sudo apt install python3-tk"; exit 1; }
"$PY" -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r requirements.txt opencv-python pillow
.venv/bin/python -c 'import cv2, paho.mqtt, requests, PIL, numpy; print("ok: .venv ready  ->  source .venv/bin/activate")'
