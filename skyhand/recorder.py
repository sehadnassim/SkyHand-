"""Logs every control tick (glove features, gesture, command, drone state) as JSONL.
Teleop demonstrations collected this way can be used to train/imitate drone behaviours."""
import json
import os
import time


class Recorder:
    def __init__(self, out_dir: str = "recordings"):
        os.makedirs(out_dir, exist_ok=True)
        self.path = os.path.join(out_dir, time.strftime("session_%Y%m%d_%H%M%S.jsonl"))
        self._f = open(self.path, "a", buffering=1)

    def log(self, **row):
        row["t"] = time.time()
        self._f.write(json.dumps(row, default=lambda o: getattr(o, "__dict__", str(o))) + "\n")

    def close(self):
        self._f.close()
