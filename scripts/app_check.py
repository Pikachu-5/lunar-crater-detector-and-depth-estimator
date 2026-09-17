"""Headless Streamlit AppTest checks for the fixes recorded in AUDIT_AFTER.md.

Drives app.py through the synthetic default flow (steps 4, 5, 7, 8) and prints
what the UI shows, so label/placeholder fixes can be verified without a browser.

Run from the project root:
    .venv/Scripts/python scripts/app_check.py
"""

from __future__ import annotations

import os
import sys

from streamlit.testing.v1 import AppTest

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")


def texts(at: AppTest) -> list[str]:
    out = []
    for kind in ("markdown", "caption", "info", "warning", "error", "metric", "text"):
        for el in getattr(at, kind, []):
            val = getattr(el, "value", None)
            if isinstance(val, str):
                out.append(val)
    return out


def main() -> int:
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    print("sidebar sliders:", [(s.label, s.min, s.max, s.value) for s in at.sidebar.slider])
    at.session_state["mission_started"] = True
    at.session_state["current_step"] = 4
    at.run()
    [b for b in at.button if b.label == "Run YOLO Detection"][0].click().run()

    det = at.session_state["detection"]
    print("exceptions step4:", [e.value for e in at.exception])
    print("detection source:", det["source"], "| n:", len(det["detections"]), "| status:", det["status"])
    print("confidences:", [d["confidence"] for d in det["detections"]])
    print("keys in detection dict:", sorted(k for k in det if k != "overlay"))
    print("ground-truth headings:", [t for t in texts(at) if "Ground Truth" in t or "GROUND TRUTH" in t])
    print("log lines:", at.session_state["terminal_logs"])

    for step in (5, 6, 7, 8, 9):
        at.session_state["current_step"] = step
        at.run()
        print(f"--- step {step} exceptions:", [e.value for e in at.exception])
        for t in texts(at):
            if any(k in t for k in ("Confidence", "Obstacles", "uncertainty", "mAP", "not measurable",
                                    "Elevation", "Incidence", "incidence", "tan(", "Path Length")):
                print(f"step {step} text:", " ".join(t.split())[:300])
    print("final log lines:", at.session_state["terminal_logs"][-6:])

    at.session_state["current_step"] = 4
    at.run()
    print("step 4 checkpoint caption:", [t for t in texts(at) if "checkpoint" in t])
    at.session_state["current_step"] = 3
    at.run()
    print("step 3 kernel text:", [" ".join(t.split()) for t in texts(at) if "kernel" in t.lower()])
    at.session_state["current_step"] = 6
    at.run()
    print("step 6 terrain caption:", [t for t in texts(at) if "3D profile" in t])
    at.sidebar.toggle[0].set_value(True)
    at.session_state["current_step"] = 5
    at.run()
    print("step 5 fusion exceptions:", [e.value for e in at.exception])
    print("step 5 fusion text:", [t for t in texts(at) if "view" in t.lower() or "fusion" in t.lower()])
    return 1 if at.exception else 0


if __name__ == "__main__":
    sys.exit(main())
