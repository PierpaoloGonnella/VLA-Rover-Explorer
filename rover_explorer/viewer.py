from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    try:
        import streamlit as st
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Install viewer support with: pip install -e .[viewer]") from exc

    st.set_page_config(page_title="Rover Explorer", layout="wide")
    st.title("Rover Explorer")
    session = Path(st.sidebar.text_input("Session directory", "sessions/latest"))
    latest_path = session / "latest.json"
    if st.sidebar.button("EMERGENCY STOP", type="primary", use_container_width=True):
        session.mkdir(parents=True, exist_ok=True)
        (session / "EMERGENCY_STOP").write_text("stop", encoding="ascii")
        st.error("Emergency stop requested. The runner will send A#0#0#.")
    if not latest_path.exists():
        st.info("Waiting for a cycle record in the selected session directory.")
        return
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    frame = session / latest["annotated_frame"]
    left, right = st.columns([3, 2])
    with left:
        st.image(str(frame), channels="BGR", use_container_width=True)
    with right:
        decision = latest.get("decision", {})
        st.subheader(str(decision.get("action", "STOP")).replace("_", " ").upper())
        st.write(decision.get("reason", ""))
        coverage = latest.get("coverage", {})
        st.metric("Coverage", f"{100*coverage.get('fraction_visited', 0):.1f}%")
        st.metric("Battery", f"{latest.get('battery_mv') or '—'} mV")
        sonar = latest.get("sonar_cm")
        sonar_left = latest.get("sonar_left_cm")
        sonar_right = latest.get("sonar_right_cm")
        st.metric("Ultrasonic front", f"{sonar} cm" if sonar is not None else "—")
        st.caption(
            f"Left: {sonar_left if sonar_left is not None else '—'} cm · "
            f"Right: {sonar_right if sonar_right is not None else '—'} cm · "
            f"Scan: {latest.get('sonar_scan_sequence', 0)}"
        )
        if latest.get("obstacle_blocked"):
            st.error("ULTRASONIC STOP: obstacle ahead")
        st.metric("Cycle latency", f"{latest.get('cycle_latency_seconds', 0):.2f} s")


if __name__ == "__main__":
    main()
