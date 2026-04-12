"""Reusable mission-control UI widgets for Streamlit."""

from __future__ import annotations

import time
from datetime import datetime
from typing import Iterable

import streamlit as st


def inject_mission_css() -> None:
    """Inject NASA-style dark HUD theme and scanline aesthetics.

    UI concept:
    A consistent visual language (palette, typography, glow, scanlines) improves
    operator cognition by signaling system state and module boundaries.
    """

    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&display=swap');

        :root {
            --bg: #0d1220;
            --safe: #7dd3fc;
            --warn: #f59e0b;
            --danger: #ef4444;
            --panel: #171e31;
            --text: #dbe7f4;
            --glow-sm: 0 0 8px rgba(125, 211, 252, 0.35);
            --glow-md: 0 0 14px rgba(125, 211, 252, 0.5);
            --transition-fast: 0.16s ease;
            --transition-med: 0.26s ease;
        }

        .stApp {
            background:
                radial-gradient(circle at 15% 10%, rgba(146, 163, 186, 0.2) 0%, transparent 45%),
                radial-gradient(circle at 85% 80%, rgba(125, 211, 252, 0.15) 0%, transparent 40%),
                linear-gradient(180deg, #11182a 0%, #0d1322 55%, #0a101c 100%);
            color: var(--text);
            font-family: "Space Mono", "Courier New", monospace;
            scroll-behavior: smooth;
        }

        html, body {
            scroll-behavior: smooth;
        }

        .scanline-overlay {
            position: fixed;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            background-image: repeating-linear-gradient(
                to bottom,
                rgba(255, 255, 255, 0.02) 0px,
                rgba(255, 255, 255, 0.02) 1px,
                transparent 2px,
                transparent 4px
            );
            opacity: 0.08;
            z-index: 1;
        }

        /* ── Fade-in for main content ── */
        .stMainBlockContainer {
            animation: fadeSlideIn 0.35s ease-out;
            will-change: transform, opacity;
        }

        @keyframes fadeSlideIn {
            from { opacity: 0; transform: translateY(12px); }
            to { opacity: 1; transform: translateY(0); }
        }

        /* ── HUD bar ── */
        .hud-bar {
            border: 1px solid var(--safe);
            box-shadow: var(--glow-sm);
            border-radius: 8px;
            background: rgba(17, 24, 39, 0.72);
            padding: 10px 14px;
            margin-bottom: 10px;
            transition: box-shadow var(--transition-fast);
        }

        .hud-grid {
            display: grid;
            grid-template-columns: 1.4fr 1fr 1fr auto;
            gap: 10px;
            align-items: center;
        }

        .hud-title {
            font-family: "Orbitron", "Space Mono", sans-serif;
            font-size: 1.2rem;
            color: var(--safe);
            letter-spacing: 2px;
            font-weight: 900;
        }

        .hud-small {
            font-size: 0.84rem;
            color: #9de5c1;
        }

        .telemetry-badge {
            border: 1px solid var(--safe);
            color: var(--safe);
            padding: 4px 8px;
            border-radius: 20px;
            font-size: 0.8rem;
            display: inline-block;
            box-shadow: var(--glow-sm);
        }

        .status-circle {
            width: 18px;
            height: 18px;
            border-radius: 50%;
            display: inline-block;
            animation: pulse 1.8s ease-in-out infinite;
            margin-right: 6px;
        }

        @keyframes pulse {
            0% { transform: scale(0.9); opacity: 0.7; }
            50% { transform: scale(1.07); opacity: 1; }
            100% { transform: scale(0.9); opacity: 0.7; }
        }

        /* ── Glow panel ── */
        .glow-panel {
            border: 1px solid var(--safe);
            box-shadow: var(--glow-sm);
            border-radius: 8px;
            padding: 12px;
            background: rgba(17, 24, 39, 0.68);
            margin-bottom: 10px;
            transition: transform var(--transition-fast), box-shadow var(--transition-fast);
            animation: panelFadeIn 0.4s ease-out;
            will-change: transform, box-shadow, opacity;
        }

        .glow-panel:hover {
            transform: translateY(-1px);
            box-shadow: var(--glow-md);
        }

        @keyframes panelFadeIn {
            from { opacity: 0; transform: scale(0.97); }
            to { opacity: 1; transform: scale(1); }
        }

        /* ── Pipeline flow boxes ── */
        .pipeline-box {
            display: inline-block;
            padding: 7px 10px;
            margin: 4px;
            border-radius: 6px;
            border: 1px solid #2f3b52;
            color: #b4c6df;
            font-size: 0.8rem;
            transition: all var(--transition-med);
        }

        .pipeline-complete {
            border-color: var(--safe);
            color: var(--safe);
            box-shadow: var(--glow-sm);
        }

        /* ── Terminal panel ── */
        .terminal-panel {
            border: 1px solid var(--safe);
            box-shadow: var(--glow-sm);
            border-radius: 6px;
            background: #090d14;
            color: var(--safe);
            font-size: 0.8rem;
            line-height: 1.35;
            max-height: 170px;
            overflow-y: auto;
            padding: 10px;
            white-space: pre-wrap;
        }

        .nav-wrap {
            text-align: center;
            margin-top: 8px;
            margin-bottom: 8px;
        }

        /* ── Typewriter ── */
        .typewriter {
            display: inline-block;
            overflow: hidden;
            border-right: 2px solid var(--safe);
            white-space: nowrap;
            animation: typing 4s steps(120, end), caret 1s step-end infinite;
            max-width: 100%;
        }

        @keyframes typing {
            from { width: 0; }
            to { width: 100%; }
        }

        @keyframes caret {
            from, to { border-color: transparent; }
            50% { border-color: var(--safe); }
        }

        /* ── Sidebar ─────────────────────────────────────────── */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1a2234 0%, #131a2a 100%);
            border-right: 1px solid rgba(125, 211, 252, 0.22);
        }

        [data-testid="stSidebar"] * {
            font-family: "Space Mono", "Courier New", monospace !important;
        }

        /* Fix Streamlit sidebar collapse button icon rendering */
        [data-testid="stSidebar"] button[kind="header"] {
            color: var(--safe) !important;
        }

        /* Robustly replace raw Material Icons text labels with explicit arrows */
        [data-testid="collapsedControl"] button,
        [data-testid="stSidebarCollapseButton"] button,
        [data-testid="stSidebar"] button[kind="header"] {
            position: relative;
            color: transparent !important;
            min-width: 2rem;
            transition: background-color var(--transition-fast), box-shadow var(--transition-fast);
        }

        [data-testid="collapsedControl"] button span,
        [data-testid="stSidebarCollapseButton"] button span,
        [data-testid="stSidebar"] button[kind="header"] span {
            display: none !important;
        }

        [data-testid="collapsedControl"] button::before,
        [data-testid="stSidebarCollapseButton"] button::before,
        [data-testid="stSidebar"] button[kind="header"]::before {
            content: "◀";
            font-size: 1.2rem;
            color: var(--safe);
            position: absolute;
            inset: 0;
            display: grid;
            place-items: center;
        }

        [data-testid="collapsedControl"] button::before {
            content: "▶";
        }

        /* Keep sidebar smooth without heavy layout thrash */
        [data-testid="stSidebar"],
        [data-testid="stSidebar"] > div {
            transition: transform var(--transition-med), width var(--transition-med);
        }

        /* Sidebar slider and input styling */
        [data-testid="stSidebar"] .stSlider > div > div {
            background: rgba(125, 211, 252, 0.2) !important;
        }

        [data-testid="stSidebar"] .stSlider [data-testid="stThumbValue"] {
            color: var(--safe) !important;
        }

        /* ── Comparison panel ── */
        .comparison-panel {
            border: 1px solid rgba(0, 255, 159, 0.3);
            border-radius: 10px;
            padding: 16px;
            background: rgba(17, 24, 39, 0.5);
            margin: 8px 0;
            animation: panelFadeIn 0.5s ease-out;
        }

        .metric-card {
            border: 1px solid rgba(0, 255, 159, 0.25);
            border-radius: 8px;
            padding: 12px 16px;
            background: rgba(10, 14, 26, 0.8);
            text-align: center;
            transition: all var(--transition-fast);
        }

        .metric-card:hover {
            border-color: var(--safe);
            box-shadow: var(--glow-sm);
            transform: translateY(-2px);
        }

        .metric-value {
            font-family: "Orbitron", sans-serif;
            font-size: 1.8rem;
            font-weight: 700;
            color: var(--safe);
            line-height: 1.2;
        }

        .metric-label {
            font-size: 0.75rem;
            color: #9de5c1;
            letter-spacing: 1px;
            text-transform: uppercase;
            margin-top: 4px;
        }

        /* ── Formula display ── */
        .formula-block {
            border: 1px solid rgba(125, 211, 252, 0.22);
            border-radius: 8px;
            padding: 16px;
            background: rgba(14, 20, 36, 0.9);
            font-family: "Courier New", monospace;
            color: #dbe7f4;
            font-size: 0.9rem;
            margin: 8px 0;
            line-height: 1.6;
        }

        .formula-block .formula-title {
            font-size: 0.75rem;
            color: #9ec7e7;
            letter-spacing: 1px;
            text-transform: uppercase;
            margin-bottom: 8px;
        }

        .stButton > button {
            border: 1px solid rgba(125, 211, 252, 0.35) !important;
            background: linear-gradient(180deg, rgba(21, 30, 49, 0.9), rgba(14, 21, 36, 0.95)) !important;
            color: #dbe7f4 !important;
            transition: transform var(--transition-fast), border-color var(--transition-fast), box-shadow var(--transition-fast) !important;
        }

        .stButton > button:hover {
            border-color: var(--safe) !important;
            box-shadow: var(--glow-sm) !important;
            transform: translateY(-1px);
        }

        .stButton > button:active {
            transform: translateY(0);
        }

        /* ── Image fade-in ── */
        .stImage {
            animation: imgReveal 0.6s ease-out;
        }

        @keyframes imgReveal {
            from { opacity: 0; filter: brightness(1.25); }
            to { opacity: 1; filter: brightness(1); }
        }

        /* ── Loading glow effect ── */
        .loading-glow {
            border: 2px solid var(--safe);
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            animation: loadPulse 1.5s ease-in-out infinite;
        }

        @keyframes loadPulse {
            0% { box-shadow: 0 0 4px rgba(0, 255, 159, 0.3); }
            50% { box-shadow: 0 0 20px rgba(0, 255, 159, 0.7); }
            100% { box-shadow: 0 0 4px rgba(0, 255, 159, 0.3); }
        }

        /* ── Detection source badge ── */
        .det-badge {
            display: inline-block;
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 0.75rem;
            letter-spacing: 0.5px;
            font-weight: 700;
        }

        .det-badge-yolo {
            background: rgba(0, 255, 159, 0.15);
            border: 1px solid var(--safe);
            color: var(--safe);
        }

        .det-badge-cv {
            background: rgba(255, 179, 0, 0.15);
            border: 1px solid var(--warn);
            color: var(--warn);
        }

        /* ── Smooth progress bar ── */
        .mission-progress {
            height: 4px;
            border-radius: 2px;
            background: rgba(0, 255, 159, 0.15);
            overflow: hidden;
            margin: 4px 0;
        }

        .mission-progress-fill {
            height: 100%;
            border-radius: 2px;
            background: linear-gradient(90deg, var(--safe), #60a5fa);
            box-shadow: 0 0 8px var(--safe);
            transition: width 0.6s ease;
        }
        </style>
        <div class="scanline-overlay"></div>
        """,
        unsafe_allow_html=True,
    )


def mission_status_color(status: str) -> str:
    """Map mission status token to themed hex color."""

    status = status.upper()
    if status == "GREEN":
        return "#7dd3fc"
    if status == "AMBER":
        return "#f59e0b"
    return "#ef4444"


def render_hud(step: int, total_steps: int, mission_status: str) -> None:
    """Render mission HUD header with mission id, clock, and status.

    Args:
        step: Current 1-based step index.
        total_steps: Total step count.
        mission_status: GREEN/AMBER/RED state token.
    """

    status_color = mission_status_color(mission_status)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    st.markdown(
        f"""
        <div class="hud-bar">
          <div class="hud-grid">
            <div>
              <div class="hud-title">DIP</div>
              <div class="hud-small">AUTONOMOUS PLANETARY LANDING SITE ANALYZER</div>
            </div>
            <div class="hud-small">STEP {step:02d} / {total_steps:02d}</div>
            <div class="hud-small">UTC {now}</div>
            <div style="display:flex; align-items:center; justify-content:flex-end; gap:8px;">
              <span class="telemetry-badge">TELEMETRY: NOMINAL</span>
              <span class="status-circle" style="background:{status_color}; box-shadow:0 0 10px {status_color};"></span>
              <span class="hud-small">MISSION STATUS</span>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_glow_title(title: str) -> None:
    """Render section title inside glowing panel frame."""

    st.markdown(f"<div class='glow-panel'><b>{title}</b></div>", unsafe_allow_html=True)


def render_pipeline_flow(current_step: int, total_steps: int, completed: set[int]) -> None:
    """Render flowchart-like pipeline state strip with completion glow.

    Args:
        current_step: Active step index.
        total_steps: Total step count.
        completed: Set of completed step indices.
    """

    labels = [
        "Upload",
        "Preprocess",
        "Detect",
        "Depth",
        "3D Terrain",
        "Score",
        "Path Plan",
        "Report",
    ]
    html = ["<div class='glow-panel'>"]
    for i, label in enumerate(labels, start=1):
        cls = "pipeline-box"
        display_label = label
        if i in completed or i < current_step:
            cls += " pipeline-complete"
            display_label = f"{label} ✓"
        html.append(f"<span class='{cls}'>{display_label}</span>")
        if i < len(labels):
            html.append("<span style='color:#6a7f9f;'>→</span>")
    html.append("</div>")

    st.markdown("".join(html), unsafe_allow_html=True)


def render_typewriter_block(text: str) -> None:
    """Render mission text using typewriter animation style."""

    st.markdown(f"<div class='glow-panel'><span class='typewriter'>{text}</span></div>", unsafe_allow_html=True)


def initialization_animation(step_key: str, label: str) -> None:
    """Show one-time module initialization animation per step.

    UX note:
    Brief initialization cues reinforce stage transitions and make dataflow feel
    procedural instead of opaque.

    Args:
        step_key: Unique key to persist animation state.
        label: Module label shown during initialization.
    """

    init_key = f"_init_{step_key}"
    if not st.session_state.get(init_key, False):
        ph = st.empty()
        with ph.container():
            st.markdown(
                f"<div class='glow-panel'>INITIALIZING MODULE {label}...</div>",
                unsafe_allow_html=True,
            )
            st.progress(100)
            time.sleep(0.08)
        st.session_state[init_key] = True
        ph.empty()


def render_terminal_log(log_lines: Iterable[str], max_lines: int = 10) -> None:
    """Render scrolling terminal-style log panel.

    Args:
        log_lines: Collection of log strings.
        max_lines: Number of trailing lines to show.
    """

    lines = list(log_lines)[-max_lines:]
    payload = "\n".join(lines) if lines else "[SYS] >> Awaiting module execution..."
    st.markdown("<div class='terminal-panel'>" + payload + "</div>", unsafe_allow_html=True)


def render_metric_card(value: str, label: str) -> None:
    """Render a single metric card with animated value display.

    Args:
        value: The metric value to display prominently.
        label: Description label below the value.
    """

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_formula(title: str, formula_html: str) -> None:
    """Render a styled formula block.

    Args:
        title: Formula section title.
        formula_html: HTML content of the formula.
    """

    st.markdown(
        f"""
        <div class="formula-block">
            <div class="formula-title">{title}</div>
            {formula_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_detection_badge(source: str) -> None:
    """Render a styled detection source badge.

    Args:
        source: Detection source (yolo, cv-hybrid, etc.).
    """

    if "yolo" in source.lower():
        cls = "det-badge det-badge-yolo"
        label = "YOLO11m CRATER DETECTOR"
    else:
        cls = "det-badge det-badge-cv"
        label = "CV HYBRID (HOUGH+LoG)"

    st.markdown(f"<span class='{cls}'>{label}</span>", unsafe_allow_html=True)
