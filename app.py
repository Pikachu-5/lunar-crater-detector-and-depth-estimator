"""Autonomous Planetary Landing Site Analyzer mission-control Streamlit app."""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from modules.depth import estimate_crater_depths, fuse_depth_estimates
from modules.detector import detect_craters, detect_craters_cv, draw_detections
from modules.pathfinder import draw_paths_on_map, plan_descent_paths
from modules.preprocess import compute_histogram, image_stats, preprocess_pipeline
from modules.reporter import build_mission_pdf, crater_rows_to_csv, image_to_png_bytes
from modules.scorer import annotate_hazard_map, build_safety_gauge, build_score_map, score_landing_safety
from modules.terrain3d import build_depth_map, make_contour_figure, make_heatmap_figure, make_surface_figure
from utils.synthetic import generate_secondary_solar_view, generate_synthetic_lunar_surface
from utils.ui_components import (
    initialization_animation,
    inject_mission_css,
    render_detection_badge,
    render_formula,
    render_hud,
    render_metric_card,
    render_pipeline_flow,
    render_terminal_log,
    render_typewriter_block,
)

try:
    from streamlit_image_comparison import image_comparison
    IMAGE_COMPARISON_ERROR = ""
except Exception as exc:  # pragma: no cover
    image_comparison = None
    IMAGE_COMPARISON_ERROR = str(exc)


TOTAL_STEPS = 9
MISSION_ID = "DIP"


STEP_TITLES = {
    1: "MISSION BRIEFING",
    2: "RAW IMAGE ACQUISITION",
    3: "PREPROCESSING & ENHANCEMENT",
    4: "CRATER DETECTION (YOLO11m + CV)",
    5: "PHOTOMETRIC DEPTH ESTIMATION",
    6: "3D TERRAIN RECONSTRUCTION",
    7: "LANDING SAFETY SCORING",
    8: "OPTIMAL DESCENT PATH PLANNING",
    9: "MISSION REPORT EXPORT",
}


def init_state() -> None:
    """Initialize application session state for full pipeline continuity.

    State design note:
    Streamlit reruns the script on every interaction. Persisting each module's
    artifacts in session state ensures users can inspect all intermediate steps
    without recomputation ambiguity.
    """

    if "current_step" not in st.session_state:
        st.session_state.current_step = 1

    if "completed_steps" not in st.session_state:
        st.session_state.completed_steps = set()

    if "mission_started" not in st.session_state:
        st.session_state.mission_started = False

    if "terminal_logs" not in st.session_state:
        st.session_state.terminal_logs = []

    if "raw_image" not in st.session_state:
        synth = generate_synthetic_lunar_surface(size=512)
        st.session_state.raw_image = synth["image"]
        st.session_state.synthetic_meta = synth
        st.session_state.image_name = "SYNTHETIC_LUNAR_FEED"
        st.session_state.file_size_bytes = None

    if "preprocess" not in st.session_state:
        st.session_state.preprocess = None

    if "detection" not in st.session_state:
        st.session_state.detection = None

    if "cv_detection" not in st.session_state:
        st.session_state.cv_detection = None

    if "depth" not in st.session_state:
        st.session_state.depth = None

    if "fusion" not in st.session_state:
        st.session_state.fusion = None

    if "terrain" not in st.session_state:
        st.session_state.terrain = None

    if "scoring" not in st.session_state:
        st.session_state.scoring = None

    if "paths" not in st.session_state:
        st.session_state.paths = None

    if "hazard_map" not in st.session_state:
        st.session_state.hazard_map = None

    if "step_logs" not in st.session_state:
        st.session_state.step_logs = set()

    if "mission_status" not in st.session_state:
        st.session_state.mission_status = "GREEN"

    if "prev_step" not in st.session_state:
        st.session_state.prev_step = st.session_state.current_step

    # Preprocessing parameters
    if "pp_clip_limit" not in st.session_state:
        st.session_state.pp_clip_limit = 2.2
    if "pp_grid_size" not in st.session_state:
        st.session_state.pp_grid_size = 8
    if "pp_sigma" not in st.session_state:
        st.session_state.pp_sigma = 1.2

    if "terrain_profile" not in st.session_state:
        st.session_state.terrain_profile = "High Fidelity (512 MB)"


def append_log(message: str) -> None:
    """Append timestamped terminal message to mission log stream.

    Args:
        message: Human-readable telemetry line.
    """

    ts = datetime.now().strftime("%H:%M:%S")
    st.session_state.terminal_logs.append(f"[{ts}] {message}")


def log_once(key: str, message: str) -> None:
    """Write a log line only once per key to avoid rerun spam.

    Args:
        key: Unique logging key.
        message: Log text.
    """

    if key in st.session_state.step_logs:
        return
    append_log(message)
    st.session_state.step_logs.add(key)


def decode_upload_to_gray(uploaded_file) -> tuple[np.ndarray, int]:
    """Decode uploaded image bytes into grayscale array.

    Computer vision note:
    Grayscale processing simplifies histogram analysis, thresholding, and shape
    detection while preserving essential crater luminance cues.

    Args:
        uploaded_file: Streamlit uploaded file object.

    Returns:
        Tuple of (grayscale image, file size bytes).
    """

    data = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Unable to decode uploaded image")
    return img, len(data)


def reset_downstream(start_step: int) -> None:
    """Clear downstream cached products when upstream data changes.

    Args:
        start_step: First step index that should be considered stale.
    """

    if start_step <= 3:
        st.session_state.preprocess = None
    if start_step <= 4:
        st.session_state.detection = None
        st.session_state.cv_detection = None
    if start_step <= 5:
        st.session_state.depth = None
        st.session_state.fusion = None
    if start_step <= 6:
        st.session_state.terrain = None
    if start_step <= 7:
        st.session_state.scoring = None
        st.session_state.hazard_map = None
    if start_step <= 8:
        st.session_state.paths = None


def _on_step_change() -> None:
    """Handle lightweight transition effects after step changes.

    UX note:
    Streamlit reruns can preserve the previous scroll offset, which feels buggy
    in wizard flows. A tiny script nudges the viewport to top smoothly.
    """

    if st.session_state.prev_step != st.session_state.current_step:
        st.session_state.prev_step = st.session_state.current_step
        components.html(
            """
            <script>
            const root = window.parent;
            if (root && typeof root.scrollTo === 'function') {
                root.scrollTo({ top: 0, behavior: 'smooth' });
            }
            </script>
            """,
            height=0,
        )


def mission_status_from_findings() -> str:
    """Derive mission status color from latest safety analysis artifacts.

    Returns:
        Status token GREEN, AMBER, or RED.
    """

    scoring = st.session_state.scoring
    if not scoring:
        return "GREEN"

    summary = scoring.get("summary", {})
    safe = int(summary.get("safe", 0))
    caution = int(summary.get("caution", 0))
    hazard = int(summary.get("hazard", 0))

    if hazard > max(safe, 0):
        return "RED"
    if hazard > 0 or caution > 0:
        return "AMBER"
    return "GREEN"


def histogram_figure(image: np.ndarray, title: str) -> go.Figure:
    """Build mission-themed histogram figure for intensity distributions.

    Args:
        image: Grayscale image.
        title: Chart title.

    Returns:
        Plotly bar chart figure.
    """

    hist = compute_histogram(image, bins=64)
    fig = go.Figure(
        data=go.Bar(x=hist["centers"], y=hist["counts"], marker=dict(color="#7dd3fc", line=dict(width=0)))
    )
    fig.update_layout(
        title=title,
        paper_bgcolor="#0a0e1a",
        plot_bgcolor="#0a0e1a",
        xaxis=dict(title="Intensity", color="#7dd3fc"),
        yaxis=dict(title="Normalized Frequency", color="#7dd3fc"),
        font=dict(color="#dbe7f4", family="Courier New"),
        margin=dict(l=12, r=12, t=40, b=10),
        height=260,
    )
    return fig


def ensure_preprocess() -> None:
    """Ensure preprocessing outputs exist for current raw image and parameters."""

    clip = st.session_state.pp_clip_limit
    grid = st.session_state.pp_grid_size
    sigma = st.session_state.pp_sigma

    if st.session_state.preprocess is None:
        st.session_state.preprocess = preprocess_pipeline(
            st.session_state.raw_image,
            clip_limit=clip,
            tile_grid_size=(grid, grid),
            sigma=sigma,
        )


def run_detection() -> None:
    """Execute YOLO crater detection and cache results."""

    ensure_preprocess()
    smoothed = st.session_state.preprocess["smoothed"]

    hint_craters = None
    if (
        st.session_state.image_name == "SYNTHETIC_LUNAR_FEED"
        and st.session_state.get("synthetic_meta")
        and st.session_state.synthetic_meta.get("craters")
    ):
        hint_craters = st.session_state.synthetic_meta["craters"]

    detection = detect_craters(
        smoothed,
        conf_threshold=0.35,
        hint_craters=hint_craters,
    )

    overlay = draw_detections(st.session_state.raw_image, detection["detections"])
    detection["overlay"] = overlay
    st.session_state.detection = detection
    st.session_state.completed_steps.add(4)

    log_once(
        "step4_detected",
        "[DETECTOR] >> "
        f"Source: {detection['source']} | Craters detected: {len(detection['detections'])} "
        f"| mAP@0.5: {detection['map50_proxy']:.3f} | Time: {detection['elapsed_s']:.2f}s",
    )
    append_log(f"[DETECTOR-DETAIL] >> {detection['status']}")


def run_cv_detection() -> None:
    """Execute CV-hybrid crater detection for comparison display."""

    ensure_preprocess()
    smoothed = st.session_state.preprocess["smoothed"]

    cv_det = detect_craters_cv(smoothed)
    cv_overlay = draw_detections(
        st.session_state.raw_image,
        cv_det["detections"],
        color=(255, 179, 0),
    )
    cv_det["overlay"] = cv_overlay
    st.session_state.cv_detection = cv_det

    log_once(
        "step4_cv",
        f"[CV-HYBRID] >> Craters: {len(cv_det['detections'])} | Time: {cv_det['elapsed_s']:.2f}s",
    )


def ensure_depth(theta_deg: float, pixel_scale_m: float) -> bool:
    """Run crater depth estimation from current detections.

    Args:
        theta_deg: Solar incidence angle.
        pixel_scale_m: Pixel-to-meter scale.
    """

    if st.session_state.detection is None:
        return False

    rows = estimate_crater_depths(
        image=st.session_state.raw_image,
        detections=st.session_state.detection["detections"],
        solar_incidence_angle_deg=theta_deg,
        pixel_scale_m=pixel_scale_m,
    )
    st.session_state.depth = rows
    st.session_state.completed_steps.add(5)
    return True


def _terrain_profile_target_size(profile: str, shape: tuple[int, int]) -> int:
    """Map selected terrain profile to downsample target size.

    Args:
        profile: User-selected fidelity profile.
        shape: Original depth-map shape (height, width).

    Returns:
        Target max dimension for 3D surface mesh.
    """

    h, w = shape
    max_dim = max(h, w)
    if profile == "Balanced (256 MB)":
        return int(np.clip(min(max_dim, 520), 320, 520))
    if profile == "Max Fidelity (1 GB)":
        return int(np.clip(min(max_dim, 960), 420, 960))
    if profile == "Adaptive":
        return int(np.clip(min(max_dim, 760), 360, 760))
    return int(np.clip(min(max_dim, 720), 380, 720))


def _estimate_surface_memory_mb(target_size: int) -> float:
    """Estimate memory footprint for Plotly 3D mesh buffers.

    Args:
        target_size: Approximate max dimension of rendered surface grid.

    Returns:
        Estimated memory usage in MB.
    """

    points = float(target_size * target_size)
    bytes_estimate = points * 48.0
    return bytes_estimate / (1024.0 * 1024.0)


def ensure_terrain(profile: str) -> None:
    """Build terrain depth-map artifacts from crater depth table."""

    if st.session_state.depth is None:
        return

    depth_map = build_depth_map(st.session_state.raw_image.shape, st.session_state.depth["rows"])
    target_size = _terrain_profile_target_size(profile, depth_map.shape)
    est_mb = _estimate_surface_memory_mb(target_size)
    st.session_state.terrain = {
        "depth_map": depth_map,
        "heatmap_fig": make_heatmap_figure(depth_map),
        "surface_fig": make_surface_figure(depth_map, downsample=True, target_size=target_size),
        "contour_fig": make_contour_figure(depth_map),
        "surface_target_size": target_size,
        "surface_memory_est_mb": est_mb,
        "surface_profile": profile,
    }
    st.session_state.completed_steps.add(6)


def ensure_scoring(td: float, gear_span_m: float, density_radius_px: int, pixel_scale_m: float) -> None:
    """Compute crater safety scores and annotated hazard map.

    Args:
        td: Depth safety threshold.
        gear_span_m: Landing gear span in meters.
        density_radius_px: Crater density radius in pixels.
        pixel_scale_m: Pixel-to-meter scale factor.
    """

    if st.session_state.depth is None:
        return

    scoring = score_landing_safety(
        depth_rows=st.session_state.depth["rows"],
        depth_threshold_m=td,
        landing_gear_span_m=gear_span_m,
        density_radius_px=density_radius_px,
        pixel_scale_m=pixel_scale_m,
    )

    hazard_map = annotate_hazard_map(st.session_state.raw_image, scoring["rows"])
    score_map = build_score_map(st.session_state.raw_image.shape, scoring["rows"])

    scoring["score_map"] = score_map
    st.session_state.scoring = scoring
    st.session_state.hazard_map = hazard_map
    st.session_state.completed_steps.add(7)


def ensure_paths(pixel_scale_m: float) -> None:
    """Plan descent routes over the safety score map.

    Args:
        pixel_scale_m: Pixel-to-meter scale.
    """

    if not st.session_state.scoring:
        return

    paths = plan_descent_paths(
        score_map=st.session_state.scoring["score_map"],
        scored_rows=st.session_state.scoring["rows"],
        pixel_scale_m=pixel_scale_m,
    )

    overlay = draw_paths_on_map(
        image=st.session_state.hazard_map,
        primary_path=paths["primary_path"],
        alternative_paths=paths["alternative_paths"],
    )
    paths["overlay"] = overlay

    st.session_state.paths = paths
    st.session_state.completed_steps.add(8)


def render_sidebar() -> dict[str, Any]:
    """Render mission parameter panel and return control values.

    Returns:
        Dictionary of mission parameter values.
    """

    with st.sidebar:
        st.markdown(
            """
            <div style="text-align:center; padding: 8px 0 16px 0;">
                 <div style="font-family: 'Orbitron', sans-serif; font-size: 1.1rem;
                     color: #7dd3fc; letter-spacing: 2px; font-weight: 900;">
                    MISSION CONTROL
                </div>
                <div style="height: 2px; background: linear-gradient(90deg, transparent,
                     #7dd3fc, transparent); margin-top: 8px;"></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### Mission Parameters")

        theta = st.slider("Solar Incidence Angle θ", min_value=10, max_value=80, value=35, step=1)
        st.caption("Controls illumination geometry for shadow-based depth. Larger angles usually increase estimated depth.")

        td = st.slider("Depth Safety Threshold Td (m)", min_value=0.5, max_value=5.0, value=1.8, step=0.1)
        st.caption("Maximum crater depth considered landing-safe. Lower values make safety scoring stricter.")

        gear_span = st.slider("Landing Gear Span (m)", min_value=1.0, max_value=5.0, value=2.6, step=0.1)
        st.caption("Reference footprint used to penalize craters wider than lander support geometry.")

        density_radius = st.slider("Crater Density Radius (px)", min_value=20, max_value=200, value=85, step=5)
        st.caption("Neighborhood radius for local hazard density. Larger radius captures broader rough-terrain risk.")

        pixel_scale = st.slider("Pixel Scale (m/px)", min_value=0.2, max_value=4.0, value=1.0, step=0.1)
        st.caption("Converts pixel distances to meters for depth, path length, and safety calculations.")

        with st.expander("What this controls"):
            st.markdown(
                """
                - Solar incidence angle changes shadow-to-depth conversion sensitivity.
                - Depth threshold shifts SAFE vs HAZARD boundaries.
                - Gear span affects diameter-based landing feasibility.
                - Density radius controls how strongly crater clustering is penalized.
                - Pixel scale affects all metric outputs in meters.
                """
            )

        st.markdown("---")
        terrain_profile = st.selectbox(
            "3D Terrain Memory Profile",
            options=["Balanced (256 MB)", "High Fidelity (512 MB)", "Max Fidelity (1 GB)", "Adaptive"],
            index=["Balanced (256 MB)", "High Fidelity (512 MB)", "Max Fidelity (1 GB)", "Adaptive"].index(
                st.session_state.terrain_profile
                if st.session_state.terrain_profile in {"Balanced (256 MB)", "High Fidelity (512 MB)", "Max Fidelity (1 GB)", "Adaptive"}
                else "High Fidelity (512 MB)"
            ),
            help="Raises 3D mesh capacity beyond the legacy 200-size render path. Higher fidelity needs more RAM/VRAM.",
        )
        if terrain_profile != st.session_state.terrain_profile:
            st.session_state.terrain_profile = terrain_profile
            st.session_state.terrain = None

        if st.session_state.file_size_bytes and st.session_state.file_size_bytes > 200 * 1024 * 1024:
            st.info("Large input detected (>200 MB). High Fidelity (512 MB) or Max Fidelity (1 GB) is recommended.")

        st.markdown("---")
        enable_fusion = st.toggle("ENABLE MULTI-ANGLE FUSION", value=False)

        st.markdown("---")
        if st.button("↻ Regenerate Synthetic Surface", use_container_width=True):
            synth = generate_synthetic_lunar_surface(size=512)
            st.session_state.raw_image = synth["image"]
            st.session_state.synthetic_meta = synth
            st.session_state.image_name = "SYNTHETIC_LUNAR_FEED"
            st.session_state.file_size_bytes = None
            reset_downstream(start_step=3)
            append_log("[SYNTH] >> Regenerated synthetic lunar surface with crater metadata")

    return {
        "theta": theta,
        "td": td,
        "gear_span": gear_span,
        "density_radius": density_radius,
        "pixel_scale": pixel_scale,
        "enable_fusion": enable_fusion,
        "terrain_profile": terrain_profile,
    }


def step_01_briefing() -> None:
    """Render mission briefing, pipeline flow, uploader, and launch control."""

    initialization_animation("step01", "BRIEFING")
    st.markdown(f"## {STEP_TITLES[1]}")

    render_typewriter_block(
        "Objective: identify safe lunar touchdown zones by exposing every stage of image enhancement, crater detection, depth inference, terrain reconstruction, risk scoring, and descent path optimization."
    )

    render_pipeline_flow(st.session_state.current_step, TOTAL_STEPS, st.session_state.completed_steps)

    uploader = st.file_uploader(
        "Upload satellite image (optional; synthetic feed is loaded by default)",
        type=["png", "jpg", "jpeg", "tif"],
        key="primary_upload",
    )

    if uploader is not None:
        try:
            img, size_bytes = decode_upload_to_gray(uploader)
            st.session_state.raw_image = img
            st.session_state.image_name = uploader.name
            st.session_state.file_size_bytes = size_bytes
            st.session_state.synthetic_meta = None
            reset_downstream(start_step=3)
            append_log(f"[UPLOAD] >> New telemetry image acquired: {uploader.name}")
        except Exception as exc:
            st.error(f"Upload failed: {exc}")

    st.image(st.session_state.raw_image, caption=f"Active Feed: {st.session_state.image_name}", use_container_width=True)

    if st.button("LAUNCH MISSION ►", use_container_width=True):
        st.session_state.mission_started = True
        st.session_state.completed_steps.add(1)
        st.session_state.current_step = 2
        append_log("[MISSION] >> Launch confirmed. Transitioning to RAW IMAGE ACQUISITION")
        st.rerun()


def step_02_raw_acquisition() -> None:
    """Render raw telemetry feed, stats table, and histogram."""

    initialization_animation("step02", "ACQUISITION")
    st.markdown(f"## {STEP_TITLES[2]}")

    image = st.session_state.raw_image
    stats = image_stats(image, st.session_state.file_size_bytes)

    col1, col2 = st.columns([1.5, 1.0])
    with col1:
        st.markdown("### RAW TELEMETRY FEED")
        st.image(image, use_container_width=True)
    with col2:
        st.markdown("### Instrument Readouts")
        st.dataframe(pd.DataFrame([stats]), use_container_width=True, hide_index=True)

    st.plotly_chart(histogram_figure(image, "Raw Pixel Intensity Histogram"), use_container_width=True)

    log_once(
        "step2_acq",
        "[ACQUISITION] >> "
        f"Image loaded: {stats['width_px']}x{stats['height_px']}px | "
        f"Mean intensity: {stats['mean_intensity']:.1f} | Noise σ: {stats['std_intensity']:.1f}",
    )
    st.session_state.completed_steps.add(2)


def _comparison_widget(before: np.ndarray, after: np.ndarray) -> None:
    """Render before/after comparison using component or fallback toggle.

    Args:
        before: Baseline image.
        after: Processed image.
    """

    st.markdown("### Before / After Comparison")
    if image_comparison is not None:
        try:
            image_comparison(
                img1=before,
                img2=after,
                label1="RAW",
                label2="SMOOTHED",
                width=720,
                starting_position=40,
            )
            return
        except Exception as exc:
            append_log(f"[COMPARE] >> Interactive comparison fallback engaged: {exc}")

    reason = IMAGE_COMPARISON_ERROR if IMAGE_COMPARISON_ERROR else "component unavailable at runtime"
    st.info(f"Interactive swipe unavailable ({reason}). Using resilient comparison mode.")

    c1, c2 = st.columns(2)
    with c1:
        st.image(before, caption="RAW", use_container_width=True)
    with c2:
        st.image(after, caption="SMOOTHED", use_container_width=True)

    blend = st.slider("Blend RAW ↔ SMOOTHED", min_value=0, max_value=100, value=50, step=1)
    alpha = blend / 100.0
    blended = cv2.addWeighted(before, 1.0 - alpha, after, alpha, 0.0)
    st.image(blended, caption=f"Blend Preview ({blend}% smoothed)", use_container_width=True)


def step_03_preprocess() -> None:
    """Render preprocessing visuals with interactive CLAHE/blur controls."""

    initialization_animation("step03", "PREPROCESS")
    st.markdown(f"## {STEP_TITLES[3]}")

    # ── Interactive parameter controls ──
    st.markdown("### Parameter Controls")
    st.caption("Tune local contrast and smoothing here. These settings directly affect crater edge clarity and detector confidence.")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        grid_size = st.slider(
            "CLAHE Grid Size",
            min_value=2,
            max_value=16,
            value=st.session_state.pp_grid_size,
            step=1,
            key="clahe_grid_slider",
            help="Tile size for local histogram equalization. Smaller = more local contrast.",
        )
    with col_b:
        clip_limit = st.slider(
            "CLAHE Clip Limit",
            min_value=1.0,
            max_value=5.0,
            value=st.session_state.pp_clip_limit,
            step=0.1,
            key="clahe_clip_slider",
            help="Contrast amplification limit. Higher = more contrast, more noise.",
        )
    with col_c:
        sigma = st.slider(
            "Gaussian Sigma (σ)",
            min_value=0.5,
            max_value=4.0,
            value=st.session_state.pp_sigma,
            step=0.1,
            key="gaussian_sigma_slider",
            help="Blur strength. Higher σ = smoother but loses fine detail.",
        )

    # Check if parameters changed, reprocess if needed
    params_changed = (
        grid_size != st.session_state.pp_grid_size
        or abs(clip_limit - st.session_state.pp_clip_limit) > 0.01
        or abs(sigma - st.session_state.pp_sigma) > 0.01
    )

    if params_changed:
        st.session_state.pp_grid_size = grid_size
        st.session_state.pp_clip_limit = clip_limit
        st.session_state.pp_sigma = sigma
        st.session_state.preprocess = None
        reset_downstream(start_step=4)

    # Run preprocessing with current parameters
    st.session_state.preprocess = preprocess_pipeline(
        st.session_state.raw_image,
        clip_limit=clip_limit,
        tile_grid_size=(grid_size, grid_size),
        sigma=sigma,
    )
    pp = st.session_state.preprocess

    # ── Formula display ──
    render_formula(
        "CLAHE — Contrast Limited Adaptive Histogram Equalization",
        f"""
        Grid: {grid_size}×{grid_size} tiles &nbsp;|&nbsp; Clip limit: {clip_limit:.1f}<br/>
        Per-tile CDF: <code>CDF(i) = Σ clipped_hist(j) / N</code><br/>
        Bilinear interpolation between tile CDFs eliminates block artifacts.
        """,
    )

    render_formula(
        "Gaussian Smoothing Kernel",
        f"""
        <code>G(x,y) = (1 / 2πσ²) × exp(-(x² + y²) / 2σ²)</code><br/>
        σ = {sigma:.1f} px &nbsp;→&nbsp;
        kernel ≈ {max(3, int(round(sigma * 4.5)) | 1)}×{max(3, int(round(sigma * 4.5)) | 1)} px
        """,
    )

    # ── Live preview of all three stages ──
    labels = [
        ("[FEED-A: RAW]", pp["raw"]),
        ("[FEED-B: CLAHE ENHANCED]", pp["enhanced"]),
        ("[FEED-C: SMOOTHED]", pp["smoothed"]),
    ]

    cols = st.columns(3)
    for c, (label, img) in zip(cols, labels):
        with c:
            st.markdown(f"#### {label}")
            st.image(img, use_container_width=True)
            st.plotly_chart(histogram_figure(img, f"Histogram — {label}"), use_container_width=True)

    _comparison_widget(pp["raw"], pp["smoothed"])

    p = pp["params"]
    log_once(
        f"step3_pre_{grid_size}_{clip_limit}_{sigma}",
        f"[PREPROCESS] >> CLAHE clip={p['clip_limit']} tile={p['tile_grid_size']} | Gaussian σ={p['sigma']}",
    )
    st.session_state.completed_steps.add(3)


def step_04_detection() -> None:
    """Render crater detection with dual YOLO vs CV comparison."""

    initialization_animation("step04", "YOLO11m")
    st.markdown(f"## {STEP_TITLES[4]}")

    st.caption("Run detectors manually to compare speed and output quality. No auto-run occurs on step transition.")

    by, bc = st.columns(2)
    with by:
        if st.button("Run YOLO Detection", use_container_width=True):
            ensure_preprocess()
            with st.spinner("Running YOLO11m crater detection..."):
                run_detection()
    with bc:
        if st.button("Run CV Detection", use_container_width=True):
            ensure_preprocess()
            with st.spinner("Running CV hybrid detection..."):
                run_cv_detection()

    detection = st.session_state.detection
    cv_detection = st.session_state.cv_detection

    if detection is None and cv_detection is None:
        st.warning("No detections yet. Use the YOLO and CV buttons above to start analysis.")
        return

    # ── Detection comparison header ──
    yolo_count = len(detection["detections"]) if detection else 0
    cv_count = len(cv_detection["detections"]) if cv_detection else 0

    st.markdown("### Detection Results")

    # Metric cards
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        render_metric_card(str(yolo_count), "YOLO CRATERS")
    with m2:
        render_metric_card(str(cv_count), "CV CRATERS")
    with m3:
        render_metric_card(f"{detection['elapsed_s']:.2f}s" if detection else "—", "YOLO TIME")
    with m4:
        render_metric_card(
            f"{cv_detection['elapsed_s']:.2f}s" if cv_detection else "—",
            "CV TIME",
        )

    if detection and cv_detection:
        yolo_t = float(detection["elapsed_s"])
        cv_t = float(cv_detection["elapsed_s"])
        faster = "YOLO" if yolo_t <= cv_t else "CV"
        delta = abs(yolo_t - cv_t)
        ratio = (max(yolo_t, cv_t) / max(min(yolo_t, cv_t), 1e-6))
        st.markdown(
            f"""
            <div class='glow-panel'>
                <b>Speed Comparison</b><br/>
                Faster: <span style='color:#7dd3fc'>{faster}</span> &nbsp;|&nbsp;
                Delta: {delta:.2f}s &nbsp;|&nbsp;
                Ratio: {ratio:.2f}x
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Side-by-side detection overlays ──
    st.markdown("### Detection Overlays")
    col_y, col_c = st.columns(2)

    with col_y:
        render_detection_badge("yolo")
        if detection:
            st.image(detection["overlay"], caption=f"YOLO11m — {yolo_count} craters", use_container_width=True)
        else:
            st.info("YOLO detection not yet run.")

    with col_c:
        render_detection_badge("cv-hybrid")
        if cv_detection and cv_detection.get("overlay") is not None:
            st.image(cv_detection["overlay"], caption=f"CV Hybrid — {cv_count} craters", use_container_width=True)
        else:
            st.info("CV detection not yet run.")

    # ── YOLO detection table (primary) ──
    if detection and yolo_count > 0:
        st.markdown("### YOLO Detection Table (used for downstream pipeline)")
        df = pd.DataFrame(detection["detections"])
        st.dataframe(
            df[["crater_id", "x1", "y1", "x2", "y2", "confidence", "diameter_px"]],
            use_container_width=True,
            hide_index=True,
        )

        fig = go.Figure(data=go.Bar(x=df["crater_id"], y=df["confidence"], marker_color="#7dd3fc"))
        fig.update_layout(
            title="YOLO Confidence Score Distribution",
            paper_bgcolor="#0a0e1a",
            plot_bgcolor="#0a0e1a",
            font=dict(color="#dbe7f4", family="Courier New"),
            yaxis=dict(range=[0, 1]),
        )
        st.plotly_chart(fig, use_container_width=True)


def step_05_depth(params: dict[str, Any]) -> None:
    """Render crater depth estimation and optional multi-angle fusion."""

    initialization_animation("step05", "DEPTH")
    st.markdown(f"## {STEP_TITLES[5]}")

    if st.session_state.detection is None:
        st.warning("YOLO detections are required before depth estimation.")
        if st.button("Run YOLO Detection Now", key="depth_run_yolo", use_container_width=True):
            ensure_preprocess()
            with st.spinner("Running YOLO11m crater detection..."):
                run_detection()
            st.rerun()
        return

    ensure_depth(theta_deg=params["theta"], pixel_scale_m=params["pixel_scale"])
    depth = st.session_state.depth
    if depth is None or len(depth["rows"]) == 0:
        st.warning("Depth module could not estimate crater depths from current detections.")
        return

    log_once(
        f"depth_formula_{params['theta']}_{params['pixel_scale']}",
        f"[DEPTH] >> Formula: {depth['formula']} | theta={params['theta']} deg | pixel_scale={params['pixel_scale']} m/px",
    )

    st.markdown("### ROI Diagnostic Viewer")
    for roi_bundle in depth["rois"][:8]:
        with st.expander(f"Crater {roi_bundle['crater_id']} ROI Panels", expanded=False):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.image(roi_bundle["original"], caption="Original ROI", use_container_width=True)
            with c2:
                st.image(roi_bundle["mask"], caption="Otsu Shadow Mask", use_container_width=True)
            with c3:
                st.image(roi_bundle["annotated"], caption="Shadow Arrow Annotation", use_container_width=True)

    df = pd.DataFrame(depth["rows"])
    st.dataframe(
        df[["crater_id", "shadow_length_px", "solar_angle_deg", "depth_m", "slope_estimate_deg"]],
        use_container_width=True,
        hide_index=True,
    )

    fig = go.Figure(data=go.Bar(x=df["crater_id"], y=df["depth_m"], marker_color="#f59e0b"))
    fig.update_layout(
        title="Crater Depth Estimates",
        paper_bgcolor="#0a0e1a",
        plot_bgcolor="#0a0e1a",
        font=dict(color="#dbe7f4", family="Courier New"),
        yaxis_title="Depth (m)",
    )
    st.plotly_chart(fig, use_container_width=True)

    if params["enable_fusion"]:
        st.markdown("### Multi-Angle Depth Fusion")

        second_image = None
        if st.session_state.image_name == "SYNTHETIC_LUNAR_FEED":
            second_image = generate_secondary_solar_view(st.session_state.raw_image)
        else:
            second_upload = st.file_uploader(
                "Upload second-angle image for fusion",
                type=["png", "jpg", "jpeg", "tif"],
                key="fusion_upload",
            )
            if second_upload is not None:
                try:
                    second_image, _ = decode_upload_to_gray(second_upload)
                except Exception as exc:
                    st.warning(f"Secondary upload failed: {exc}")

        if second_image is not None:
            second_pp = preprocess_pipeline(second_image)
            second_hint_craters = None
            if st.session_state.image_name == "SYNTHETIC_LUNAR_FEED" and st.session_state.get("synthetic_meta"):
                second_hint_craters = st.session_state.synthetic_meta.get("craters")

            if second_hint_craters:
                second_det = detect_craters(
                    second_pp["smoothed"],
                    conf_threshold=0.35,
                    hint_craters=second_hint_craters,
                )
            else:
                second_det = detect_craters(second_pp["smoothed"], conf_threshold=0.35)

            second_depth = estimate_crater_depths(
                image=second_image,
                detections=second_det["detections"],
                solar_incidence_angle_deg=float(np.clip(params["theta"] + 15, 10, 80)),
                pixel_scale_m=params["pixel_scale"],
            )

            fusion = fuse_depth_estimates(depth["rows"], second_depth["rows"], iou_threshold=0.25)
            st.session_state.fusion = fusion

            if len(fusion["rows"]) > 0:
                fdf = pd.DataFrame(fusion["rows"])
                st.dataframe(fdf, use_container_width=True, hide_index=True)
                log_once(
                    f"fusion_{params['theta']}",
                    f"[FUSION] >> Depth uncertainty reduced by {fusion['uncertainty_reduction_pct']:.2f}%",
                )
            else:
                st.info("Fusion attempted but no crater correspondences passed IoU threshold.")
        else:
            st.info("Enable fusion by uploading a second-angle image or using synthetic mode.")


def step_06_terrain(params: dict[str, Any]) -> None:
    """Render heatmap, contour, and interactive 3D terrain surface."""

    initialization_animation("step06", "TERRAIN")
    st.markdown(f"## {STEP_TITLES[6]}")

    if st.session_state.depth is None or len(st.session_state.depth.get("rows", [])) == 0:
        if not ensure_depth(theta_deg=params["theta"], pixel_scale_m=params["pixel_scale"]):
            st.info("Run YOLO detection first in Step 4 to enable terrain reconstruction.")
            return

    if st.session_state.depth is None or len(st.session_state.depth.get("rows", [])) == 0:
        st.info("Depth estimates required before terrain reconstruction.")
        return

    ensure_terrain(profile=params["terrain_profile"])
    terrain = st.session_state.terrain

    st.caption(
        f"3D profile: {terrain['surface_profile']} | mesh target: {terrain['surface_target_size']} px | "
        f"estimated render memory: {terrain['surface_memory_est_mb']:.1f} MB"
    )

    col1, col2 = st.columns([1.2, 1.0])
    with col1:
        st.plotly_chart(terrain["heatmap_fig"], use_container_width=True)
    with col2:
        st.plotly_chart(terrain["contour_fig"], use_container_width=True)

    st.plotly_chart(terrain["surface_fig"], use_container_width=True)

    if st.button("▶ Auto-Rotate 3D Camera"):
        ph = st.empty()
        frames = 60
        for i in range(frames):
            t = 2.0 * np.pi * i / frames
            # Smooth easing: accelerate then decelerate
            ease = 0.5 * (1.0 - np.cos(np.pi * i / frames))
            angle = t + 0.3 * ease

            fig = make_surface_figure(
                terrain["depth_map"],
                downsample=True,
                target_size=int(terrain["surface_target_size"]),
            )
            fig.update_layout(
                scene_camera=dict(
                    eye=dict(
                        x=1.8 * np.cos(angle),
                        y=1.8 * np.sin(angle),
                        z=0.7 + 0.2 * np.sin(t * 2),
                    )
                )
            )
            ph.plotly_chart(fig, use_container_width=True)
            time.sleep(0.05)
        ph.empty()
        st.plotly_chart(terrain["surface_fig"], use_container_width=True)

    log_once("terrain_step", "[TERRAIN] >> Generated depth heatmap, contour map, and interactive 3D surface")


def _zone_style(zone: str) -> tuple[str, str]:
    """Return background and border colors for zone score cards."""

    if zone == "SAFE":
        return "#1a2a3f", "#7dd3fc"
    if zone == "CAUTION":
        return "#3a2a08", "#f59e0b"
    return "#3b1010", "#ef4444"


def step_07_scoring(params: dict[str, Any]) -> None:
    """Render crater safety cards, hazard map, and summary gauge."""

    initialization_animation("step07", "SCORING")
    st.markdown(f"## {STEP_TITLES[7]}")

    if st.session_state.depth is None or len(st.session_state.depth.get("rows", [])) == 0:
        if not ensure_depth(theta_deg=params["theta"], pixel_scale_m=params["pixel_scale"]):
            st.info("Run YOLO detection first in Step 4 to enable safety scoring.")
            return

    if st.session_state.depth is None or len(st.session_state.depth.get("rows", [])) == 0:
        st.info("Depth estimates required before safety scoring.")
        return

    ensure_scoring(
        td=params["td"],
        gear_span_m=params["gear_span"],
        density_radius_px=params["density_radius"],
        pixel_scale_m=params["pixel_scale"],
    )
    scoring = st.session_state.scoring

    if not scoring or len(scoring["rows"]) == 0:
        st.warning("No crater rows available for scoring.")
        return

    st.markdown("### Crater Score Cards")
    cols = st.columns(3)
    for i, row in enumerate(scoring["rows"]):
        bg, border = _zone_style(row["zone"])
        with cols[i % 3]:
            st.markdown(
                f"""
                <div style="background:{bg}; border:1px solid {border}; box-shadow:0 0 8px {border}; border-radius:6px; padding:10px; margin-bottom:8px;">
                <div><b>{row['crater_id']}</b></div>
                <div>Score: <b>{row['safety_score']:.1f}</b></div>
                <div>Zone: {row['zone']}</div>
                <div>Depth: {row['depth_m']:.2f} m</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    s = scoring["summary"]
    st.markdown(
        f"""
        <div class='glow-panel' style='text-align:center;'>
            <span style='color:#7dd3fc; font-size:1.2rem;'>{s['safe']} SAFE ZONES</span>
            &nbsp; | &nbsp;
            <span style='color:#f59e0b; font-size:1.2rem;'>{s['caution']} CAUTION ZONES</span>
            &nbsp; | &nbsp;
            <span style='color:#ef4444; font-size:1.2rem;'>{s['hazard']} HAZARD ZONES</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.image(st.session_state.hazard_map, caption="Annotated Hazard Map", use_container_width=True)
    st.plotly_chart(build_safety_gauge(scoring["overall_score"]), use_container_width=True)

    log_once(
        f"score_{params['td']}_{params['gear_span']}_{params['density_radius']}",
        "[SCORER] >> "
        f"safe={s['safe']} caution={s['caution']} hazard={s['hazard']} | overall={scoring['overall_score']:.1f}",
    )


def _path_overlay_figure(
    base_image: np.ndarray,
    primary: list[tuple[int, int]],
    alternatives: list[list[tuple[int, int]]],
    max_primary_points: int | None = None,
) -> go.Figure:
    """Create plotly overlay with dashed primary path and alternates.

    Args:
        base_image: RGB hazard map.
        primary: Primary path coordinates.
        alternatives: Alternative path coordinate sets.
        max_primary_points: Optional cap for progressive path rendering.

    Returns:
        Plotly figure.
    """

    if base_image.ndim == 2:
        img = cv2.cvtColor(base_image, cv2.COLOR_GRAY2RGB)
    else:
        img = base_image

    fig = go.Figure()
    fig.add_trace(go.Image(z=img))

    for path in alternatives:
        if len(path) >= 2:
            xs = [p[0] for p in path]
            ys = [p[1] for p in path]
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines",
                    line=dict(color="rgba(120,180,255,0.6)", width=2),
                    name="Alternative",
                )
            )

    visible_primary = primary
    if max_primary_points is not None:
        visible_primary = primary[: max(0, max_primary_points)]

    if len(visible_primary) >= 2:
        xs = [p[0] for p in visible_primary]
        ys = [p[1] for p in visible_primary]
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line=dict(color="#60a5fa", width=3, dash="dash"),
                name="Optimal Path",
            )
        )

    fig.update_yaxes(autorange="reversed", visible=False)
    fig.update_xaxes(visible=False)
    fig.update_layout(
        title="Descent Path Overlay",
        paper_bgcolor="#0a0e1a",
        plot_bgcolor="#0a0e1a",
        font=dict(color="#dbe7f4", family="Courier New"),
        margin=dict(l=5, r=5, t=40, b=5),
        legend=dict(bgcolor="rgba(0,0,0,0.3)"),
    )

    return fig


def step_08_path(params: dict[str, Any]) -> None:
    """Render A* path planning outputs and recommended coordinates."""

    initialization_animation("step08", "PATHFINDER")
    st.markdown(f"## {STEP_TITLES[8]}")

    if st.session_state.scoring is None:
        ensure_scoring(
            td=params["td"],
            gear_span_m=params["gear_span"],
            density_radius_px=params["density_radius"],
            pixel_scale_m=params["pixel_scale"],
        )

    if st.session_state.scoring is None:
        st.info("Safety scoring must run before path planning.")
        return

    ensure_paths(pixel_scale_m=params["pixel_scale"])
    paths = st.session_state.paths
    if not paths or len(paths["primary_path"]) == 0:
        st.warning("Pathfinder could not identify a valid route.")
        return

    fig = _path_overlay_figure(
        base_image=st.session_state.hazard_map,
        primary=paths["primary_path"],
        alternatives=paths["alternative_paths"],
    )
    st.plotly_chart(fig, use_container_width=True)

    if st.button("▶ Animate Descent Dashes"):
        ph = st.empty()
        total = len(paths["primary_path"])
        if total > 2:
            frames = np.linspace(2, total, num=min(30, total), dtype=int)
            for n in frames:
                anim_fig = _path_overlay_figure(
                    base_image=st.session_state.hazard_map,
                    primary=paths["primary_path"],
                    alternatives=paths["alternative_paths"],
                    max_primary_points=int(n),
                )
                ph.plotly_chart(anim_fig, use_container_width=True)
                time.sleep(0.08)
            ph.empty()
            st.plotly_chart(fig, use_container_width=True)

    gx, gy = paths["goal"]
    st.markdown(
        f"""
        <div class='glow-panel'>
            <b>RECOMMENDED LANDING COORDINATES</b><br/>
            Pixel: ({gx}, {gy})<br/>
            Confidence: <span style='color:#7dd3fc'>{paths['landing_confidence']:.1f}%</span><br/>
            Path Length: {paths['path_length_m']:.1f} m
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.image(paths["overlay"], caption="Path Overlay on Hazard Map", use_container_width=True)

    log_once(
        "step8_path",
        "[PATHFINDER] >> A* complete | "
        f"Path length: {paths['path_length_m']:.1f}m | Obstacles avoided: {paths['obstacles_avoided']} | "
        f"LZ confidence: {paths['landing_confidence']:.0f}%",
    )


def step_09_report(params: dict[str, Any]) -> None:
    """Render final mission summary and export downloads."""

    initialization_animation("step09", "REPORT")
    st.markdown(f"## {STEP_TITLES[9]}")

    if st.session_state.scoring is None:
        ensure_scoring(
            td=params["td"],
            gear_span_m=params["gear_span"],
            density_radius_px=params["density_radius"],
            pixel_scale_m=params["pixel_scale"],
        )
        if st.session_state.scoring is not None:
            ensure_paths(pixel_scale_m=params["pixel_scale"])

    if st.session_state.scoring is None:
        st.info("Mission report requires at least scoring outputs.")
        return

    render_pipeline_flow(st.session_state.current_step, TOTAL_STEPS, set(range(1, 10)))

    rows = st.session_state.scoring["rows"]
    st.markdown("### Final Crater Table")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    if st.session_state.hazard_map is not None:
        st.image(st.session_state.hazard_map, caption="Final Hazard Map", use_container_width=True)

    goal = st.session_state.paths["goal"] if st.session_state.paths else None
    if goal:
        st.markdown(f"### Recommended Landing Zone: ({goal[0]}, {goal[1]})")

    pdf_bytes = build_mission_pdf(
        mission_id=MISSION_ID,
        crater_rows=rows,
        hazard_map=st.session_state.hazard_map,
        recommended_coordinates=goal,
        overall_score=st.session_state.scoring["overall_score"],
    )
    png_bytes = image_to_png_bytes(st.session_state.hazard_map) if st.session_state.hazard_map is not None else b""
    csv_bytes = crater_rows_to_csv(rows)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button(
            "DOWNLOAD PDF REPORT",
            data=pdf_bytes,
            file_name=f"{MISSION_ID.lower()}_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    with c2:
        st.download_button(
            "DOWNLOAD HAZARD MAP (PNG)",
            data=png_bytes,
            file_name=f"{MISSION_ID.lower()}_hazard_map.png",
            mime="image/png",
            use_container_width=True,
        )
    with c3:
        st.download_button(
            "DOWNLOAD CRATER DATA (CSV)",
            data=csv_bytes,
            file_name=f"{MISSION_ID.lower()}_crater_data.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.session_state.completed_steps.add(9)
    log_once("step9_report", "[REPORT] >> Export package ready (PDF, PNG, CSV)")


def _can_advance_from_step(step: int) -> tuple[bool, str]:
    """Validate whether moving right from the current step is allowed.

    Args:
        step: Current step index.

    Returns:
        Tuple of (allowed, reason_message_if_not_allowed).
    """

    if step == 1 and not st.session_state.mission_started:
        return False, "Press LAUNCH MISSION ► in Step 01 first."
    return True, ""


def render_navigation(key_prefix: str) -> None:
    """Render large left/right arrow controls for step navigation.

    Args:
        key_prefix: Unique key prefix to allow rendering multiple nav bars.
    """

    st.markdown("<div class='nav-wrap'></div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 1.2, 1])
    with c2:
        left, right = st.columns(2)
        with left:
            if st.button("◄", key=f"{key_prefix}_left", use_container_width=True, disabled=st.session_state.current_step <= 1):
                st.session_state.current_step = max(1, st.session_state.current_step - 1)
                st.rerun()
        with right:
            if st.button("►", key=f"{key_prefix}_right", use_container_width=True, disabled=st.session_state.current_step >= TOTAL_STEPS):
                allowed, reason = _can_advance_from_step(st.session_state.current_step)
                if not allowed:
                    st.warning(reason)
                else:
                    st.session_state.current_step = min(TOTAL_STEPS, st.session_state.current_step + 1)
                    st.rerun()


def main() -> None:
    """Application entrypoint for mission-control interactive pipeline."""

    st.set_page_config(page_title="Autonomous Planetary Landing Site Analyzer", layout="wide")
    init_state()
    inject_mission_css()

    params = render_sidebar()
    st.session_state.mission_status = mission_status_from_findings()
    _on_step_change()

    render_hud(
        step=st.session_state.current_step,
        total_steps=TOTAL_STEPS,
        mission_status=st.session_state.mission_status,
    )

    step = st.session_state.current_step
    if step == 1:
        step_01_briefing()
    elif step == 2:
        step_02_raw_acquisition()
    elif step == 3:
        step_03_preprocess()
    elif step == 4:
        step_04_detection()
    elif step == 5:
        step_05_depth(params)
    elif step == 6:
        step_06_terrain(params)
    elif step == 7:
        step_07_scoring(params)
    elif step == 8:
        step_08_path(params)
    elif step == 9:
        step_09_report(params)

    render_navigation("bottom")

    st.markdown("### TERMINAL LOG")
    render_terminal_log(st.session_state.terminal_logs, max_lines=12)


if __name__ == "__main__":
    main()
