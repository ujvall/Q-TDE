"""
QR Code Security Inspection Engine - Minimal Technical Dashboard.

Single-page web GUI built with Streamlit.
Consumes app/inspector.py to evaluate two strictly independent evidence dimensions:
1. Physical / Structural Tampering Integrity (Reed-Solomon ECC + Classical Peak Edge Gradient)
2. Cyber / Destination URL Risk (Deterministic Static URL Structural Analysis)
"""

import os
import sys
import numpy as np
import cv2
import streamlit as st

# Ensure project root is present in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.inspector import inspect_image
from app.camera import QRCameraProcessor

camera_processor = QRCameraProcessor(target_size=370)


# ============================================================
# PAGE CONFIGURATION & MINIMAL DESIGN SYSTEM
# ============================================================

st.set_page_config(
    page_title="QR Security Inspector",
    page_icon="▪",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    /* Base Neutral Reset */
    #MainMenu, header, footer { visibility: hidden; height: 0; }
    
    .stApp {
        background-color: #090a0c;
        color: #d1d5db;
        font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", Roboto, sans-serif;
    }
    
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 3rem !important;
        max-width: 1240px !important;
    }
    
    /* Typography */
    h1, h2, h3, h4 {
        color: #f3f4f6;
        font-weight: 500;
        letter-spacing: -0.02em;
    }
    
    /* Header layout */
    .top-header {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        padding-bottom: 1rem;
        border-bottom: 1px solid #1f2329;
        margin-bottom: 1.5rem;
    }
    
    .brand-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #f9fafb;
        letter-spacing: -0.01em;
    }
    
    .brand-subtitle {
        font-size: 0.85rem;
        color: #6b7280;
        margin-top: 2px;
    }
    
    .spec-badges {
        display: flex;
        gap: 8px;
        align-items: center;
    }
    
    .spec-tag {
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 0.72rem;
        color: #9ca3af;
        background: #111418;
        border: 1px solid #22272e;
        padding: 3px 8px;
        border-radius: 4px;
    }
    
    /* Streamlit Input Overrides for Dark Mode */
    [data-testid="stFileUploader"] section {
        background-color: #111418 !important;
        border: 1px dashed #262c36 !important;
        border-radius: 6px !important;
        padding: 16px !important;
    }
    [data-testid="stFileUploader"] section:hover {
        border-color: #3b4352 !important;
    }
    [data-testid="stFileUploader"] small {
        color: #6b7280 !important;
    }
    
    .stSelectbox div[data-baseweb="select"] > div {
        background-color: #111418 !important;
        border: 1px solid #262c36 !important;
        border-radius: 6px !important;
        color: #e5e7eb !important;
    }

    /* Overall Verdict Callout (No gradients, crisp borders) */
    .verdict-callout {
        padding: 14px 18px;
        border-radius: 6px;
        margin: 1.25rem 0 1.5rem 0;
        border: 1px solid #22272e;
        background-color: #111418;
    }
    .verdict-callout-clean {
        border-left: 3px solid #22c55e;
        background-color: #0c1510;
    }
    .verdict-callout-warning {
        border-left: 3px solid #eab308;
        background-color: #17140b;
    }
    .verdict-callout-danger {
        border-left: 3px solid #ef4444;
        background-color: #180d0e;
    }
    .verdict-heading {
        font-size: 0.98rem;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .verdict-heading-clean { color: #4ade80; }
    .verdict-heading-warning { color: #facc15; }
    .verdict-heading-danger { color: #f87171; }
    
    .verdict-sub {
        font-size: 0.8rem;
        color: #9ca3af;
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    }

    /* Minimal Technical Panels */
    .panel {
        background: #111418;
        border: 1px solid #1f2329;
        border-radius: 6px;
        padding: 18px;
        margin-bottom: 1.2rem;
    }
    
    .panel-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding-bottom: 10px;
        margin-bottom: 14px;
        border-bottom: 1px solid #1a1e24;
    }
    
    .panel-title {
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #9ca3af;
    }
    
    .status-pill {
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        font-size: 0.72rem;
        font-weight: 600;
        padding: 3px 8px;
        border-radius: 4px;
        border: 1px solid #22272e;
    }
    .status-pill-clean { color: #4ade80; border-color: #15803d; background: #062312; }
    .status-pill-danger { color: #f87171; border-color: #b91c1c; background: #260c0e; }
    .status-pill-warning { color: #facc15; border-color: #a16207; background: #261e07; }
    .status-pill-muted { color: #9ca3af; border-color: #374151; background: #111827; }

    /* Clean Key-Value Data Table */
    .data-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.82rem;
        margin-bottom: 8px;
    }
    
    .data-table tr {
        border-bottom: 1px solid #171b21;
    }
    .data-table tr:last-child {
        border-bottom: none;
    }
    
    .data-table td {
        padding: 7px 0;
    }
    
    .data-label {
        color: #6b7280;
        width: 50%;
    }
    
    .data-value {
        color: #e5e7eb;
        text-align: right;
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    }
    
    .section-label {
        font-size: 0.75rem;
        font-weight: 600;
        color: #4b5563;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin: 14px 0 6px 0;
    }
    
    /* Code box for URL */
    .url-box {
        background: #090a0c;
        border: 1px solid #1f2329;
        border-radius: 4px;
        padding: 8px 12px;
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        font-size: 0.82rem;
        color: #f3f4f6;
        word-break: break-all;
        margin-bottom: 12px;
    }
    
    /* Evidence Item */
    .evidence-item {
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        font-size: 0.76rem;
        color: #f87171;
        background: #200d0f;
        border: 1px solid #451318;
        padding: 3px 8px;
        border-radius: 4px;
        display: inline-block;
        margin-right: 6px;
        margin-bottom: 6px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="top-header">
    <div>
        <div class="brand-title">QR Code Tampering & URL Risk Inspector</div>
        <div class="brand-subtitle">Orthogonal physical/structural tampering detection & static destination risk analyzer</div>
    </div>
    <div class="spec-badges">
        <span class="spec-tag">RS-ECC · GF(256)</span>
        <span class="spec-tag">Peak Edge · τ=0.525490</span>
        <span class="spec-tag">Static URL · 21 Rules</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# INPUT MODE SELECTOR
# ============================================================

input_mode = st.radio(
    "Input Source",
    options=["Upload Image", "Camera"],
    horizontal=True,
    index=0
)

st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)

col_ctrl, col_view = st.columns([1.1, 0.9])

input_image = None
image_path_to_inspect = None
res = None

if input_mode == "Upload Image":
    PRESET_SAMPLES = {
        "— Select a demonstration preset or upload below —": None,
        "Genuine QR: Val Sample 001 (Clean & Benign URL)": "data/val/genuine/genuine_001.png",
        "Tampered QR: Val Sample 028 (Reed-Solomon ECC Violation)": "data/val/tampered/tampered_028.png",
        "Tampered QR: Val Sample 065 (Peak High-Frequency Anomaly)": "data/val/tampered/tampered_065.png",
    }

    with col_ctrl:
        selected_preset = st.selectbox(
            "Demonstration Preset",
            options=list(PRESET_SAMPLES.keys()),
            index=0,
            label_visibility="collapsed"
        )
        
        uploaded_file = st.file_uploader(
            "Upload QR Code Image",
            type=["png", "jpg", "jpeg", "webp"],
            label_visibility="collapsed"
        )

    if uploaded_file is not None:
        try:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            input_image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        except Exception as e:
            st.error(f"Failed to decode uploaded image: {str(e)}")
    elif selected_preset and PRESET_SAMPLES[selected_preset] is not None:
        image_path_to_inspect = PRESET_SAMPLES[selected_preset]
        if os.path.exists(image_path_to_inspect):
            input_image = cv2.imread(image_path_to_inspect, cv2.IMREAD_GRAYSCALE)

    if input_image is not None:
        try:
            if image_path_to_inspect:
                res = inspect_image(image_path=image_path_to_inspect)
            else:
                res = inspect_image(image=input_image)
        except Exception as err:
            st.error(f"Inspection error: {str(err)}")
            res = None

    with col_view:
        if input_image is not None:
            h, w = input_image.shape[:2]
            st.image(input_image, caption=f"Input Image ({w} × {h} px)", width=220)

            # If shared preprocessing rectified a distorted/resized image, show rectified crop
            if res and res.get("preprocessing"):
                prep_meta = res["preprocessing"]
                if not prep_meta.get("is_version_3", True):
                    st.warning(f"⚠️ {prep_meta.get('message', 'Unsupported QR geometry')}")
        else:
            st.markdown("""
            <div style="border: 1px dashed #1f2329; border-radius: 6px; padding: 24px; text-align: center; color: #4b5563; font-size: 0.82rem;">
                No QR code loaded. Choose a preset or upload an image file.
            </div>
            """, unsafe_allow_html=True)

else:  # Camera Mode
    with col_ctrl:
        st.markdown(
            '<div style="font-size:0.82rem;color:#9ca3af;margin-bottom:10px;">'
            'Capture a frame from the local webcam or browser camera input. '
            'The engine will localize the QR code, correct perspective distortion, and inspect integrity.'
            '</div>',
            unsafe_allow_html=True
        )
        btn_capture = st.button("📸 Capture Local Webcam", use_container_width=True)
        camera_snapshot = st.camera_input("Browser Camera Snapshot", label_visibility="collapsed")

    captured_frame = None
    if btn_capture:
        with st.spinner("Accessing local webcam..."):
            ok, frame, msg = camera_processor.capture_frame(device_id=0)
            if ok and frame is not None:
                captured_frame = frame
            else:
                st.warning(f"⚠️ {msg}")

    if camera_snapshot is not None and captured_frame is None:
        file_bytes = np.asarray(bytearray(camera_snapshot.read()), dtype=np.uint8)
        captured_frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if captured_frame is not None:
        res, corrected_crop, cam_msg = camera_processor.inspect_frame(captured_frame)
        _, _, _, pts, _ = camera_processor.detect_and_correct(captured_frame)
        annotated_frame = camera_processor.draw_overlay(captured_frame, pts)

        with col_view:
            st.image(annotated_frame, channels="BGR" if len(annotated_frame.shape) == 3 else "GRAY", caption="Camera Capture (with Detection Overlay)", width=240)
            if corrected_crop is not None:
                st.image(corrected_crop, caption="Warped QR Crop (370×370 px)", width=180)
            else:
                st.warning(f"ℹ️ {cam_msg}")
    else:
        with col_view:
            st.markdown("""
            <div style="border: 1px dashed #1f2329; border-radius: 6px; padding: 24px; text-align: center; color: #4b5563; font-size: 0.82rem;">
                Camera idle. Click "Capture Local Webcam" or use the camera snapshot tool.
            </div>
            """, unsafe_allow_html=True)


# ============================================================
# RESULTS PRESENTATION
# ============================================================

if res is not None:

    if res:
        tampering = res.get("tampering", {})
        url_sec = res.get("url", {})
        overall = res.get("overall", {})

        t_status = overall.get("tampering_status", "unavailable")
        u_risk = overall.get("url_risk_level", "unavailable")
        summary_text = overall.get("inspection_summary", "")

        is_tampered = (t_status == "tampering_detected")
        is_clean_tamper = (t_status == "no_tampering_evidence_detected")

        # Format URL risk text
        if u_risk in ["none", "low", "moderate", "high"]:
            url_risk_str = f"URL Risk: {u_risk.capitalize()}"
        elif u_risk == "non_url_payload":
            url_risk_str = "Payload: Non-URL Data"
        elif u_risk == "unavailable":
            url_risk_str = "URL Risk: Unavailable"
        else:
            url_risk_str = f"URL Risk: {u_risk}"

        # Format Tampering text
        if is_clean_tamper:
            tamper_str = "No Tampering Evidence Detected"
        elif is_tampered:
            tamper_str = "QR Tampering Detected"
        elif t_status == "detection_failed":
            tamper_str = "QR Detection Failed"
        elif t_status == "decode_failed":
            tamper_str = "QR Decode Failed"
        else:
            tamper_str = t_status.replace("_", " ").title()

        # Build exact orthogonal banner wording
        if is_clean_tamper and u_risk == "none":
            callout_class = "verdict-callout-clean"
            heading_class = "verdict-heading-clean"
            banner_title = f"No Tampering Evidence Detected · {url_risk_str}"
        elif is_tampered and u_risk == "none":
            callout_class = "verdict-callout-danger"
            heading_class = "verdict-heading-danger"
            banner_title = f"QR Tampering Detected · {url_risk_str}"
        elif is_clean_tamper and u_risk in ["high", "moderate"]:
            callout_class = "verdict-callout-danger" if u_risk == "high" else "verdict-callout-warning"
            heading_class = "verdict-heading-danger" if u_risk == "high" else "verdict-heading-warning"
            banner_title = f"{url_risk_str} · {tamper_str}"
        elif is_clean_tamper and u_risk == "low":
            callout_class = "verdict-callout-clean"
            heading_class = "verdict-heading-clean"
            banner_title = f"{url_risk_str} · {tamper_str}"
        elif is_tampered and u_risk in ["moderate", "high", "low"]:
            callout_class = "verdict-callout-danger"
            heading_class = "verdict-heading-danger"
            banner_title = f"QR Tampering Detected · {url_risk_str}"
        else:
            callout_class = "verdict-callout-warning"
            heading_class = "verdict-heading-warning"
            banner_title = f"{tamper_str} · {url_risk_str}"

        # Clean Verdict Callout
        st.markdown(f"""
        <div class="verdict-callout {callout_class}">
            <div class="verdict-heading {heading_class}">{banner_title}</div>
            <div class="verdict-sub">Physical: {t_status} &nbsp;|&nbsp; Cyber: {u_risk} &nbsp;|&nbsp; {summary_text}</div>
        </div>
        """, unsafe_allow_html=True)

        # ------------------------------------------------------------
        # DUAL-DIMENSION DETAILS
        # ------------------------------------------------------------
        col_tamper, col_url = st.columns(2)

        # ------------------------------------------------------------
        # PANEL 1: PHYSICAL / STRUCTURAL INTEGRITY
        # ------------------------------------------------------------
        with col_tamper:
            if t_status == "tampering_detected":
                pill_html = '<span class="status-pill status-pill-danger">TAMPERING DETECTED</span>'
            elif t_status == "no_tampering_evidence_detected":
                pill_html = '<span class="status-pill status-pill-clean">NO EVIDENCE DETECTED</span>'
            else:
                pill_html = f'<span class="status-pill status-pill-muted">{t_status.upper()}</span>'

            st.markdown(f"""
            <div class="panel">
                <div class="panel-header">
                    <span class="panel-title">01 · Physical & Structural Integrity</span>
                    {pill_html}
                </div>
            """, unsafe_allow_html=True)

            # Evidence List
            evidence = tampering.get("evidence", [])
            st.markdown('<div class="section-label">Evidence Signals</div>', unsafe_allow_html=True)
            if evidence:
                ev_html = "".join([f'<span class="evidence-item">{e}</span>' for e in evidence])
                st.markdown(ev_html, unsafe_allow_html=True)
            else:
                st.markdown('<span style="font-size:0.8rem;color:#6b7280;">No structural or high-frequency anomalies detected.</span>', unsafe_allow_html=True)

            # ECC Measurements
            ecc = tampering.get("ecc", {})
            st.markdown('<div class="section-label">Reed-Solomon ECC Analysis</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <table class="data-table">
                <tr><td class="data-label">Codeword Status</td><td class="data-value">{ecc.get('status', 'N/A')}</td></tr>
                <tr><td class="data-label">Module Discrepancy</td><td class="data-value">{ecc.get('module_diff', 0)}</td></tr>
                <tr><td class="data-label">Non-Zero Syndromes</td><td class="data-value">{ecc.get('nonzero_syndromes', 0)} / 26</td></tr>
                <tr><td class="data-label">Syndrome Residual Sum</td><td class="data-value">{ecc.get('residual', 0)}</td></tr>
            </table>
            """, unsafe_allow_html=True)

            # Peak Edge Anomaly
            peak = tampering.get("peak_anomaly", {})
            st.markdown('<div class="section-label">Peak High-Frequency Gradient</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <table class="data-table">
                <tr><td class="data-label">Peak Edge Residual</td><td class="data-value">{peak.get('score', 0.0):.6f}</td></tr>
                <tr><td class="data-label">Calibrated Threshold (τ)</td><td class="data-value">{peak.get('threshold', 0.525490):.6f}</td></tr>
                <tr><td class="data-label">Threshold Exceeded</td><td class="data-value">{str(peak.get('triggered', False))}</td></tr>
                <tr><td class="data-label">Visual Anomaly Status</td><td class="data-value">{peak.get('status', 'N/A')}</td></tr>
            </table>
            </div>
            """, unsafe_allow_html=True)

        # ------------------------------------------------------------
        # PANEL 2: CYBER / DESTINATION URL RISK
        # ------------------------------------------------------------
        with col_url:
            if u_risk == "high":
                u_pill = '<span class="status-pill status-pill-danger">HIGH RISK</span>'
            elif u_risk == "moderate":
                u_pill = '<span class="status-pill status-pill-warning">MODERATE RISK</span>'
            elif u_risk == "low":
                u_pill = '<span class="status-pill status-pill-warning">LOW RISK</span>'
            elif u_risk == "none":
                u_pill = '<span class="status-pill status-pill-clean">ZERO STATIC RISKS</span>'
            else:
                u_pill = f'<span class="status-pill status-pill-muted">{u_risk.upper()}</span>'

            st.markdown(f"""
            <div class="panel">
                <div class="panel-header">
                    <span class="panel-title">02 · Destination URL Security</span>
                    {u_pill}
                </div>
            """, unsafe_allow_html=True)

            # Decoded URL box
            st.markdown('<div class="section-label">Decoded Destination Payload</div>', unsafe_allow_html=True)
            raw_url = url_sec.get("url", "")
            if raw_url:
                st.markdown(f'<div class="url-box">{raw_url}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="url-box" style="color:#6b7280;">No payload available.</div>', unsafe_allow_html=True)

            # URL Structural Attributes
            st.markdown('<div class="section-label">Structural Attributes</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <table class="data-table">
                <tr><td class="data-label">Heuristic Risk Score</td><td class="data-value">{url_sec.get('risk_score', 0)} / 100</td></tr>
                <tr><td class="data-label">Protocol Scheme</td><td class="data-value">{url_sec.get('scheme', 'none') or 'missing'}</td></tr>
                <tr><td class="data-label">HTTPS Encrypted</td><td class="data-value">{str(url_sec.get('is_https', False))}</td></tr>
                <tr><td class="data-label">Hostname</td><td class="data-value">{url_sec.get('hostname', 'none') or 'N/A'}</td></tr>
                <tr><td class="data-label">IP Literal Host</td><td class="data-value">{str(url_sec.get('is_ip_address', False))}</td></tr>
                <tr><td class="data-label">URL Shortener</td><td class="data-value">{str(url_sec.get('uses_shortener', False))}</td></tr>
            </table>
            """, unsafe_allow_html=True)

            # Indicators
            st.markdown('<div class="section-label">Triggered Risk Indicators</div>', unsafe_allow_html=True)
            indicators = url_sec.get("risk_indicators", [])
            if indicators:
                ind_rows = "".join([
                    f"<tr><td class='data-label' style='color:#f87171;'>{ind.get('indicator')}</td>"
                    f"<td class='data-value' style='font-size:0.75rem;'>+{ind.get('weight', 0)} ({ind.get('severity')})</td></tr>"
                    for ind in indicators
                ])
                st.markdown(f'<table class="data-table">{ind_rows}</table></div>', unsafe_allow_html=True)
            else:
                st.markdown('<span style="font-size:0.8rem;color:#6b7280;">No static URL structural risks flagged.</span></div>', unsafe_allow_html=True)

        # ------------------------------------------------------------
        # RAW JSON & ARCHITECTURAL DISCLOSURE
        # ------------------------------------------------------------
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        with st.expander("Raw Inspection Output (JSON)", expanded=False):
            st.json(res)

        with st.expander("Inspection Architecture & Methodology", expanded=False):
            st.markdown("""
            **Dual-Dimension Independence**
            - **Physical Integrity**: Version 3-M single-block Reed-Solomon polynomial syndrome evaluation ($70$ codewords, $26$ ECC) over $\\text{GF}(2^8)$ identifies codeword-level alterations. Classical peak high-frequency edge residual analysis ($\max |I - \\text{GaussianBlur}(I)| / 255$) detects localized visual anomalies exceeding $\\tau = 0.525490$.
            - **Cyber Risk**: Purely deterministic static structural URL analysis evaluating 21 risk indicator rules (IP host literals, userinfo `@` obfuscation, dangerous schemes, and shorteners) without network requests or execution.
            - **Negative State Terminology**: Absences of detected evidence are designated `no_tampering_evidence_detected` rather than claiming definitive proof of authenticity.
            - **Characterized Inactive Models**: Vision Transformer (ViT) whole-image classifiers and deep autoencoders were empirically characterized during audit as non-discriminative for reference-free tampering detection and remain excluded from the active decision engine.
            """)
