"""
Q-TDE: QR Code Tampering & URL Risk Inspection, Streamlit interface.

Consumes app/inspector.py and reports two strictly independent evidence dimensions:
1. Physical integrity (Reed-Solomon ECC for Version 3-M + classical peak high-frequency residual)
2. URL structure (deterministic static analysis; nothing is fetched or executed)

Run from the project root:  streamlit run app/gui.py
Styling lives in app/gui.css; theme colors and fonts in .streamlit/config.toml.
"""

import base64
import html
import os
import sys

import cv2
import numpy as np
import qrcode
import streamlit as st

# Ensure project root is present in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.inspector import inspect_image
from app.camera import QRCameraProcessor
from preprocessing.qr_preprocessor import preprocess_qr_image

camera_processor = QRCameraProcessor(target_size=370)

REPO_URL = "https://github.com/ujvall/Q-TDE"
CSS_PATH = os.path.join(os.path.dirname(__file__), "gui.css")

# Dataset samples are not committed, so only offer the ones present on disk.
_SAMPLE_CANDIDATES = {
    "Genuine, benign URL": ("data/val/genuine/genuine_001.png", "Validation sample 001"),
    "Tampered, ECC violation": ("data/val/tampered/tampered_028.png", "Validation sample 028"),
    "Tampered, visual anomaly": ("data/val/tampered/tampered_065.png", "Validation sample 065"),
}
SAMPLES = {
    label: (os.path.join(PROJECT_ROOT, path), caption)
    for label, (path, caption) in _SAMPLE_CANDIDATES.items()
    if os.path.exists(os.path.join(PROJECT_ROOT, path))
}



def esc(value) -> str:
    """HTML-escape untrusted text; `$` is escaped too so Streamlit markdown never renders it as LaTeX."""
    return html.escape(str(value)).replace("$", "&#36;")


def html_block(markup: str) -> None:
    """Render trusted markup as one raw HTML block (blank lines would end the block in CommonMark)."""
    st.markdown("\n".join(line.strip() for line in markup.splitlines() if line.strip()), unsafe_allow_html=True)


# ============================================================
# STATE VOCABULARY
# Each state: (title, tone, glyph). Glyph: fill = evidence, hollow = none found, hatch = not assessed.
# ============================================================

PHYSICAL_STATES = {
    "tampering_detected": ("Tampering detected", "danger", "fill"),
    "no_tampering_evidence_detected": ("No tampering evidence", "clear", "hollow"),
    "unsupported_geometry": ("Unsupported geometry", "muted", "hatch"),
    "detection_failed": ("No QR code located", "muted", "hatch"),
    "decode_failed": ("Located, not decoded", "warn", "hatch"),
    "unavailable": ("Analysis unavailable", "muted", "hatch"),
}

URL_STATES = {
    "none": ("No structural indicators", "clear", "hollow"),
    "low": ("Low structural risk", "caution", "fill"),
    "moderate": ("Moderate structural risk", "warn", "fill"),
    "high": ("High structural risk", "danger", "fill"),
    "non_url_payload": ("Not a URL", "muted", "hatch"),
    "unavailable": ("No payload to analyse", "muted", "hatch"),
}

EVIDENCE = {
    "ecc_integrity_violation": (
        "Reed-Solomon syndrome mismatch",
        "Sampled codewords no longer satisfy the Version 3-M error-correction equations.",
    ),
    "local_visual_anomaly": (
        "High-frequency residual above τ",
        "A local edge residual exceeded the calibrated threshold.",
    ),
}

SEVERITY_TONE = {"CRITICAL": "danger", "HIGH": "danger", "MODERATE": "warn", "LOW": "caution"}

PREP_STATUS = {
    "SUCCESS": "Located, rectified, decoded",
    "DECODE_FAILED": "Located and rectified, not decoded",
    "UNSUPPORTED_VERSION": "Located, unsupported version",
    "NO_QR_DETECTED": "No symbol found",
    "PERSPECTIVE_WARP_FAILED": "Rectification failed",
    "INVALID_IMAGE": "Unreadable image",
}


def mod(kind: str) -> str:
    return f'<span class="qt-mod" data-k="{kind}" aria-hidden="true"></span>'


def physical_meaning(status: str, res: dict) -> str:
    prep = res.get("preprocessing") or {}
    if status == "tampering_detected":
        return "At least one physical check found evidence that the symbol was altered."
    if status == "no_tampering_evidence_detected":
        return "Neither physical check fired. That is not proof the code is genuine."
    if status == "unsupported_geometry":
        version = prep.get("detected_version")
        found = f"Version {version} symbol found" if version else "Symbol geometry not recognised"
        return f"{found}. ECC analysis currently supports Version 3-M only; the visual residual stayed below τ."
    if status == "detection_failed":
        return esc(prep.get("message") or "No QR symbol could be localized in this image.")
    if status == "decode_failed":
        return "A symbol was found but its payload could not be read, so the ECC check could not run."
    return esc(res.get("tampering", {}).get("summary") or "The physical analysis did not complete.")


def url_meaning(level: str) -> str:
    return {
        "none": "No structural rule matched. This says nothing about whether the destination is safe.",
        "low": "Minor structural indicators matched. The score is a heuristic, not a probability of harm.",
        "moderate": "Several structural indicators matched. Treat the destination with caution.",
        "high": "Critical or multiple structural indicators matched. Do not open without verifying the source.",
        "non_url_payload": "The payload is plain text or other data; URL analysis does not apply.",
    }.get(level, "There is no decoded payload, so URL structure could not be assessed.")


# ============================================================
# RENDERING HELPERS
# ============================================================

def to_data_uri(img: np.ndarray, max_side: int = 720) -> str:
    h, w = img.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", img)
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii") if ok else ""


def frame_html(img, caption: str, meta: str, corners=None, pixelated: bool = False) -> str:
    if img is None:
        return (
            f'<figure class="qt-frame"><div class="qt-empty-frame">{esc(meta)}</div>'
            f'<figcaption><span>{esc(caption)}</span><span>none</span></figcaption></figure>'
        )
    h, w = img.shape[:2]
    overlay = ""
    if corners is not None:
        pts = np.asarray(corners, dtype=float).reshape(-1, 2)
        if len(pts) == 4:
            points = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
            overlay = (
                f'<svg viewBox="0 0 {w} {h}" preserveAspectRatio="none" aria-hidden="true">'
                f'<polygon class="qt-corners" pathLength="1" points="{points}"/></svg>'
            )
    px = " qt-px" if pixelated else ""
    return (
        f'<figure class="qt-frame"><div class="qt-frame-img{px}">'
        f'<img src="{to_data_uri(img)}" alt="{esc(caption)}"/>{overlay}</div>'
        f'<figcaption><span>{esc(caption)}</span><span>{w} × {h} px</span></figcaption></figure>'
    )


def scale_html(value: float, hi: float, tone: str, ticks, bands, labels, aria: str) -> str:
    pct = lambda v: f"{max(0.0, min(1.0, v / hi)) * 100:.2f}%"
    band_html = "".join(
        f'<span class="qt-band" data-tone="{t}" style="left:{pct(a)};width:calc({pct(b)} - {pct(a)})"></span>'
        for a, b, t in bands
    )
    tick_html = "".join(f'<span class="qt-tick {cls}" style="left:{pct(v)}"></span>' for v, cls in ticks)
    label_html = "".join(f'<span style="left:{pct(v)}">{esc(t)}</span>' for v, t in labels)
    return (
        f'<div class="qt-scale" role="img" aria-label="{esc(aria)}">'
        f'<div class="qt-track">{band_html}{tick_html}'
        f'<span class="qt-needle" data-tone="{tone}" style="--v:{max(0.0, min(1.0, value / hi)):.4f}"></span></div>'
        f'<div class="qt-scale-labels" aria-hidden="true">{label_html}</div></div>'
    )


ACRONYMS = {"URL", "HTTP", "IP", "IDN"}


def humanize(code: str) -> str:
    """EXECUTABLE_DOWNLOAD_EXTENSION -> Executable download extension, keeping acronyms upright."""
    words = [w if w in ACRONYMS else w.lower() for w in str(code).split("_")]
    if words and words[0] not in ACRONYMS:
        words[0] = words[0].capitalize()
    return " ".join(words)


def yes_no(value) -> str:
    return "Yes" if value else "No"


# ============================================================
# STATIC SECTIONS
# ============================================================

MARK_SVG = (
    '<svg class="qt-mark" viewBox="0 0 7 7" aria-hidden="true">'
    '<path fill="currentColor" fill-rule="evenodd" d="M0 0h7v7H0z M1 1v5h5V1z M2 2h3v3H2z"/></svg>'
)

NAV_HTML = f"""
<header class="qt-nav">
  <a class="qt-brand" href="#top" aria-label="Q-TDE home">{MARK_SVG}
    <span class="qt-word">Q-TDE</span>
    <span class="qt-word-sub">QR tampering &amp; URL risk inspection</span>
  </a>
  <nav class="qt-nav-links" aria-label="Sections">
    <a href="#inspect">Inspect</a>
    <a href="#method">Method</a>
    <a href="#limits" class="qt-hide-sm">Limits</a>
    <a href="{REPO_URL}" class="qt-ext" target="_blank" rel="noopener">GitHub</a>
  </nav>
</header>
"""


@st.cache_data
def hero_figure_html() -> str:
    """A genuine Version 3-M symbol for the repository URL, with an illustrated altered region."""
    qr = qrcode.QRCode(version=3, error_correction=qrcode.constants.ERROR_CORRECT_M, border=4)
    qr.add_data(REPO_URL)
    qr.make(fit=False)
    matrix = [list(row) for row in qr.get_matrix()]
    n = len(matrix)

    r0, c0, rows, cols = 15, 16, 4, 6  # data region, clear of finder, timing and alignment patterns
    altered = set()
    for r in range(r0, r0 + rows):
        for c in range(c0, c0 + cols):
            if (r * 5 + c * 3) % 3 == 0:
                matrix[r][c] = not matrix[r][c]
                altered.add((r, c))

    cell = lambda r, c: f"M{c} {r}h1v1h-1z"
    base = "".join(cell(r, c) for r in range(n) for c in range(n) if matrix[r][c] and (r, c) not in altered)
    alt = "".join(cell(r, c) for r, c in altered if matrix[r][c])

    return f"""
<figure class="qt-figure">
  <svg viewBox="0 0 {n} {n}" role="img" aria-label="Illustration of a Version 3-M QR code with a small region of altered modules">
    <rect width="{n}" height="{n}" fill="#f9fafc"/>
    <path d="{base}" fill="#191d24"/>
    <path class="qt-alter" d="{alt}"/>
    <rect class="qt-alter-box" pathLength="1" x="{c0 - 0.5}" y="{r0 - 0.5}" width="{cols + 1}" height="{rows + 1}"/>
  </svg>
  <ul class="qt-notes">
    <li><span class="qt-key qt-key-alter"></span><span><b>Altered modules</b>
      Physical integrity: do the codewords still satisfy Reed-Solomon, and is there a residual spike above τ?</span></li>
    <li><span class="qt-key qt-key-payload"></span><span><b>Encoded payload</b>
      <code>{esc(REPO_URL)}</code><br/>URL structure: 21 static rules. No request is made.</span></li>
  </ul>
  <figcaption>Illustration. A real Version 3-M symbol with a simulated altered region.</figcaption>
</figure>
"""


HERO_HTML = """
<div class="qt-hero" id="top">
  <h1>Inspect the code. <span>Understand the risk.</span></h1>
  <p class="qt-lede">Q-TDE examines a QR image on two independent axes:
    <strong>whether the code was altered</strong>, and <strong>how its URL is constructed</strong>.</p>
</div>
"""

METHOD_HTML = """
<section class="qt-section" aria-labelledby="method-title">
  <span id="method" class="qt-anchor"></span>
  <h2 class="qt-h2" id="method-title">Two questions, answered separately</h2>
  <p class="qt-intro">Shared preprocessing feeds two branches that never read each other's output.
    The interface reports both and does not merge them into a single verdict.</p>

  <div class="qt-pipe">
    <div class="qt-stage-label">Shared preprocessing</div>
    <div class="qt-shared">
      <div class="qt-step"><h3><i>1</i>Localize</h3>
        <p>OpenCV <code>QRCodeDetector</code>, with an ArUco-based fallback, finds the four symbol corners.</p></div>
      <div class="qt-step"><h3><i>2</i>Rectify</h3>
        <p>Perspective warp to <code>370 × 370</code> px with a normalized four-module quiet zone.</p></div>
      <div class="qt-step"><h3><i>3</i>Decode</h3>
        <p>The payload and symbol version are read; the version decides whether ECC analysis can run.</p></div>
    </div>
    <div class="qt-fork" aria-hidden="true"></div>
    <div class="qt-lanes">
      <div class="qt-lane">
        <div class="qt-lane-head"><h3>Physical integrity</h3><span>image evidence</span></div>
        <ol>
          <li><b>Reed-Solomon syndromes</b>
            <p>Version 3-M only: 70 codewords (44 data, 26 ECC) over GF(2<sup>8</sup>).
              Any non-zero syndrome means sampled codewords were altered.</p></li>
          <li><b>High-frequency residual</b>
            <p>Peak of <code>|I − GaussianBlur(I)| / 255</code> on a 128 px rendering,
              compared against <code>τ = 0.525490</code>.</p></li>
        </ol>
        <div class="qt-out" aria-label="Possible outcomes">
          <span data-tone="danger">{fill}Tampering detected</span>
          <span data-tone="clear">{hollow}No evidence detected</span>
          <span data-tone="muted">{hatch}Unsupported or failed</span>
        </div>
      </div>
      <div class="qt-lane">
        <div class="qt-lane-head"><h3>URL structure</h3><span>payload text only</span></div>
        <ol>
          <li><b>Static parsing</b>
            <p>Scheme, host, path and query are parsed locally. Nothing is fetched, resolved or executed.</p></li>
          <li><b>21 weighted rules</b>
            <p>IP-literal hosts, embedded credentials, shorteners, punycode, dangerous schemes,
              executable downloads, redirect parameters and similar structural signals.</p></li>
        </ol>
        <div class="qt-out" aria-label="Possible outcomes">
          <span data-tone="clear">{hollow}None · 0</span>
          <span data-tone="caution">{fill}Low · 1-24</span>
          <span data-tone="warn">{fill}Moderate · 25-49</span>
          <span data-tone="danger">{fill}High · 50-100</span>
        </div>
      </div>
    </div>
    <p class="qt-merge"><b>No fusion step.</b> A clean URL never upgrades the physical readout,
      and a tampering finding never changes the URL score.</p>
  </div>
</section>
""".format(fill=mod("fill"), hollow=mod("hollow"), hatch=mod("hatch"))

READING_HTML = f"""
<section class="qt-section" aria-labelledby="reading-title">
  <h2 class="qt-h2" id="reading-title">What a result can and cannot tell you</h2>
  <div class="qt-compare">
    <div>
      <h3 data-tone="clear" style="color:var(--tone)">{mod("hollow")}<span style="color:var(--ink)">Can indicate</span></h3>
      <ul>
        <li>Codeword-level alteration in a Version 3-M symbol
          <small>Non-zero Reed-Solomon syndromes after sampling the rectified grid.</small></li>
        <li>A localized visual anomaly
          <small>A high-frequency residual above the calibrated threshold.</small></li>
        <li>URL structure commonly associated with abuse
          <small>Weighted rule matches, reported with the rules that fired.</small></li>
      </ul>
    </div>
    <div>
      <h3 data-tone="danger" style="color:var(--tone)">{mod("hatch")}<span style="color:var(--ink)">Cannot establish</span></h3>
      <ul>
        <li>That a code is genuine
          <small>No evidence detected means neither check fired, nothing more.</small></li>
        <li>That a decoded URL is safe
          <small>A successful decode only means the symbol was readable.</small></li>
        <li>That a URL is malicious
          <small>The score is a structural heuristic, not a probability.</small></li>
        <li>Anything about the live destination
          <small>No requests and no threat-intelligence lookups are made.</small></li>
      </ul>
    </div>
  </div>
</section>
"""

LIMITS_HTML = f"""
<section class="qt-section" aria-labelledby="limits-title">
  <span id="limits" class="qt-anchor"></span>
  <h2 class="qt-h2" id="limits-title">Known limitations</h2>
  <dl class="qt-limits">
    <dt>ECC coverage</dt>
    <dd>Reed-Solomon analysis supports Version 3-M symbols only. Other versions are reported as
      unsupported geometry, although the visual residual still runs.</dd>
    <dt>Threshold calibration</dt>
    <dd><code>τ</code> was calibrated on the project's synthetic, binary QR renders. Photographs with
      glare, blur, noise or print texture can shift the residual in either direction.</dd>
    <dt>Learned models</dt>
    <dd>A Vision Transformer classifier and a convolutional autoencoder were trained and evaluated during
      research. Neither showed useful discrimination, so neither is part of the active decision path.</dd>
    <dt>No network checks</dt>
    <dd>URLs are never opened, resolved or compared against threat-intelligence feeds.
      Reputation and live content are out of scope.</dd>
    <dt>Server webcam</dt>
    <dd>Server capture reads the webcam of the machine running this app. On a remote deployment,
      use the browser camera or upload an image instead.</dd>
  </dl>
</section>
<footer class="qt-foot">
  <span>Q-TDE is a research project. Source and evaluation scripts on
    <a href="{REPO_URL}" target="_blank" rel="noopener">GitHub</a>.</span>
  <span>Run locally <code>streamlit run app/gui.py</code></span>
</footer>
"""


# ============================================================
# RESULT RENDERING
# ============================================================

def readouts_html(res: dict) -> str:
    overall = res.get("overall", {})
    t_status = overall.get("tampering_status", "unavailable")
    u_level = overall.get("url_risk_level", "unavailable")
    url = res.get("url", {})

    t_title, t_tone, t_glyph = PHYSICAL_STATES.get(t_status, (t_status.replace("_", " ").capitalize(), "muted", "hatch"))
    u_title, u_tone, u_glyph = URL_STATES.get(u_level, (u_level.replace("_", " ").capitalize(), "muted", "hatch"))

    score = ""
    if u_level in ("none", "low", "moderate", "high"):
        score = f'<span class="qt-num">{int(url.get("risk_score", 0))}<small> / 100</small></span>'

    return f"""
<div class="qt-readouts" role="status" aria-live="polite">
  <div class="qt-readout qt-reveal" data-tone="{t_tone}" style="--i:0">
    <span class="qt-axis">Physical integrity</span>
    <div class="qt-state"><span class="qt-title">{mod(t_glyph)}<span>{esc(t_title)}</span></span></div>
    <p class="qt-meaning">{physical_meaning(t_status, res)}</p>
  </div>
  <div class="qt-readout qt-reveal" data-tone="{u_tone}" style="--i:1">
    <span class="qt-axis">URL structure</span>
    <div class="qt-state"><span class="qt-title">{mod(u_glyph)}<span>{esc(u_title)}</span></span>{score}</div>
    <p class="qt-meaning">{url_meaning(u_level)}</p>
  </div>
</div>
"""


def physical_panel_html(res: dict, prep, input_img: np.ndarray) -> str:
    tampering = res.get("tampering", {})
    prep_meta = res.get("preprocessing") or {}
    ecc = tampering.get("ecc", {})
    peak = tampering.get("peak_anomaly", {})

    corners = getattr(prep, "corners", None)
    frames = (
        '<div class="qt-frames">'
        + frame_html(input_img, "Input, localized" if corners is not None else "Input", "", corners=corners)
        + frame_html(
            getattr(prep, "rectified_image", None), "Rectified", "No rectified crop: the symbol was not localized.",
            pixelated=True,
        )
        + "</div>"
    )

    evidence = tampering.get("evidence", [])
    evidence_html = ""
    if evidence:
        items = "".join(
            f'<li data-tone="danger">{mod("fill")}<b>{esc(EVIDENCE.get(e, (e, ""))[0])}</b>'
            f'<code>{esc(e)}</code><p>{esc(EVIDENCE.get(e, ("", ""))[1])}</p></li>'
            for e in evidence
        )
        evidence_html = f'<div class="qt-block"><h3 class="qt-h3">Evidence</h3><ul class="qt-findings">{items}</ul></div>'

    ecc_status = ecc.get("status", "unavailable")
    if ecc_status in ("valid_codewords", "integrity_violation_detected"):
        violated = ecc_status == "integrity_violation_detected"
        ecc_body = f"""
<dl class="qt-kv">
  <dt>Codeword check</dt><dd data-tone="{'danger' if violated else 'muted'}">{'Syndrome mismatch' if violated else 'Consistent'}</dd>
  <dt>Non-zero syndromes</dt><dd>{int(ecc.get('nonzero_syndromes', 0))} / 26</dd>
  <dt>Syndrome residual</dt><dd>{int(ecc.get('residual', 0))}</dd>
  <dt>Modules differing from re-encoded payload</dt><dd>{int(ecc.get('module_diff', 0))}</dd>
</dl>"""
    else:
        reason = {
            "unsupported_version": f"Not run: Version {prep_meta.get('detected_version') or '?'} symbol. ECC analysis requires Version 3-M.",
            "decode_failed": "Not run: the payload could not be decoded, so there is no reference to compare against.",
            "extraction_failed": "Codeword extraction failed on the rectified grid.",
        }.get(ecc_status, "Not run: no symbol was available for analysis.")
        ecc_body = f'<p class="qt-note">{esc(reason)}</p>'

    peak_status = peak.get("status", "unavailable")
    tau = float(peak.get("threshold", 0.525490))
    if peak_status in ("within_normal_limits", "anomaly_detected"):
        score = float(peak.get("score", 0.0))
        above = peak_status == "anomaly_detected"
        tone = "danger" if above else "muted"
        peak_body = scale_html(
            score, 1.0, tone,
            ticks=[(tau, "qt-tick-tau")],
            bands=[(tau, 1.0, "danger")],
            labels=[(0, "0"), (tau, f"τ {tau:.4f}"), (1.0, "1.0")],
            aria=f"Peak residual {score:.4f} against threshold {tau:.4f}",
        ) + f"""
<dl class="qt-kv">
  <dt>Peak residual</dt><dd data-tone="{tone}">{score:.6f}</dd>
  <dt>Threshold τ</dt><dd>{tau:.6f}</dd>
  <dt>Result</dt><dd data-tone="{tone}">{'Above τ' if above else 'Below τ'}</dd>
</dl>"""
    else:
        peak_body = f'<p class="qt-note">{"Residual analysis failed." if peak_status == "analysis_failed" else "Not run: no rectified symbol was available."}</p>'

    version = prep_meta.get("detected_version")
    prep_body = f"""
<dl class="qt-kv">
  <dt>Status</dt><dd>{esc(PREP_STATUS.get(prep_meta.get('status', ''), prep_meta.get('status') or 'Unknown'))}</dd>
  <dt>Detected version</dt><dd>{esc(str(version)) if version else 'Unknown'}</dd>
  <dt>Quiet zone normalized</dt><dd>{yes_no(prep_meta.get('has_normalized_quiet_zone'))}</dd>
</dl>"""

    return f"""
<div class="qt-panel qt-reveal" style="--i:2">
  {frames}
  {evidence_html}
  <div class="qt-block"><h3 class="qt-h3">Reed-Solomon syndromes <small>V3-M · 26 ECC</small></h3>{ecc_body}</div>
  <div class="qt-block"><h3 class="qt-h3">High-frequency residual <small>128 px · 5×5 blur</small></h3>{peak_body}</div>
  <div class="qt-block"><h3 class="qt-h3">Preprocessing</h3>{prep_body}</div>
</div>
"""


def url_panel_html(res: dict) -> str:
    url = res.get("url", {})
    level = res.get("overall", {}).get("url_risk_level", "unavailable")
    payload = url.get("url", "")

    payload_html = (
        f'<p class="qt-payload">{esc(payload)}</p>' if payload
        else '<p class="qt-payload qt-none">No payload was decoded.</p>'
    )

    body = ""
    if level in ("none", "low", "moderate", "high"):
        score = int(url.get("risk_score", 0))
        tone = URL_STATES[level][1] if level != "none" else "muted"
        scale = scale_html(
            score, 100, tone,
            ticks=[(25, ""), (50, "")],
            bands=[(0, 25, "caution"), (25, 50, "warn"), (50, 100, "danger")],
            labels=[(0, "0"), (25, "25"), (50, "50"), (100, "100")],
            aria=f"Heuristic score {score} of 100",
        )
        facts = f"""
<dl class="qt-kv qt-kv-2">
  <dt>Scheme</dt><dd>{esc(url.get('scheme') or 'missing')}</dd>
  <dt>HTTPS</dt><dd>{yes_no(url.get('is_https'))}</dd>
  <dt>Host</dt><dd>{esc(url.get('hostname') or 'none')}</dd>
  <dt>IP-literal host</dt><dd>{yes_no(url.get('is_ip_address'))}</dd>
  <dt>Shortener</dt><dd>{yes_no(url.get('uses_shortener'))}</dd>
  <dt>Punycode</dt><dd>{yes_no(url.get('uses_punycode'))}</dd>
  <dt>Length</dt><dd>{int(url.get('url_length', 0))}</dd>
  <dt>Subdomains</dt><dd>{int(url.get('subdomain_count', 0))}</dd>
</dl>"""
        indicators = url.get("risk_indicators", [])
        if indicators:
            rows = "".join(
                f'<li data-tone="{SEVERITY_TONE.get(i.get("severity", ""), "muted")}">{mod("fill")}'
                f'<b>{esc(humanize(i.get("indicator", "")))}</b>'
                f'<span class="qt-w">+{int(i.get("weight", 0))} · {esc(str(i.get("severity", "")).lower())}</span>'
                f'<p>{esc(i.get("description", ""))} <code>{esc(str(i.get("indicator", "")))}</code></p></li>'
                for i in indicators
            )
            matched = f'<ul class="qt-findings">{rows}</ul>'
        else:
            matched = '<p class="qt-note">No structural rule matched.</p>'
        body = f"""
<div class="qt-block"><h3 class="qt-h3">Heuristic score <small>sum of rule weights, capped at 100</small></h3>{scale}</div>
<div class="qt-block"><h3 class="qt-h3">Matched rules <small>{len(indicators)} of 21</small></h3>{matched}</div>
<div class="qt-block"><h3 class="qt-h3">Structure</h3>{facts}</div>
<p class="qt-note">The score is a structural heuristic. It is not a probability that the URL is malicious,
  and no request was made to the destination.</p>"""
    elif level == "non_url_payload":
        body = '<p class="qt-note">The payload does not look like a URL, so no structural rules were applied.</p>'

    return f"""
<div class="qt-panel qt-reveal" style="--i:3">
  <div class="qt-block"><h3 class="qt-h3">Decoded payload</h3>{payload_html}</div>
  {body}
</div>
"""


def alert_html(tone: str, title: str, text: str) -> str:
    return f'<div class="qt-alert" data-tone="{tone}" role="alert">{mod("hatch")}<div><b>{esc(title)}</b> {esc(text)}</div></div>'


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="Q-TDE · QR tampering & URL risk inspection",
    page_icon=":material/qr_code_scanner:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

with open(CSS_PATH, encoding="utf-8") as f:
    html_block(f"<style>{f.read()}</style>")

html_block(NAV_HTML)

ss = st.session_state
ss.setdefault("nonce", 0)


def reset_input():
    ss.nonce += 1
    ss.pop("server_frame", None)
    ss.pop("server_msg", None)


def capture_server_frame():
    ok, frame, msg = camera_processor.capture_frame(device_id=0)
    ss.server_frame = frame if ok else None
    ss.server_msg = None if ok else msg


input_image = None
source_label = ""
read_error = None
res = prep = None

with st.container(key="hero"):
    col_copy, col_tool = st.columns([1.08, 1], gap="large")

    with col_copy:
        html_block(HERO_HTML + hero_figure_html())

    with col_tool:
        with st.container(key="tool"):
            html_block(
                '<span id="inspect" class="qt-anchor"></span>'
                '<div class="qt-tool-head"><h2>Inspect an image</h2><span>PNG · JPG · WebP</span></div>'
            )
            modes = ["Upload", "Camera"] + (["Samples"] if SAMPLES else [])
            mode = st.segmented_control(
                "Input source", modes, default="Upload", required=True,
                key="mode", label_visibility="collapsed", width="stretch",
            )
            n = ss.nonce

            if mode == "Upload":
                uploaded = st.file_uploader("QR image", type=["png", "jpg", "jpeg", "webp"], key=f"upload-{n}")
                html_block('<p class="qt-hint">Analysed by the machine running this app. Decoded URLs are never opened.</p>')
                if uploaded is not None:
                    source_label = uploaded.name
                    file_bytes = np.frombuffer(uploaded.getvalue(), dtype=np.uint8)
                    input_image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE) if file_bytes.size else None
                    if input_image is None:
                        read_error = ("This file could not be read as an image.", "Try a PNG, JPG or WebP export of the QR code.")

            elif mode == "Camera":
                snapshot = st.camera_input("Browser camera", key=f"camera-{n}")
                st.button(
                    "Capture from server webcam", key=f"server-cam-{n}",
                    on_click=capture_server_frame, width="stretch",
                )
                html_block('<p class="qt-hint">Server capture uses the webcam of the machine running this app.</p>')
                if snapshot is not None:
                    source_label = "Browser camera snapshot"
                    input_image = cv2.imdecode(np.frombuffer(snapshot.getvalue(), dtype=np.uint8), cv2.IMREAD_COLOR)
                elif ss.get("server_frame") is not None:
                    source_label = "Server webcam frame"
                    input_image = ss.server_frame
                elif ss.get("server_msg"):
                    read_error = ("Webcam unavailable.", ss.server_msg)

            else:
                labels = list(SAMPLES)
                choice = st.radio(
                    "Validation sample", labels, index=None, key=f"sample-{n}",
                    captions=[SAMPLES[label][1] for label in labels],
                )
                html_block('<p class="qt-hint">Images from the local validation split. Labels describe how each was generated.</p>')
                if choice:
                    source_label = SAMPLES[choice][1]
                    input_image = cv2.imread(SAMPLES[choice][0], cv2.IMREAD_GRAYSCALE)

            if input_image is not None:
                try:
                    with st.spinner("Locating the symbol and running both analyses"):
                        # Same path as inspect_image(image=...), kept explicit so the
                        # rectified crop and corners can be shown alongside the result.
                        prep = preprocess_qr_image(image=input_image, target_size=370)
                        res = inspect_image(preprocessed=prep)
                except Exception as err:
                    read_error = ("Inspection failed.", str(err))
                    res = None

            if read_error:
                html_block(alert_html("warn", *read_error))
            if res is not None:
                html_block(
                    '<p class="qt-status">Inspection complete. '
                    '<a href="#result">View result ↓</a></p>'
                )
            if input_image is not None or read_error:
                st.button("Inspect another image", key=f"reset-{n}", on_click=reset_input, type="tertiary")


if res is not None:
    with st.container(key="result"):
        h, w = input_image.shape[:2]
        html_block(
            f'<span id="result" class="qt-anchor"></span>'
            f'<div class="qt-result-head"><h2 class="qt-h2">Inspection result</h2>'
            f'<p>{esc(source_label)} · {w} × {h} px. Two independent readouts; neither qualifies the other.</p></div>'
            + readouts_html(res)
        )
        col_phys, col_url = st.columns(2, gap="large")
        with col_phys:
            html_block(physical_panel_html(res, prep, input_image))
        with col_url:
            html_block(url_panel_html(res))
        with st.expander("Raw inspection output (JSON)"):
            st.json(res, expanded=2)

html_block(METHOD_HTML + READING_HTML + LIMITS_HTML)
