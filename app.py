"""
app.py
------
SkinSense — Skin Disease Detection for African Skin Types
Streamlit single-page application with premium UI.

Run with:
    streamlit run app.py
"""

import io
import os
import subprocess
import sys

import streamlit as st
from PIL import Image

# ── First-run: generate model files if missing ─────────────────────────────────

if not (os.path.exists("model_architecture.json") and os.path.exists("model_weights.h5")):
    with st.spinner("First-run setup: building demo model weights (once only) …"):
        subprocess.run([sys.executable, "model.py"], check=True)

from predict import predict  # noqa: E402

# ── Clinical notes ─────────────────────────────────────────────────────────────

CLINICAL_NOTES = {
    "Eczema": "Eczema (atopic dermatitis) presents as inflamed, itchy skin and may appear darker or as hyperpigmented patches on African skin types.",
    "Tinea Versicolor": "Tinea versicolor is a common fungal infection that causes discoloured patches which may appear lighter or darker than surrounding skin.",
    "Acne Vulgaris": "Acne vulgaris frequently causes post-inflammatory hyperpigmentation on darker skin tones, which can persist long after the active lesion resolves.",
    "Vitiligo": "Vitiligo causes depigmented patches that are often more visually prominent on darker skin types due to high contrast with surrounding pigmented skin.",
    "Psoriasis": "Psoriasis on darker skin may present as violet or grey plaques rather than the pink-red colour typical of lighter skin presentations.",
    "Melanoma": "Melanoma on darker skin types most commonly occurs on the palms, soles, and nail beds; early detection is critical for survival.",
    "Acne Keloidalis Nuchae": "Acne keloidalis nuchae presents as firm papules and keloid plaques at the nape of the neck and predominantly affects individuals with darker skin types.",
    "Seborrheic Dermatitis": "Seborrheic dermatitis on darker skin may present with perifollicular scaling and hypopigmentation rather than the classic erythema seen on lighter skin.",
}

CONDITION_ICONS = {
    "Eczema": "🔴",
    "Tinea Versicolor": "🟤",
    "Acne Vulgaris": "🟠",
    "Vitiligo": "⚪",
    "Psoriasis": "🟣",
    "Melanoma": "⚫",
    "Acne Keloidalis Nuchae": "🟡",
    "Seborrheic Dermatitis": "🟢",
}

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="SkinSense AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu, footer { visibility: hidden; }
header { visibility: hidden; }
header [data-testid="stToolbar"] { visibility: hidden; }
[data-testid="collapsedControl"] { visibility: visible !important; display: flex !important; }
.block-container { padding-top: 1.8rem; padding-bottom: 2rem; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0f1b2d 0%, #1a2e4a 100%);
    border-right: 1px solid rgba(255,255,255,0.07);
}
[data-testid="stSidebar"] * { color: #e2eaf5 !important; }
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.12) !important; }
[data-testid="stSidebar"] h1 { font-size:1.25rem !important; color:#fff !important; font-weight:700 !important; }
[data-testid="stSidebar"] h3 { font-size:0.75rem !important; letter-spacing:0.09em; text-transform:uppercase; color:#7fa8d4 !important; margin-top:1.2rem !important; }
.sidebar-badge { display:inline-block; background:rgba(255,255,255,0.08); border:1px solid rgba(255,255,255,0.13); border-radius:20px; padding:3px 10px; font-size:0.73rem; margin:3px 3px 3px 0; color:#c8daf0; }
.sidebar-disclaimer { background:rgba(239,68,68,0.13); border:1px solid rgba(239,68,68,0.3); border-radius:10px; padding:10px 14px; font-size:0.78rem; color:#fca5a5 !important; margin-top:1rem; line-height:1.5; }

/* Hero */
.hero-header { background:linear-gradient(135deg,#0f1b2d 0%,#1a3a5c 60%,#0e4d8a 100%); border-radius:18px; padding:2rem 2.4rem; margin-bottom:1.6rem; border:1px solid rgba(255,255,255,0.08); position:relative; overflow:hidden; }
.hero-header::before { content:""; position:absolute; top:-70px; right:-70px; width:220px; height:220px; background:radial-gradient(circle,rgba(56,139,253,0.2) 0%,transparent 70%); border-radius:50%; }
.hero-title { font-size:1.9rem; font-weight:800; color:#fff; margin:0 0 0.25rem 0; letter-spacing:-0.02em; }
.hero-subtitle { font-size:0.9rem; color:#90b8d8; margin:0 0 1.1rem 0; }
.hero-pills { display:flex; gap:8px; flex-wrap:wrap; }
.hero-pill { background:rgba(255,255,255,0.1); border:1px solid rgba(255,255,255,0.18); border-radius:20px; padding:4px 14px; font-size:0.76rem; color:#cce0f5; font-weight:500; }

/* Upload */
.upload-label { font-size:0.75rem; font-weight:700; letter-spacing:0.07em; text-transform:uppercase; color:#6b7f99; margin-bottom:0.35rem; }
[data-testid="stFileUploader"] { border:2px dashed #2a4a6b !important; border-radius:14px !important; padding:0.8rem !important; }

/* Buttons */
.stButton > button { border-radius:10px !important; font-weight:600 !important; font-size:0.88rem !important; padding:0.52rem 1.3rem !important; transition:all 0.18s ease !important; border:none !important; }
.stButton > button[kind="primary"] { background:linear-gradient(135deg,#1d6bd8,#3b82f6) !important; color:#fff !important; box-shadow:0 4px 14px rgba(59,130,246,0.35) !important; }
.stButton > button[kind="primary"]:hover { transform:translateY(-1px) !important; box-shadow:0 6px 20px rgba(59,130,246,0.45) !important; }
.stButton > button[kind="secondary"] { background:#f1f5f9 !important; color:#475569 !important; border:1px solid #e2e8f0 !important; }
.stButton > button[kind="secondary"]:hover { background:#e2e8f0 !important; }

/* Result card */
.result-card { background:#fff; border-radius:16px; padding:1.6rem 1.8rem; border:1px solid #e8eef6; box-shadow:0 2px 16px rgba(0,0,0,0.06); margin-bottom:1rem; }
.result-label { font-size:0.7rem; font-weight:700; letter-spacing:0.1em; text-transform:uppercase; color:#94a3b8; margin-bottom:0.45rem; }
.result-disease { font-size:1.7rem; font-weight:800; color:#0f172a; letter-spacing:-0.02em; margin-bottom:0.25rem; }
.result-conf-label { font-size:0.82rem; color:#64748b; margin-bottom:0.5rem; }
.conf-bar-bg { background:#e8f0fe; border-radius:999px; height:10px; overflow:hidden; margin-bottom:1.1rem; }
.conf-bar-fill { height:10px; border-radius:999px; background:linear-gradient(90deg,#2563eb,#60a5fa); }
.rank-row { display:flex; align-items:center; gap:10px; padding:8px 0; border-bottom:1px solid #f1f5f9; }
.rank-row:last-child { border-bottom:none; }
.rank-badge { width:24px; height:24px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:0.7rem; font-weight:700; flex-shrink:0; }
.rank-1 { background:#fef9c3; color:#854d0e; }
.rank-2 { background:#f1f5f9; color:#475569; }
.rank-3 { background:#f1f5f9; color:#64748b; }
.rank-name { flex:1; font-size:0.88rem; font-weight:500; color:#1e293b; }
.rank-pct  { font-size:0.86rem; font-weight:600; color:#3b82f6; }
.rank-minibar-bg { width:72px; height:6px; background:#e8f0fe; border-radius:999px; overflow:hidden; }
.rank-minibar-fill { height:6px; border-radius:999px; background:#93c5fd; }

/* Clinical note & disclaimer */
.clinical-note { background:#eff6ff; border-left:4px solid #3b82f6; border-radius:0 10px 10px 0; padding:0.85rem 1rem; font-size:0.86rem; color:#1e3a5f; line-height:1.6; margin-bottom:0.8rem; }
.clinical-note strong { color:#1d4ed8; }
.disclaimer { background:#fff7ed; border:1px solid #fed7aa; border-radius:10px; padding:0.75rem 1rem; font-size:0.8rem; color:#7c2d12; display:flex; align-items:flex-start; gap:8px; line-height:1.5; }

/* Empty states */
.empty-state { text-align:center; padding:2.5rem 1rem; color:#94a3b8; }
.empty-icon  { font-size:2.8rem; margin-bottom:0.8rem; }
.empty-title { font-size:1rem; font-weight:600; color:#64748b; }
.empty-sub   { font-size:0.83rem; margin-top:0.3rem; }
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────

if "result"         not in st.session_state: st.session_state.result = None
if "uploaded_bytes" not in st.session_state: st.session_state.uploaded_bytes = None

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("# 🔬 SkinSense")
    st.markdown("**AI-powered skin disease detection** optimised for African skin types.")
    st.markdown("---")
    st.markdown("### About")
    st.markdown(
        "SkinSense uses MobileNetV2 transfer learning to classify skin lesion images "
        "into 8 common dermatological conditions — designed as a **decision-support tool**, "
        "not a replacement for clinical diagnosis."
    )
    st.markdown("### Detectable conditions")
    badges = "".join(f'<span class="sidebar-badge">{c}</span>' for c in CLINICAL_NOTES)
    st.markdown(badges, unsafe_allow_html=True)
    st.markdown(
        '<div class="sidebar-disclaimer">⚠️ <strong>Prototype notice</strong><br>'
        'This build uses demonstration weights. Predictions are illustrative only. '
        'Replace <code>model_weights.h5</code> with trained weights before clinical use.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown("""
    <div style="background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.11);
                border-radius:10px;padding:10px 12px;font-size:0.72rem;line-height:1.7;color:#c8daf0;">
      <div style="font-size:0.65rem;letter-spacing:0.08em;text-transform:uppercase;
                  color:#7fa8d4;margin-bottom:4px;">Developed by</div>
      <div style="font-weight:700;font-size:0.82rem;color:#ffffff;">Worlu Chimenem Lucky</div>
      <div style="color:#90b8d8;">U/2022/22410</div>
      <div style="margin-top:6px;font-size:0.65rem;letter-spacing:0.08em;text-transform:uppercase;
                  color:#7fa8d4;">Department</div>
      <div style="color:#c8daf0;">Computer Science</div>
      <div style="color:#90b8d8;font-size:0.71rem;">Ignatius Ajuru University<br>of Education (IAUE)</div>
    </div>
    """, unsafe_allow_html=True)
    st.caption("SkinSense v1.0 · 2025")

# ── Hero ───────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero-header">
  <p class="hero-title">🔬 SkinSense</p>
  <p class="hero-subtitle">Skin Disease Detection for African Skin Types — Powered by Transfer Learning</p>
  <div class="hero-pills">
    <span class="hero-pill">8 Conditions</span>
    <span class="hero-pill">MobileNetV2</span>
    <span class="hero-pill">CPU Inference</span>
    <span class="hero-pill">Decision Support</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Two-column layout ──────────────────────────────────────────────────────────

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown('<div class="upload-label">Upload skin image</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("upload", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    if uploaded_file:
        st.session_state.uploaded_bytes = uploaded_file.read()

    if st.session_state.uploaded_bytes:
        image = Image.open(io.BytesIO(st.session_state.uploaded_bytes))
        st.image(image, use_container_width=True, caption="Uploaded image")

        b1, b2 = st.columns([3, 1])
        with b1:
            analyse = st.button("🔍  Analyse Image", type="primary", use_container_width=True)
        with b2:
            clear = st.button("✖ Clear", type="secondary", use_container_width=True)

        if analyse:
            with st.spinner("Running inference …"):
                st.session_state.result = predict(image)
        if clear:
            st.session_state.result = None
            st.session_state.uploaded_bytes = None
            st.rerun()
    else:
        st.markdown("""
        <div class="empty-state">
          <div class="empty-icon">📷</div>
          <div class="empty-title">No image uploaded yet</div>
          <div class="empty-sub">Select a JPG or PNG skin image above to get started.</div>
        </div>""", unsafe_allow_html=True)

with col_right:
    if st.session_state.result:
        res  = st.session_state.result
        icon = CONDITION_ICONS.get(res["top_class"], "🔵")
        conf = res["confidence"]

        top3_html = "".join(
            f'<div class="rank-row">'
            f'<div class="rank-badge rank-{i+1}">{i+1}</div>'
            f'<div class="rank-name">{r["class"]}</div>'
            f'<div class="rank-minibar-bg"><div class="rank-minibar-fill" style="width:{r["confidence"]:.1f}%"></div></div>'
            f'<div class="rank-pct">{r["confidence"]:.1f}%</div>'
            f'</div>'
            for i, r in enumerate(res["top3"])
        )

        st.markdown(f"""
        <div class="result-card">
          <div class="result-label">Top prediction</div>
          <div class="result-disease">{icon} {res["top_class"]}</div>
          <div class="result-conf-label">Confidence: <strong>{conf:.1f}%</strong></div>
          <div class="conf-bar-bg"><div class="conf-bar-fill" style="width:{conf:.1f}%"></div></div>
          <div class="result-label">Top 3 predictions</div>
          {top3_html}
        </div>
        <div class="clinical-note">
          <strong>🩺 Clinical note:</strong> {CLINICAL_NOTES.get(res["top_class"], "")}
        </div>
        <div class="disclaimer">
          <span>⚠️</span>
          <span>This result is for <strong>decision support only</strong>. Always confirm with clinical examination by a qualified healthcare professional.</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="empty-state" style="margin-top:2rem">
          <div class="empty-icon">📊</div>
          <div class="empty-title">Results will appear here</div>
          <div class="empty-sub">Upload an image and click Analyse Image to see predictions.</div>
        </div>""", unsafe_allow_html=True)

# ── Page footer ────────────────────────────────────────────────────────────────

st.markdown("""
<div style="margin-top:2.5rem;padding-top:1.2rem;border-top:1px solid #e2e8f0;
            text-align:center;color:#94a3b8;font-size:0.78rem;line-height:1.8;">
  <span style="font-weight:600;color:#64748b;">SkinSense</span> &nbsp;·&nbsp;
  Developed by <span style="font-weight:600;color:#475569;">Worlu Chimenem Lucky</span>
  &nbsp;<span style="color:#cbd5e1;">(U/2022/22410)</span>&nbsp;·&nbsp;
  Department of Computer Science &nbsp;·&nbsp;
  Ignatius Ajuru University of Education (IAUE)
</div>
""", unsafe_allow_html=True)
