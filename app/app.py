"""
TraceReader — HDFS Log Anomaly Detector (Streamlit app)
Group 19 · BINUS NLP Final Project

Paste raw HDFS log text  the app parses it into block-level event sequences
(E1–E29 templates), then predicts which blocks are anomalous using the trained
models. No labels required for prediction.

Run:
    pip install -r requirements.txt
    streamlit run app.py
"""
import os
import re
import pickle

import numpy as np
import pandas as pd
import streamlit as st

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
DATA_DIR  = "preprocessed/"
CKPT_DIR  = "models/"
EVENT_COLS = [f"E{i}" for i in range(1, 30)]
MAX_LEN   = 100
WINDOW    = 10
TOPK      = 3
DEVICE    = torch.device("cpu")

NORMAL_COLOR  = "#5e6ad2"   # Linear lavender-blue (accent)
ANOMALY_COLOR = "#DD8452"   # warm orange retained for anomaly contrast

LOG_LINE_RE = re.compile(r"^(\d+)\s+(\d+)\s+(\d+)\s+(\w+)\s+([\w$.]+):\s*(.*)$")
BLOCK_RE    = re.compile(r"blk_-?\d+")

st.set_page_config(page_title="TraceReader · HDFS Anomaly Detector",
                   layout="wide")


# --------------------------------------------------------------------------- #
# Linear Portfolio Design System: theme
# Tokens mirror the "Linear Portfolio Design System" project (colors_and_type.css):
# near-black canvas #010102, lavender-blue accent #5e6ad2, Inter + JetBrains Mono.
# --------------------------------------------------------------------------- #
def inject_linear_theme():
    st.markdown(
        """
        <style>
        @import url("https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap");

        :root {
          --color-primary: #5e6ad2;
          --color-on-primary: #ffffff;
          --color-primary-hover: #828fff;
          --color-canvas: #010102;
          --color-surface-1: #0f1011;
          --color-surface-2: #141516;
          --color-surface-3: #18191a;
          --color-hairline: #23252a;
          --color-hairline-strong: #34343a;
          --color-ink: #f7f8f8;
          --color-ink-muted: #d0d6e0;
          --color-ink-subtle: #8a8f98;
          --color-success: #27a644;
          --font-display: "Inter","SF Pro Display",-apple-system,system-ui,"Segoe UI",Roboto,sans-serif;
          --font-text: "Inter","SF Pro Text",-apple-system,system-ui,"Segoe UI",Roboto,sans-serif;
          --font-mono: "JetBrains Mono",ui-monospace,"SF Mono",Menlo,Consolas,monospace;
          --radius-md: 8px;
          --radius-lg: 12px;
        }

        /* ---- Canvas & base type ---- */
        .stApp {
          background: var(--color-canvas);
          color: var(--color-ink);
          font-family: var(--font-text);
          letter-spacing: -0.05px;
          -webkit-font-smoothing: antialiased;
        }
        [data-testid="stHeader"] { background: transparent; }
        .block-container { padding-top: 2.5rem; max-width: 1180px; }

        h1, h2, h3, h4 {
          font-family: var(--font-display) !important;
          color: var(--color-ink) !important;
          letter-spacing: -0.8px;
          font-weight: 600 !important;
        }
        p, label, span, .stMarkdown { color: var(--color-ink-muted); }
        code, kbd, pre, .stCode, [data-testid="stCode"] {
          font-family: var(--font-mono) !important;
        }
        a { color: var(--color-ink) !important; text-decoration: none; }
        a:hover { color: var(--color-primary-hover) !important; }
        ::selection { background: var(--color-primary); color: #fff; }

        /* ---- Hero header ---- */
        .tr-eyebrow {
          font-family: var(--font-text);
          font-size: 13px; font-weight: 500; letter-spacing: 0.4px;
          text-transform: uppercase; color: var(--color-ink-subtle);
          margin: 0 0 10px 0;
        }
        .tr-hero-title {
          font-family: var(--font-display);
          font-size: 44px; font-weight: 600; line-height: 1.1;
          letter-spacing: -1.6px; color: var(--color-ink); margin: 0;
        }
        .tr-hero-title .accent { color: var(--color-primary); }
        .tr-hero-sub {
          font-family: var(--font-text); font-size: 18px; font-weight: 400;
          line-height: 1.5; color: var(--color-ink-subtle);
          margin: 12px 0 4px 0; max-width: 680px;
        }
        .tr-rule {
          border: 0; border-top: 1px solid var(--color-hairline);
          margin: 28px 0 8px 0;
        }

        /* ---- Sidebar ---- */
        section[data-testid="stSidebar"] {
          background: var(--color-surface-1);
          border-right: 1px solid var(--color-hairline);
        }
        section[data-testid="stSidebar"] * { color: var(--color-ink-muted); }
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 { color: var(--color-ink) !important; }

        /* ---- Buttons ---- */
        .stButton > button, .stDownloadButton > button {
          font-family: var(--font-text); font-weight: 500; font-size: 14px;
          border-radius: var(--radius-md);
          border: 1px solid var(--color-hairline-strong);
          background: var(--color-surface-2);
          color: var(--color-ink);
          transition: all 120ms ease;
        }
        .stButton > button:hover, .stDownloadButton > button:hover {
          border-color: var(--color-primary);
          background: var(--color-surface-3);
          color: var(--color-ink);
        }
        .stButton > button[kind="primary"] {
          background: var(--color-primary);
          border-color: var(--color-primary);
          color: var(--color-on-primary);
          box-shadow: 0 1px 0 0 rgba(255,255,255,0.06) inset;
        }
        .stButton > button[kind="primary"]:hover {
          background: var(--color-primary-hover);
          border-color: var(--color-primary-hover);
          color: #fff;
        }
        .stButton > button:focus, .stDownloadButton > button:focus,
        .stButton > button[kind="primary"]:focus {
          box-shadow: 0 0 0 2px rgba(94,105,210,0.5) !important;
        }

        /* ---- Inputs (textarea, selectbox, slider, uploader) ---- */
        .stTextArea textarea, .stTextInput input {
          background: var(--color-surface-1) !important;
          color: var(--color-ink) !important;
          font-family: var(--font-mono) !important;
          font-size: 13px !important;
          border-radius: var(--radius-md) !important;
        }
        [data-baseweb="textarea"], [data-baseweb="input"],
        [data-baseweb="select"] > div {
          background: var(--color-surface-1) !important;
          border-color: var(--color-hairline) !important;
          border-radius: var(--radius-md) !important;
        }
        [data-baseweb="textarea"]:focus-within,
        [data-baseweb="select"] > div:focus-within {
          border-color: var(--color-primary) !important;
          box-shadow: 0 0 0 2px rgba(94,105,210,0.4) !important;
        }
        [data-testid="stFileUploaderDropzone"] {
          background: var(--color-surface-1);
          border: 1px dashed var(--color-hairline-strong);
          border-radius: var(--radius-md);
        }
        .stSlider [data-baseweb="slider"] [role="slider"] {
          background: var(--color-primary) !important;
        }

        /* ---- Metric cards (Linear surface + hairline) ---- */
        [data-testid="stMetric"] {
          background: var(--color-surface-1);
          border: 1px solid var(--color-hairline);
          border-radius: var(--radius-lg);
          padding: 16px 18px;
          box-shadow: 0 1px 0 0 rgba(255,255,255,0.04) inset;
        }
        [data-testid="stMetricLabel"] {
          color: var(--color-ink-subtle) !important;
          text-transform: uppercase; letter-spacing: 0.4px; font-size: 12px;
        }
        [data-testid="stMetricValue"] {
          font-family: var(--font-display) !important;
          color: var(--color-ink) !important; letter-spacing: -1px;
        }

        /* ---- Dataframe ---- */
        [data-testid="stDataFrame"] {
          border: 1px solid var(--color-hairline);
          border-radius: var(--radius-lg);
          overflow: hidden;
        }

        /* ---- Expander ---- */
        [data-testid="stExpander"] {
          background: var(--color-surface-1);
          border: 1px solid var(--color-hairline);
          border-radius: var(--radius-lg);
        }
        [data-testid="stExpander"] summary:hover { color: var(--color-primary-hover); }

        /* ---- Alerts ---- */
        [data-testid="stAlert"] { border-radius: var(--radius-md); }

        /* ---- Section labels ---- */
        .tr-section {
          display: flex; align-items: baseline; gap: 12px;
          margin: 40px 0 18px 0;
        }
        .tr-section-num {
          font-family: var(--font-mono); font-size: 12px; font-weight: 500;
          color: var(--color-primary); letter-spacing: 0.5px;
        }
        .tr-section-title {
          font-family: var(--font-display); font-size: 22px; font-weight: 600;
          letter-spacing: -0.6px; color: var(--color-ink); margin: 0;
        }

        /* ---- Sidebar model status ---- */
        .tr-status {
          display: flex; align-items: center; gap: 10px;
          padding: 7px 0; font-size: 13px;
        }
        .tr-status .dot {
          width: 7px; height: 7px; border-radius: 9999px; flex: none;
        }
        .tr-status .dot.ok   { background: var(--color-success); }
        .tr-status .dot.warn { background: var(--color-ink-subtle); }
        .tr-status-name { color: var(--color-ink); font-weight: 500; }
        .tr-status-msg  { color: var(--color-ink-subtle); margin-left: auto;
                          font-size: 12px; font-family: var(--font-mono); }
        .tr-sidebar-label {
          font-family: var(--font-text); font-size: 12px; font-weight: 500;
          letter-spacing: 0.4px; text-transform: uppercase;
          color: var(--color-ink-subtle); margin: 4px 0 8px 0;
        }

        /* ---- Motion: clean entrance on results ---- */
        @keyframes tr-rise {
          from { opacity: 0; transform: translateY(8px); }
          to   { opacity: 1; transform: translateY(0); }
        }
        @keyframes tr-fade {
          from { opacity: 0; }
          to   { opacity: 1; }
        }
        [data-testid="stMain"] [data-testid="stMetric"] {
          animation: tr-rise 380ms cubic-bezier(0.22, 1, 0.36, 1) both;
        }
        [data-testid="stMain"] [data-testid="stColumn"]:nth-child(1) [data-testid="stMetric"] { animation-delay: 0ms; }
        [data-testid="stMain"] [data-testid="stColumn"]:nth-child(2) [data-testid="stMetric"] { animation-delay: 70ms; }
        [data-testid="stMain"] [data-testid="stColumn"]:nth-child(3) [data-testid="stMetric"] { animation-delay: 140ms; }
        [data-testid="stMain"] [data-testid="stColumn"]:nth-child(4) [data-testid="stMetric"] { animation-delay: 210ms; }
        [data-testid="stMain"] [data-testid="stDataFrame"] {
          animation: tr-rise 440ms cubic-bezier(0.22, 1, 0.36, 1) both;
          animation-delay: 240ms;
        }
        [data-testid="stMain"] [data-testid="stAlert"] {
          animation: tr-rise 380ms cubic-bezier(0.22, 1, 0.36, 1) both;
          animation-delay: 320ms;
        }
        [data-testid="stMain"] .stDownloadButton,
        [data-testid="stMain"] [data-testid="stExpander"] {
          animation: tr-fade 500ms ease both;
          animation-delay: 360ms;
        }
        .tr-section, .tr-empty { animation: tr-fade 320ms ease both; }
        @media (prefers-reduced-motion: reduce) {
          [data-testid="stMain"] *, .tr-section, .tr-empty { animation: none !important; }
        }

        /* ---- Empty state ---- */
        .tr-empty {
          background: var(--color-surface-1);
          border: 1px dashed var(--color-hairline-strong);
          border-radius: var(--radius-lg);
          padding: 44px 24px; text-align: center;
        }
        .tr-empty-title {
          font-family: var(--font-display); font-size: 16px; font-weight: 500;
          color: var(--color-ink-muted); margin: 0 0 6px 0;
          letter-spacing: -0.3px;
        }
        .tr-empty-sub {
          font-family: var(--font-text); font-size: 13px;
          color: var(--color-ink-subtle); margin: 0;
        }

        /* ---- Captions: quieter ---- */
        [data-testid="stCaptionContainer"], .stCaption {
          color: var(--color-ink-subtle) !important;
        }

        /* ---- Divider ---- */
        hr { border-top: 1px solid var(--color-hairline) !important; }
        [data-testid="stSidebar"] hr { margin: 18px 0; }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_linear_theme()


# --------------------------------------------------------------------------- #
# Model architectures (must match the notebook exactly to load state_dicts)
# --------------------------------------------------------------------------- #
class HDFSLSTMClassifier(nn.Module):
    def __init__(self, vocab_size=30, embed_dim=32, hidden_dim=64,
                 num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x, lengths):
        emb = self.embedding(x)
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True,
                                      enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        h_last = self.dropout(h_n[-1])
        return torch.sigmoid(self.fc(h_last)).squeeze(1)


class DeepLog(nn.Module):
    def __init__(self, vocab=30, embed=32, hidden=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.emb  = nn.Embedding(vocab, embed, padding_idx=0)
        self.lstm = nn.LSTM(embed, hidden, num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.out  = nn.Linear(hidden, vocab)

    def forward(self, x):
        e = self.emb(x)
        h, _ = self.lstm(e)
        return self.out(h[:, -1])


# --------------------------------------------------------------------------- #
# Cached resources: templates + models
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load_templates():
    df = pd.read_csv(os.path.join(DATA_DIR, "HDFS.log_templates.csv"))
    matchers = []
    for _, row in df.iterrows():
        parts = row["EventTemplate"].split("[*]")
        pattern = ".*?".join(re.escape(p) for p in parts)
        matchers.append((row["EventId"], re.compile("^" + pattern + "$")))
    return df, matchers


@st.cache_resource(show_spinner="Loading / training models (one-time)...")
def load_models():
    """Load saved checkpoints if present; otherwise train the fast models
    (LR + Isolation Forest) from preprocessed/ so the app always works."""
    status = {}
    models = {}

    have_pkls = all(os.path.exists(os.path.join(CKPT_DIR, f)) for f in
                    ["lr_temporal.pkl", "scaler_temporal.pkl", "iso_temporal.pkl"])

    if have_pkls:
        models["lr"]     = pickle.load(open(os.path.join(CKPT_DIR, "lr_temporal.pkl"), "rb"))
        models["scaler"] = pickle.load(open(os.path.join(CKPT_DIR, "scaler_temporal.pkl"), "rb"))
        models["iso"]    = pickle.load(open(os.path.join(CKPT_DIR, "iso_temporal.pkl"), "rb"))
        status["LR / IsoForest"] = "loaded from models/"
    else:
        df_occ = pd.read_csv(os.path.join(DATA_DIR, "Event_occurrence_matrix.csv"))
        X = df_occ[EVENT_COLS].values.astype(np.float32)
        y = (df_occ["Label"] == "Fail").astype(int).values
        scaler = StandardScaler().fit(X)
        lr = LogisticRegression(class_weight="balanced", max_iter=1000,
                                solver="lbfgs", random_state=42).fit(scaler.transform(X), y)
        iso = IsolationForest(contamination=0.03, n_estimators=100,
                              random_state=42, n_jobs=-1).fit(X)
        models["lr"], models["scaler"], models["iso"] = lr, scaler, iso
        status["LR / IsoForest"] = "trained on launch from preprocessed/"

    lstm_path = os.path.join(CKPT_DIR, "lstm_temporal.pth")
    if os.path.exists(lstm_path):
        m = HDFSLSTMClassifier().to(DEVICE)
        m.load_state_dict(torch.load(lstm_path, map_location=DEVICE))
        m.eval()
        models["lstm"] = m
        status["LSTM"] = "loaded from models/"
    else:
        status["LSTM"] = "not found — run notebook SAVE cell to enable"

    dl_path = os.path.join(CKPT_DIR, "deeplog.pth")
    if os.path.exists(dl_path):
        m = DeepLog().to(DEVICE)
        m.load_state_dict(torch.load(dl_path, map_location=DEVICE))
        m.eval()
        models["deeplog"] = m
        status["DeepLog"] = "loaded from models/"
    else:
        status["DeepLog"] = "not found — run notebook SAVE cell to enable"

    return models, status


# --------------------------------------------------------------------------- #
# Preprocessing: raw text -> block-level event sequences
# --------------------------------------------------------------------------- #
def parse_log(text, matchers):
    """Return {block_id: [event_id, ...]} in chronological order, plus stats."""
    block_events = {}
    total = matched = malformed = 0
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        total += 1
        m = LOG_LINE_RE.match(line)
        if not m:
            malformed += 1
            continue
        content = m.group(6)
        event_id = None
        for eid, pat in matchers:
            if pat.match(content):
                event_id = eid
                break
        if event_id is None:
            continue
        matched += 1
        # dedupe block IDs per line: one log line = one event per distinct block
        for blk in set(BLOCK_RE.findall(content)):
            block_events.setdefault(blk, []).append(event_id)
    stats = dict(total=total, matched=matched, malformed=malformed,
                 blocks=len(block_events))
    return block_events, stats


def build_features(block_events):
    blocks = list(block_events.keys())
    # occurrence matrix
    occ = np.zeros((len(blocks), 29), dtype=np.float32)
    for r, b in enumerate(blocks):
        for e in block_events[b]:
            occ[r, int(e[1:]) - 1] += 1
    # padded sequences for LSTM/DeepLog
    seqs = np.zeros((len(blocks), MAX_LEN), dtype=np.int64)
    lens = np.zeros(len(blocks), dtype=np.int64)
    for r, b in enumerate(blocks):
        ids = [int(e[1:]) for e in block_events[b]][:MAX_LEN]
        seqs[r, :len(ids)] = ids
        lens[r] = max(len(ids), 1)
    return blocks, occ, seqs, lens


def predict_lstm(model, seqs, lens, batch=512):
    probs = []
    with torch.no_grad():
        for i in range(0, len(seqs), batch):
            x = torch.tensor(seqs[i:i + batch], dtype=torch.long)
            n = torch.tensor(lens[i:i + batch], dtype=torch.long)
            probs.extend(model(x, n).numpy())
    return np.array(probs)


def deeplog_violation_rate(model, seqs, lens, k=TOPK, window=WINDOW):
    rates = []
    with torch.no_grad():
        for r in range(len(seqs)):
            L = int(lens[r])
            seq = seqs[r, :L]
            if L <= window:
                rates.append(0.0)
                continue
            xs = np.lib.stride_tricks.sliding_window_view(seq, window)[:-1]
            ys = seq[window:]
            logits = model(torch.tensor(np.ascontiguousarray(xs), dtype=torch.long))
            topk = torch.topk(logits, k, dim=1).indices.numpy()
            viol = sum(ys[i] not in topk[i] for i in range(len(ys)))
            rates.append(viol / len(ys))
    return np.array(rates)


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
def section_label(num, title):
    st.markdown(
        f'<div class="tr-section"><span class="tr-section-num">{num}</span>'
        f'<h2 class="tr-section-title">{title}</h2></div>',
        unsafe_allow_html=True,
    )


st.markdown(
    """
    <div class="tr-eyebrow">Group 19 · BINUS NLP · HDFS_v1</div>
    <h1 class="tr-hero-title">TraceReader <span class="accent">HDFS Log</span> Anomaly Detector</h1>
    <p class="tr-hero-sub">Paste raw HDFS log lines. TraceReader parses them into block-level
    event sequences (E1–E29) and flags anomalous blocks with the trained models, no labels required.</p>
    <hr class="tr-rule" />
    """,
    unsafe_allow_html=True,
)

templates_df, matchers = load_templates()
models, status = load_models()

with st.sidebar:
    st.markdown('<div class="tr-sidebar-label">Models</div>', unsafe_allow_html=True)
    for name, s in status.items():
        ok = "loaded" in s or "trained" in s
        short = "loaded" if "loaded" in s else "trained" if "trained" in s else "not found"
        st.markdown(
            f'<div class="tr-status"><span class="dot {"ok" if ok else "warn"}"></span>'
            f'<span class="tr-status-name">{name}</span>'
            f'<span class="tr-status-msg">{short}</span></div>',
            unsafe_allow_html=True,
        )
    st.divider()
    primary = st.selectbox(
        "Decision model (verdict)",
        [m for m in ["LR", "LSTM"] if (m == "LR") or ("lstm" in models)],
        help="LR is the most reliable (F1≈0.99). LSTM available if checkpoint loaded.",
    )
    threshold = st.slider("Anomaly threshold", 0.0, 1.0, 0.5, 0.01,
                          help="A block is flagged anomalous if its probability ≥ threshold.")
    st.divider()
    with st.expander("Event template reference (E1–E29)"):
        st.dataframe(templates_df, height=300, use_container_width=True)

section_label("01", "Input raw HDFS log")
col_a, col_b = st.columns([3, 1])
with col_b:
    st.caption("Sample = 1 complete normal block + 1 complete anomalous block.")
    if st.button("Load sample (complete blocks)", use_container_width=True):
        if os.path.exists("sample_log.txt"):
            st.session_state["log_text"] = open("sample_log.txt", errors="replace").read()
        else:
            st.warning("sample_log.txt not found in the project folder.")
    st.caption("Blocks must be **complete** (full lifecycle). Truncated blocks "
               "look abnormal and get false-flagged.")
    uploaded = st.file_uploader("…or upload a .log / .txt file", type=["log", "txt"])
    if uploaded is not None:
        st.session_state["log_text"] = uploaded.read().decode("utf-8", errors="replace")

with col_a:
    log_text = st.text_area(
        "Paste log lines here",
        value=st.session_state.get("log_text", ""),
        height=320,
        placeholder="081109 203518 143 INFO dfs.DataNode$DataXceiver: "
                    "Receiving block blk_-1608999687919862906 src: /10.250.19.102:54106 "
                    "dest: /10.250.19.102:50010",
    )

run = st.button("Detect anomalies", type="primary", use_container_width=True)

# Pressing the button only *computes* predictions and stashes them in session
# state. Rendering happens below from that state, so widgets in the results
# section (e.g. the block inspector) can rerun the script without wiping the
# results. Threshold / model changes re-render live from the cached arrays.
if run:
    if not log_text.strip():
        st.error("Please paste some log text or load a sample first.")
    else:
        block_events, stats = parse_log(log_text, matchers)
        if stats["blocks"] == 0:
            st.error("No HDFS blocks (blk_…) could be parsed. "
                     "Check that the input is raw HDFS log text.")
        else:
            with st.spinner(f"Scanning {stats['blocks']:,} block(s) for anomalies…"):
                blocks, occ, seqs, lens = build_features(block_events)

                # Predictions (computed once per detect run)
                occ_scaled = models["scaler"].transform(occ)
                p_lr  = models["lr"].predict_proba(occ_scaled)[:, 1]
                iso_score = -models["iso"].score_samples(occ)
                p_lstm = predict_lstm(models["lstm"], seqs, lens) if "lstm" in models else None
                dl_rate = deeplog_violation_rate(models["deeplog"], seqs, lens) if "deeplog" in models else None

            st.session_state["results"] = dict(
                block_events=block_events, blocks=blocks, stats=stats,
                p_lr=p_lr, p_lstm=p_lstm, iso_score=iso_score, dl_rate=dl_rate,
            )

# ----- Render results from session state (survives reruns) -----
res = st.session_state.get("results")
section_label("02", "Results")

if not res:
    st.markdown(
        '<div class="tr-empty">'
        '<p class="tr-empty-title">No results yet</p>'
        '<p class="tr-empty-sub">Paste or load HDFS log lines above, then press '
        '<strong>Detect anomalies</strong> to see block-level predictions here.</p>'
        '</div>',
        unsafe_allow_html=True,
    )
else:
    block_events = res["block_events"]
    blocks       = res["blocks"]
    stats        = res["stats"]
    p_lr         = res["p_lr"]
    p_lstm       = res["p_lstm"]
    iso_score    = res["iso_score"]
    dl_rate      = res["dl_rate"]

    verdict_prob = p_lstm if (primary == "LSTM" and p_lstm is not None) else p_lr
    decision = "LSTM" if (primary == "LSTM" and p_lstm is not None) else "LR"
    verdict = verdict_prob >= threshold

    # ----- Summary -----
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Lines parsed", f"{stats['matched']:,}/{stats['total']:,}")
    c2.metric("Blocks found", f"{stats['blocks']:,}")
    c3.metric("Flagged anomalous", f"{int(verdict.sum()):,}")
    rate = verdict.mean() * 100 if len(verdict) else 0
    c4.metric("Anomaly rate", f"{rate:.1f}%")

    # ----- Per-block table -----
    table = pd.DataFrame({
        "BlockId": blocks,
        "Events": [len(block_events[b]) for b in blocks],
        f"{decision} P(anomaly)": np.round(verdict_prob, 4),
        "Verdict": np.where(verdict, "ANOMALY", "normal"),
        "IsoForest score": np.round(iso_score, 3),
    })
    if p_lstm is not None and decision != "LSTM":
        table["LSTM P(anomaly)"] = np.round(p_lstm, 4)
    if dl_rate is not None:
        table["DeepLog violation rate"] = np.round(dl_rate, 3)

    table = table.sort_values(f"{decision} P(anomaly)", ascending=False).reset_index(drop=True)

    st.dataframe(table, use_container_width=True, height=360)

    flagged = table[table["Verdict"].str.contains("ANOMALY")]
    if len(flagged):
        st.error(f"{len(flagged)} anomalous block(s) detected.")
    else:
        st.success("No anomalous blocks detected at the current threshold.")

    # ----- Download -----
    st.download_button(
        "Download results (CSV)",
        table.to_csv(index=False).encode(),
        file_name="anomaly_predictions.csv",
        mime="text/csv",
    )

    # ----- Per-block detail -----
    with st.expander("Inspect a single block's event sequence"):
        pick = st.selectbox("Block", blocks, key="inspect_block")
        st.code("  ".join(block_events[pick]), language=None)
