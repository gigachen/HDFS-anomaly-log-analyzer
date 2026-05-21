"""
TraceReader — HDFS Log Anomaly Detector (Streamlit app)
Group 19 · BINUS NLP Final Project

Paste raw HDFS log text → the app parses it into block-level event sequences
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

NORMAL_COLOR  = "#4C72B0"
ANOMALY_COLOR = "#DD8452"

LOG_LINE_RE = re.compile(r"^(\d+)\s+(\d+)\s+(\d+)\s+(\w+)\s+([\w$.]+):\s*(.*)$")
BLOCK_RE    = re.compile(r"blk_-?\d+")

st.set_page_config(page_title="TraceReader · HDFS Anomaly Detector",
                   page_icon="🔍", layout="wide")


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
st.title("🔍 TraceReader — HDFS Log Anomaly Detector")
st.caption("Group 19 · BINUS NLP · Paste raw HDFS log lines and detect anomalous blocks.")

templates_df, matchers = load_templates()
models, status = load_models()

with st.sidebar:
    st.header("Models")
    for name, s in status.items():
        icon = "✅" if "loaded" in s or "trained" in s else "⚠️"
        st.write(f"{icon} **{name}** — {s}")
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

st.subheader("1 · Input raw HDFS log")
col_a, col_b = st.columns([3, 1])
with col_b:
    st.caption("Sample = 1 complete normal block + 1 complete anomalous block.")
    if st.button("Load sample (complete blocks)", use_container_width=True):
        if os.path.exists("sample_log.txt"):
            st.session_state["log_text"] = open("sample_log.txt", errors="replace").read()
        else:
            st.warning("sample_log.txt not found in the project folder.")
    st.caption("⚠️ Blocks must be **complete** (full lifecycle). Truncated blocks "
               "look abnormal and get false-flagged.")
    uploaded = st.file_uploader("…or upload a .log / .txt file", type=["log", "txt"])
    if uploaded is not None:
        st.session_state["log_text"] = uploaded.read().decode("utf-8", errors="replace")

with col_a:
    log_text = st.text_area(
        "Paste log lines here",
        value=st.session_state.get("log_text", ""),
        height=240,
        placeholder="081109 203518 143 INFO dfs.DataNode$DataXceiver: "
                    "Receiving block blk_-1608999687919862906 src: /10.250.19.102:54106 "
                    "dest: /10.250.19.102:50010",
    )

run = st.button("🚀 Detect anomalies", type="primary", use_container_width=True)

if run:
    if not log_text.strip():
        st.error("Please paste some log text or load a sample first.")
        st.stop()

    block_events, stats = parse_log(log_text, matchers)
    if stats["blocks"] == 0:
        st.error("No HDFS blocks (blk_…) could be parsed. "
                 "Check that the input is raw HDFS log text.")
        st.stop()

    blocks, occ, seqs, lens = build_features(block_events)

    # Predictions
    occ_scaled = models["scaler"].transform(occ)
    p_lr  = models["lr"].predict_proba(occ_scaled)[:, 1]
    iso_score = -models["iso"].score_samples(occ)
    iso_flag  = models["iso"].predict(occ) == -1

    p_lstm = predict_lstm(models["lstm"], seqs, lens) if "lstm" in models else None
    dl_rate = deeplog_violation_rate(models["deeplog"], seqs, lens) if "deeplog" in models else None

    verdict_prob = p_lr if primary == "LR" else p_lstm
    verdict = verdict_prob >= threshold

    # ----- Summary -----
    st.subheader("2 · Results")
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
        f"{primary} P(anomaly)": np.round(verdict_prob, 4),
        "Verdict": np.where(verdict, "🔴 ANOMALY", "🟢 normal"),
        "IsoForest score": np.round(iso_score, 3),
    })
    if p_lstm is not None and primary != "LSTM":
        table["LSTM P(anomaly)"] = np.round(p_lstm, 4)
    if dl_rate is not None:
        table["DeepLog violation rate"] = np.round(dl_rate, 3)

    table = table.sort_values(f"{primary} P(anomaly)", ascending=False).reset_index(drop=True)

    st.dataframe(table, use_container_width=True, height=360)

    flagged = table[table["Verdict"].str.contains("ANOMALY")]
    if len(flagged):
        st.error(f"⚠️ {len(flagged)} anomalous block(s) detected.")
    else:
        st.success("✅ No anomalous blocks detected at the current threshold.")

    # ----- Download -----
    st.download_button(
        "⬇️ Download results (CSV)",
        table.to_csv(index=False).encode(),
        file_name="anomaly_predictions.csv",
        mime="text/csv",
    )

    # ----- Per-block detail -----
    with st.expander("Inspect a single block's event sequence"):
        pick = st.selectbox("Block", blocks)
        st.code(" → ".join(block_events[pick]), language=None)
