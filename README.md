# TraceReader — HDFS Log Analyzer Powered by NLP

Anomaly detection on the LogHub **HDFS_v1** benchmark for COMP6885001 (NLP), BINUS University — **Group 19**.

Raw Hadoop logs are parsed into 29 event templates (E1–E29), aggregated into per-block traces, and classified as normal or anomalous by four models spanning frequency-based and sequence-based, supervised and unsupervised paradigms.

## Results (anomaly-class F1)

| Model | Random split | Temporal split |
|---|---|---|
| Logistic Regression | 0.981 | 0.998 |
| Isolation Forest | 0.618 | 0.755 |
| LSTM | 0.991 | 0.996 |
| DeepLog (unsupervised) | — | 0.210 |

Key finding: the sequence models genuinely encode event order (shuffle attack: DeepLog flags 90% of order-corrupted blocks, LSTM 50%, LR 0%), yet HDFS's labeled anomalies are mostly **frequency-based**, so a simple count model matches a deep sequence model.

## Repository layout

| Path | Description |
|---|---|
| `TraceReader_HDFS_Anomaly_Detection.ipynb` | All experiments (4 models, 2 splits, analysis) |
| `app.py` | Streamlit app — raw log text → per-block anomaly predictions |
| `preprocess_hdfs.py` | Raw `HDFS.log` → the five `preprocessed/` artifacts |
| `preprocessed/` | Input features (incl. order-aligned `HDFS_aligned.npz`) |
| `models/` | Saved checkpoints (`.pkl` sklearn, `.pth` PyTorch) |
| `sample_log.txt` | Demo input: 1 complete normal + 1 complete anomaly block |
| `REPORT.md` | Written report (academic structure) |
| `METHODOLOGY.md` | Detailed start-to-finish methodology |

## Quick start

```bash
pip install -r requirements.txt

# 1) Reproduce experiments
jupyter lab TraceReader_HDFS_Anomaly_Detection.ipynb

# 2) Run the application
streamlit run app.py
#    → click "Load sample (complete blocks)" → "Detect anomalies"
```

GPU is used automatically if available (CUDA). On Apple Silicon, MPS can be enabled; CPU works but the LSTM is slow (~20 min vs ~2 min on a T4).

## Regenerating data from raw logs (optional)

```bash
python preprocess_hdfs.py --log_file HDFS.log --output_dir preprocessed_real
```

## Dataset & citation

LogHub HDFS_v1: https://github.com/logpai/loghub

- W. Xu et al., "Detecting Large-Scale System Problems by Mining Console Logs," SOSP 2009.
- J. Zhu et al., "Loghub: A Large Collection of System Log Datasets for AI-driven Log Analytics," ISSRE 2023.

## Team

| Member | NIM |
|---|---|
| Dominicius Francis Ang Gunadi | 2802561293 |
| Evelyn Ang | 2802472060 |
| Felicia Pardamean | 2802544873 |
