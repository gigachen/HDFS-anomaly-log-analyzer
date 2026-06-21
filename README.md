# TraceReader — Code

HDFS log anomaly detection with NLP. Group 19 · BINUS University · NLP Final Project (COMP6885001) · 2026.

This folder has two self-contained parts — the **experiment** (reproduces the paper's results) and the **app** (interactive demo). Each bundles the data/checkpoints it needs, so both run as-is.

```
code/
├── experiment/    reproduce the results
└── app/           interactive Streamlit demo
```

## experiment/

Trains and evaluates the four detectors (Logistic Regression, Isolation Forest, LSTM, DeepLog) on HDFS_v1 under random and temporal splits, plus the shuffle experiment.

```
experiment/
├── TraceReader_experiment.ipynb   main notebook (run top to bottom)
├── preprocess_hdfs.py             rebuilds preprocessed/ from raw HDFS.log (only if needed)
├── preprocessed/                  input features the notebook reads (~135 MB)
└── models/                        saved checkpoints (LOAD cell skips retraining)
```

Run:
```bash
cd experiment
jupyter lab TraceReader_experiment.ipynb
```
- `DATA_DIR` is set to the bundled `preprocessed/`; the Colab `drive.mount` cell is guarded, so it runs locally too.
- Full run trains all models. To skip training, run the setup/data cells (0–13) then the **LOAD** cell to load checkpoints from `models/`.
- Requires the standard scientific Python stack plus `torch` (uses CUDA if available, else CPU).

## app/

Streamlit app that loads the trained checkpoints and classifies pasted raw HDFS log text with all four models.

```
app/
├── app.py
├── requirements.txt   streamlit, torch, scikit-learn, pandas, numpy
├── sample_log.txt     for the "Load sample" button
├── preprocessed/      HDFS.log_templates.csv (parsing)
└── models/            checkpoints
```

Run:
```bash
cd app
pip install -r requirements.txt
streamlit run app.py
```
Paste raw HDFS log lines (or click **Load sample**), then hit **Detect anomalies**.

## Notes

- Dataset: LogHub **HDFS_v1** — 11,175,629 log lines grouped into 575,061 block traces (2.93% anomalous).
- The two large source files `HDFS.npz` and `Event_traces.csv` are intentionally omitted (not read by the notebook).
- The bundled `deeplog.pth` is the original checkpoint; the app's DeepLog scoring matches it (self-consistent).
- If your submission system has a size cap, you may delete `experiment/preprocessed/` and point `DATA_DIR` at the dataset link in the report.
