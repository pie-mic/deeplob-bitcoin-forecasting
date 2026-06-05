# Deep Learning for High-Frequency Bitcoin LOB Forecasting

Comparative study of five architectures for short-horizon mid-price prediction on Bitcoin perpetual futures limit order book data. We benchmark OFI, Plain LSTM, DeepLOB, LiT and a proposed extension — **Attentive DeepLOB** — on a uniform 250 ms clock-time sample of ~1.7M snapshots (BTCUSDT.P, 10–14 January 2023).

---

## Overview

Most prior LOB forecasting work evaluates on event-sampled data, where a new snapshot is recorded only when an order book event occurs. This concentrates observations on moments of market activity and makes the prediction task structurally easier. We re-evaluate all models under a stricter uniform clock-time regime (250 ms intervals), where quiet stretches are preserved and prediction horizons in seconds correspond to real clock time.

After benchmarking four established architectures under this shared protocol, we propose **Attentive DeepLOB**, which retains DeepLOB's strided convolutional front end and inserts two axial self-attention blocks before the LSTM. The intuition: DeepLOB's local convolutional inductive bias is well-suited to LOB spatial structure, but adding global attention over levels and time can capture dependencies the convolutional path misses.

---

## Results

All metrics reported on the held-out test set (~344K sequences, 14 January 2023).

| Model | 1s Acc | 1s Macro F1 | 2s Acc | 2s Macro F1 | 5s Acc | 5s Macro F1 |
|---|---|---|---|---|---|---|
| OFI Logistic | 0.430 | 0.431 | 0.410 | 0.403 | 0.401 | 0.368 |
| Plain LSTM | 0.639 | 0.638 | 0.584 | 0.569 | 0.602 | 0.515 |
| LiT | 0.661 | 0.661 | 0.646 | 0.627 | 0.633 | 0.547 |
| DeepLOB | 0.641 | 0.640 | **0.697** | **0.672** | 0.653 | 0.570 |
| **Attentive DeepLOB** | **0.685** | **0.685** | 0.690 | 0.653 | **0.662** | **0.577** |

Key findings:
- Attentive DeepLOB achieves best accuracy and macro F1 at 1s and 5s horizons; DeepLOB leads at 2s
- All deep models substantially outperform the OFI baseline (40–43% accuracy) by 20+ percentage points
- Fine-tuning only the final dense layers on the first half of the test day recovers 3–6 pp for every model, reaching 70.6% for Attentive DeepLOB at 1s
- Axial time-attention shows no recency bias (ratio ≈ 1×), while gradient saliency shows strong recency bias (ratio ≈ 9×) — suggesting the attention path learns qualitatively different features from the convolutional/LSTM path

---

## Architecture: Attentive DeepLOB

```
Input (50 × 40 × 1)
    ↓
Conv Block 1 — (1×2) stride, 16 ch   [price-volume pairing]
Conv Block 2 — (1×2) stride, 16 ch   [cross-level aggregation]
Conv Block 3 — (1×10), 16 ch         [level features, padding=same]
Inception     — 3 parallel branches, 96 ch  [multi-timescale features]
Conv 1×1      — 32 ch                 [channel reduction]
    ↓
Axial Attention (level axis)          [cross-level dependencies, 4 heads]
Axial Attention (time axis)           [long-range temporal, 4 heads]
    ↓
Reshape (50 × 320)
LSTM (64 units)
Softmax (3 classes: Down / Stationary / Up)
```

Total parameters: 130,931. Compared with 61,907 for DeepLOB and 105,347 for LiT.

---

## Data

**Source:** BTCUSDT.P perpetual futures, Binance, 250 ms uniform snapshots  
**Window:** 10–14 January 2023 (5 full trading days, ~1,723K snapshots)  
**Features:** Top 10 bid and ask price/volume levels → 40 features per snapshot, ordered as {P_ask_i, V_ask_i, P_bid_i, V_bid_i} following Zhang et al. (2019)  
**Normalisation:** Rolling z-score per feature, using up to 5 prior dates as the window  
**Labels:** Smoothed mid-price direction (Down / Stationary / Up) at k ∈ {4, 8, 20} snapshots ahead (1s, 2s, 5s); threshold α tuned per horizon to target ~33% stationary fraction on training data

**Train / Val / Test split (day-based, chronological):**

| Split | Dates | Rows |
|---|---|---|
| Train | 10–12 Jan | ~1,034K |
| Validation | 13 Jan | ~345K |
| Test | 14 Jan | ~345K |

The raw data cannot be redistributed. See `data/README.md` for instructions on sourcing or recreating the dataset.

---

## Models

| Model | Architecture | Params |
|---|---|---|
| OFI Logistic | Cumulative order flow imbalance → multinomial logistic regression | — |
| Plain LSTM | Raw LOB → LSTM (64) → Softmax | 27K |
| DeepLOB | CNN (3 blocks) + Inception → LSTM (64) → Softmax | 62K |
| LiT | Structured patch embedding → 2× Transformer → LSTM (64) → Softmax | 105K |
| Attentive DeepLOB | CNN + Inception → Axial attention (level + time) → LSTM (64) → Softmax | 131K |

---

## Training Protocol

- **Optimiser:** Adam with architecture-specific learning rates (5×10⁻⁴ for LSTM/Attentive DeepLOB, 1×10⁻³ for DeepLOB, 3×10⁻⁴ for LiT)
- **Loss:** Categorical cross-entropy with class-balanced sample weights
- **Early stopping:** patience 8 on validation loss, best-weight restore
- **LR schedule:** Reduce by 0.5 on plateau (patience 5), minimum 10⁻⁶
- **Batch size:** 256; max epochs: 50; lookback window T = 50 (12.5 s of history)
- One independent model trained per architecture per horizon (15 models total)

---

## Repository Structure

```
.
├── DeepLOB_notebook.ipynb     # Full pipeline: preprocessing, all 5 models, evaluation
├── requirements.txt
├── data/
│   └── README.md              # Dataset description and sourcing instructions
├── figures/                   # Key plots exported from the paper
│   ├── results_accuracy_f1.png
│   ├── confusion_matrices.png
│   ├── training_curves.png
│   ├── finetuning_results.png
│   └── axial_attention_maps.png
└── Deep_LOB_Bitcoin.pdf       # Full paper
```

---

## Running the Notebook

The notebook was developed in Google Colab with a GPU runtime. To run it:

1. Upload `DeepLOB_notebook.ipynb` to Google Colab
2. Mount your Google Drive and place the raw data CSV at the path specified in cell 5
3. Install dependencies (most are pre-installed in Colab; see `requirements.txt` for the full list)
4. Run cells in order — sections 0–9 cover data loading through model training, section 10 produces comparison figures, sections 11–12 cover fine-tuning and attention analysis

Training all 15 models (5 architectures × 3 horizons) takes approximately 2–3 hours on a T4 GPU.

---

## References

- Zhang et al. (2019). *DeepLOB: Deep convolutional neural networks for limit order books.* IEEE Transactions on Signal Processing.
- Xiao et al. (2025). *LiT: Limit order book transformer.* Frontiers in Artificial Intelligence.
- Cont, Kukanov & Stoikov (2014). *The price impact of order book events.* Journal of Financial Econometrics.
- Kisiel & Gorse (2022). *Axial-LOB: High-frequency trading with axial attention.* IEEE SSCI.
- Briola et al. (2025). *Deep limit order book forecasting: A microstructural guide.* Quantitative Finance.

---

## Authors

Pietro Micara  
ST456 Deep Learning — MSc Financial Statistics, London School of Economics, 2025–26
