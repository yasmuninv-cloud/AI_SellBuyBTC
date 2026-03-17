

# **AI_SellBuyBTC — Transformer‑Based Inference of BTC Buy/Sell Signals**

This repository provides a compact and reproducible inference pipeline for generating **model‑derived BUY and SELL signals** for Bitcoin (BTC) using a pretrained Transformer architecture.  
The objective of this public release is to demonstrate the inference workflow, model architecture, and signal‑generation process without exposing proprietary training methodology.

The **training code, data‑labeling logic, and feature‑engineering pipeline are maintained in a private repository** and are intentionally excluded from this distribution.

---

## **1. Overview**

The system implements a lightweight Transformer encoder that processes a set of **11 slope‑based statistical features** derived from daily BTC price data.  
The model outputs two independent binary classification signals:

- **BUY signal**  
- **SELL signal**

These outputs are obtained through a forward pass of the pretrained model and visualized on top of the BTC price series using an interactive Plotly chart.

This repository is intended for:

- researchers evaluating model‑driven signal inference  
- practitioners exploring Transformer architectures for financial time‑series  
- users interested in reproducing the inference pipeline without access to training internals  

---

## **2. Repository Structure**

```
AI_SellBuyBTC/
│
├── AI_SellBuyBTC.py              # Main inference script
├── multi_transformer_weights.pth # Pretrained Transformer weights
└── README.md
```

---

## **3. Running the Inference Pipeline**

### **3.1 Install Dependencies**

```
pip install torch numpy pandas plotly requests
```

### **3.2 Execute the Script**

```
python AI_SellBuyBTC.py
```

### **3.3 What the Script Performs**

1. Loads cached BTC daily data or fetches it from Binance’s public API  
2. Computes 11 rolling OLS‑based slope features  
3. Loads the pretrained Transformer model and its weights  
4. Performs forward inference to obtain BUY/SELL signal outputs  
5. Visualizes the results in an interactive Plotly chart  

The entire workflow is CPU‑compatible and does not require GPU acceleration.

---

## **4. Model Architecture**

The model is a compact Transformer encoder consisting of:

- **Input layer:** 11‑dimensional feature vector  
- **Transformer encoder:**  
  - 1 encoder layer  
  - multi‑head self‑attention  
  - feed‑forward expansion  
- **Two linear output heads:**  
  - BUY signal head  
  - SELL signal head  

The architecture is fully defined inside `AI_SellBuyBTC.py` for transparency.  
The pretrained weights are stored in `multi_transformer_weights.pth`.

The training process (oversampling, loss weighting, feature engineering, and label construction) is **not included** and remains private.

---

## **5. Output Visualization**

The script produces an interactive Plotly chart containing:

- BTC daily closing price  
- Model‑derived BUY signals (cyan markers)  
- Model‑derived SELL signals (orange markers)

This visualization enables qualitative assessment of the model’s signal‑generation behavior across the full BTC historical dataset.

---

## **6. System Specifications**

The inference pipeline was executed under the following environment:

```
Python: 3.11.14 (Anaconda)
Torch: 2.9.1+cpu
CUDA available: False
Platform: Windows-10-10.0.26200-SP0
```

The model is lightweight and runs efficiently on CPU‑only systems.

---

## **7. Notes and Limitations**

- This repository contains **inference‑only code**.  
- Training logic, data labeling, and proprietary feature‑engineering methods are **not included**.  
- The model outputs **binary classification signals**, not financial advice.  
- The pretrained weights are provided solely for reproducibility and academic/technical demonstration.

---

## **8. License**
License Code: MY-FullRights-Proprietary-1.0

