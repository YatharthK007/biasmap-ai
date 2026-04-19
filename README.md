# 🗺️ BiasMap AI — Dataset Bias Cartography & Compliance Auditor

> **"A GPS for Data Fairness"** — A pre-training dataset audit tool that maps bias terrain, calculates legal compliance scores, and suggests algorithmic fixes before a single epoch of training begins.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?style=flat-square&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![DPDP](https://img.shields.io/badge/DPDP%20Act%202023-Compliant-brightgreen?style=flat-square)

---

## 🧠 The Problem

Most AI bias tools audit models **post-hoc** (after training). **BiasMap AI** audits the data **before the first epoch** — because bias baked into training data cannot be fixed by the model itself.

### Specific Challenges Solved:

1. **The Intersectional Gap:** Traditional tools check `Gender` or `Geography` in isolation. We identify the *"Rural Woman" desert* — where both attributes exist but their intersection is absent.
2. **The Compliance Hurdle:** India's **DPDP Act 2023** and NITI Aayog guidelines require "Fair Processing." Developers now have a tool to quantify this.
3. **The Black Box of Pre-training:** Get a **Data Nutrition Label** before you burn GPU hours on a biased dataset.

---

## ✨ Features (4 Core USPs)

| # | Feature | What It Does |
|---|---------|-------------|
| 1 | **DPDP Compliance Score** | KL Divergence vs. benchmark → Alpha Grade (A+ to F) |
| 2 | **Intersectional Desert Detection** | Multi-dim binning → flags < 1% representation gaps |
| 3 | **Auto-Fix Engine** | SMOTE (synthetic samples) or Strategic Under-sampling |
| 4 | **Ghost Bias Detector** | Decision Tree Gini Importance reveals proxy bias columns |

---

## 🖥️ Tech Stack

| Library | Role |
|---|---|
| **Python 3.10+** | Core orchestration logic |
| **Streamlit** | Dark-mode, local-first UI |
| **Pandas + NumPy** | High-speed data processing |
| **SciPy** | Statistical engine (Entropy & KL Divergence) |
| **UMAP-learn** | 2D dimensionality reduction for "Bias Terrain" mapping |
| **Plotly** | Interactive cluster visualisation |
| **Scikit-learn** | Decision Tree shadow models for Ghost Bias detection |
| **Imbalanced-learn** | SMOTE & stratified under-sampling |
| **FPDF2** | Automated PDF audit report generation |

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/YatharthK007/BiasMap-AI.git
cd BiasMap-AI
```

### 2. Set up Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies & run

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`. All processing is done **locally** — no data ever leaves your machine.

---

## 📂 Project Structure
### BiasMap-AI/
- **app.py**  
  Main Streamlit UI & pipeline orchestration  

- **auditor.py**  
  KL Divergence & Shannon Entropy engine  

- **mitigation.py**  
  SMOTE & under-sampling logic  

- **report.py**  
  PDF generator (Data Nutrition Label)  

- **requirements.txt**  
  Project dependencies  

- **README.md**  
  Documentation  
---

## 📊 How It Works

1. **Schema Verification** — Identify Sensitive, Target, and Excluded columns via a confirmation UI before any processing begins.
2. **UMAP Terrain Map** — Visualise dataset clusters to spot physical "deserts" in representation space.
3. **Audit Engine** — Calculates Shannon Entropy for diversity and KL Divergence for fairness against a benchmark.
4. **Auto-Fix** — Balance the data using synthetic over-sampling (SMOTE) or strategic cluster pruning.
5. **Ghost Bias Detector** — Trains a shadow Decision Tree to surface features acting as hidden proxies for sensitive attributes.
6. **PDF Export** — Downloads a professional 3-page audit report (the "Data Nutrition Label").

---

## 📐 Mathematical Foundation

### 1. Shannon Entropy

Measures the diversity of a sensitive attribute $X$:

$$H(X) = -\sum_{i=1}^{n} P(x_i) \log_2(P(x_i))$$

**Goal:** Maximise diversity toward $\log_2(n)$.

### 2. KL Divergence

Measures the distance from a "Fair" benchmark distribution:

$$D_{KL}(P \parallel Q) = \sum_{x \in \mathcal{X}} P(x) \log\left(\frac{P(x)}{Q(x)}\right)$$

A score of **0** = perfect alignment with the benchmark.

### 3. Laplace Smoothing

Prevents division-by-zero for missing categories (e.g., a category present in the benchmark but absent in the dataset):

$$\hat{P}(x) = \frac{\text{count}(x) + \alpha}{N + \alpha \cdot |k|} \quad (\alpha = 10^{-6})$$

---

## 🗂️ Demo Datasets

To prove the tool's effectiveness, the initial demo uses three world-standard open-source datasets:

| Dataset | Primary Bias Target | Audit Goal |
|---|---|---|
| **UCI Adult Income** | Sex & Race | Demonstrate skew in high-income representation |
| **German Credit Dataset** | Age & Foreign Worker Status | Show how "Age" acts as a proxy for creditworthiness bias |
| **India Census Subset** | Rural/Urban & State | Prove the NITI Aayog compliance scoring logic |

---

## ⚖️ Compliance & Privacy

Designed to support compliance with:

- **Digital Personal Data Protection (DPDP) Act, 2023** (India)
- **NITI Aayog** Responsible AI principles

**Privacy-First:** This tool uses no cloud APIs, no external trackers, and no telemetry. Your data never leaves your machine.

---

## 📈 Success Metrics

| Metric | Target |
|---|---|
| UMAP Projection Latency | < 60s on 100k rows |
| Crash Rate (high NaN density) | 0% |
| Shannon Entropy Improvement (Auto-Fix) | > 20% gain |

---

## 📄 License

MIT License — free to use, modify, and distribute.