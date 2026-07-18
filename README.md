# Dimensional Aspect Sentiment Regression (DimASR) via Transformers 🧠💬

Dimensional Aspect Sentiment Regression using fine-tuned BERT and Huber Loss.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c.svg)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow)
![NLP](https://img.shields.io/badge/Domain-NLP-purple)

## 📌 Project Overview
This repository contains a neural regression system designed for the **Dimensional Aspect Sentiment Regression (DimASR)** task. Unlike traditional discrete polarity classification (Positive/Negative/Neutral), this system predicts continuous, real-valued emotional coordinates for **Valence** (pleasantness) and **Arousal** (intensity) on a scale of 1.00 to 9.00.

## 🔬 Methodology & Architecture

1. **Contextual Transformer Model:**
   * Fine-tuned a `bert-base-cased` architecture (110M parameters) using Hugging Face and PyTorch.
   * Replaced the standard sequence classification head with a dual-output linear layer to regress continuous Valence and Arousal scores.
   * **Attention Routing:** Formatted inputs using specific token delimiters (`[CLS] Text [SEP] Aspect [SEP]`) to force self-attention matrices to compute target-specific context and resolve multi-aspect cross-talk.

2. **Mathematical Optimization (Huber Loss):**
   * **The Challenge:** The dataset exhibited severe Gaussian distribution bias clustered around the neutral 5.0 score. Standard MSE loss caused the network to conservatively predict mean values, ignoring high-arousal outliers.
   * **The Solution:** Implemented PyTorch's `SmoothL1Loss` (Huber Loss) to penalize mean-regression and reliably map extreme emotional coordinates. Post-inference NumPy bounding matrices were used to strictly enforce the [1.00, 9.00] limits.

## 📊 Dataset
* Evaluated on a filtered English-language multilingual corpus.
* **Corpus Size:** 2,779 training samples, 340 validation samples, and 1,504 test samples.

## 🏆 Key Results

The contextual Transformer was benchmarked against a lexical Support Vector Regressor (SVR) utilizing TF-IDF vectors (1-2 n-grams, 3000 max features). Performance was evaluated using **Root Mean Square Error (RMSE)**.

| Model Architecture | Feature Representation | Validation RMSE |
| :--- | :--- | :--- |
| SVR (Baseline) | TF-IDF (Lexical) | 1.1465 |
| **BERT-base** | **Transformer (Contextual)** | **0.8912** |

* **Performance Delta:** The transition to the contextual Transformer yielded a **22% reduction in RMSE**. 
* **Significance:** Achieving a sub-1.0 RMSE on an 8-point continuous scale signifies an average deviation of just ~11% from true emotional coordinates, effectively approaching the noise floor of inter-annotator disagreement.

## 🛠️ Usage & Installation

```bash
# Clone the repository
git clone https://github.com/LamiaT/DimASR-transformer-regression.git
cd DimASR-transformer-regression

# Install requirements
pip install -r requirements.txt
