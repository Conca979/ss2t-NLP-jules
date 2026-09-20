# SST-2 Binary Sentiment Classification with Stacked BiLSTM and Luong Attention

This repository implements a clean, strongly-typed PyTorch deep learning pipeline for binary sentiment classification on the Stanford Sentiment Treebank (SST-2) dataset. The core model architecture is a 2-layer stacked Bidirectional LSTM equipped with Luong (General) Attention.

---

## Table of Contents
- [1. Overview & Architecture](#1-overview--architecture)
- [2. Codebase Structure & Explanations](#2-codebase-structure--explanations)
- [3. End-to-End Pipeline Workflow](#3-end-to-end-pipeline-workflow)
  - [3.1 Data Processing & Vocabulary](#31-data-processing--vocabulary)
  - [3.2 Training & Validation](#32-training--validation)
  - [3.3 Inference & Live Demo](#33-inference--live-demo)
- [4. Setup & Execution Commands](#4-setup--execution-commands)
- [5. Experimental Results & Performance Report](#5-experimental-results--performance-report)
- [6. References](#6-references)

---

## 1. Overview & Architecture

The objective is to classify input sentences into binary sentiment categories: `0` (negative) or `1` (positive).

### Model Components (`src/models/bilstm_attention.py`):
1. **Embedding Layer:**
   - Vocabulary size $|V| \le 25,000$ (frequency thresholded).
   - Embedding dimension $d = 128$.
   - `<pad>` index set to 0, `<unk>` index set to 1.
2. **Stacked Bidirectional LSTM Encoder:**
   - 2-layer stacked BiLSTM with hidden dimension $h = 128$.
   - Output representation $H \in \mathbb{R}^{B \times L \times 2h}$ ($2h = 256$).
   - Inter-layer dropout $p = 0.2$.
3. **Luong Attention Mechanism (General Score):**
   - **Query vector $s$:** Concatenation of the top-layer forward and backward final hidden states $[B, 2h]$.
   - **Keys / Values $H$:** Full sequence output of the encoder $[B, L, 2h]$.
   - **Score function:** $\text{score}(s, H_i) = s W_a H_i^T$, where $W_a \in \mathbb{R}^{2h \times 2h}$.
   - **Sequence Masking:** Padding tokens receive $-\infty$ logits before softmax.
   - **Alignment vector $\alpha$:** Softmax normalized attention distribution $\alpha = \text{softmax}(\text{score}) \in \mathbb{R}^{B \times L}$.
   - **Context vector $c$:** Weighted sum $c = \sum_{i=1}^L \alpha_i H_i \in \mathbb{R}^{B \times 2h}$.
4. **Classification Head:**
   - Combined representation $\tilde{h} = \tanh(W_c [c; s]) \in \mathbb{R}^{B \times 2h}$.
   - Linear projection: $\tilde{h} \to 1$ (raw logits evaluated via `BCEWithLogitsLoss`).

---

## 2. Codebase Structure & Explanations

```
.
├── dataset/
│   ├── train.parquet        # Training dataset (~67,349 rows)
│   ├── validation.parquet   # Validation dataset (~872 rows)
│   └── test.parquet         # Test dataset
├── src/
│   ├── data/
│   │   ├── vocab.py         # Vocabulary class: tokenization, mapping, serialization
│   │   └── dataset.py       # SST2Dataset & dynamic padding collate function
│   ├── models/
│   │   └── bilstm_attention.py # LuongAttention & BiLSTMAttentionClassifier modules
│   ├── train.py             # CLI training loop with AdamW and BCEWithLogitsLoss
│   └── predict.py           # Inference CLI returning Pydantic v2 SentimentResponse schema
├── tests/
│   ├── test_model.py        # Unit tests for attention weights, shapes, and padding invariance
│   └── test_pipeline.py     # End-to-end 1-epoch pipeline smoke test
├── demo.py                  # Interactive CLI demo for live terminal classification
├── artifacts/               # Directory storing trained model checkpoints & vocab.json
│   ├── model.pt             # Saved PyTorch model checkpoint weights
│   └── vocab.json           # Serialized vocabulary index mapping
├── pyproject.toml           # Project dependencies managed via `uv`
└── README.md
```

### Key Modules Explained:
- **`src/data/vocab.py`**:
  Defines `Vocabulary` to convert raw text strings to token indices. Includes regex-based lowercasing tokenization, frequency thresholding, special token handling (`<pad>`, `<unk>`), and JSON persistence (`save()` / `load()`).
- **`src/data/dataset.py`**:
  `SST2Dataset` reads parquet, tsv, or csv files into PyTorch-compatible samples. `collate_fn` handles dynamic batch padding up to the maximum sequence length in the batch (capped at `max_len=64`).
- **`src/models/bilstm_attention.py`**:
  `LuongAttention` implements the matrix projection $W_a$ and computes context vectors and attention distributions. `BiLSTMAttentionClassifier` integrates embeddings, packed sequence processing with `nn.LSTM`, attention, and classification linear layers.
- **`src/train.py`**:
  Automated CLI trainer managing dataloaders, training/evaluation metrics, model checkpoints (`artifacts/model.pt`), and vocabulary serialization (`artifacts/vocab.json`).
- **`src/predict.py`**:
  Provides `predict_sentiment()` returning a validated Pydantic v2 `SentimentResponse` model (`label`, `sentiment`, `confidence`, `attention_weights`).
- **`demo.py`**:
  Interactive terminal application that loads `artifacts/model.pt` and `artifacts/vocab.json` to allow users to input text interactively in Windows, macOS, or Linux terminals and view live sentiment predictions and token attention weight distributions.

---

## 3. End-to-End Pipeline Workflow

### 3.1 Data Processing & Vocabulary
1. **Tokenization:** Text is converted to lowercase and tokenized using regular expressions (`\w+|[^\w\s]`).
2. **Vocabulary Construction:** Built from training sentences with frequency counting up to a maximum vocabulary size of 25,000.
3. **Dynamic Padding:** Within each batch, sequences are truncated to `max_len=64` and padded to the max length of the batch with `padding_idx=0`.

### 3.2 Training & Validation
- **Optimizer:** `AdamW(lr=1e-3)`
- **Loss Function:** `nn.BCEWithLogitsLoss()`
- **Batching:** `DataLoader` with batch size 64.
- **Checkpointing:** Model weights (`model.pt`) and vocabulary (`vocab.json`) are automatically serialized into `artifacts/`.
- **Evaluation:** Measures Loss and Binary Accuracy ($I(\sigma(\text{logit}) \ge 0.5)$) after each epoch. Best checkpoints based on validation accuracy are saved.

### 3.3 Inference & Live Demo
Run real-time sentiment analysis interactively in terminal or via CLI argument:
```bash
# Interactive Live Terminal Classification Demo
uv run python demo.py

# Non-interactive CLI Single Input Run
uv run python demo.py --text "An absolute masterpiece with incredible performances."
```

Example output:
```
--- Prediction Result ---
Sentiment:  [POSITIVE] (POSITIVE)
Confidence: 99.55%
Token Attention Weights:
  an              : 0.2416 | #############
  absolute        : 0.1140 | ######
  masterpiece     : 0.1031 | #####
  with            : 0.1058 | #####
  incredible      : 0.2545 | ##############
  performances    : 0.1253 | ######
  .               : 0.0557 | ##
--------------------------------------------------
```

---

## 4. Setup & Execution Commands

### Environment Setup (`uv`)
```bash
uv sync
```

### Verification & Linting Commands
```bash
# 1. Code Formatting & Linting
uv run ruff check .

# 2. Strict Type Checking
uv run mypy src/ demo.py

# 3. Unit and Integration Tests
uv run pytest tests/ -v

# 4. Training Model Pipeline
uv run python -m src.train --epochs 2 --batch-size 64

# 5. Interactive Live Classification Demo
uv run python demo.py

# 6. Single Input Sentiment Inference Command
uv run python -m src.predict --text "This movie was absolutely fantastic and engaging."
```

---

## 5. Experimental Results & Performance Report

- **Total Trainable Parameters:** 2,626,689 (~2.6M)
- **Training Epochs:** 2 epochs
- **Hardware / Device:** CPU (x86_64)
- **Training Duration:** 266.14 seconds (~4.4 minutes)
- **Training & Validation Metrics:**
  - **Epoch 1:** Train Loss: `0.4288` | Train Acc: `79.26%` | Val Loss: `0.6169` | Val Acc: `76.38%`
  - **Epoch 2:** Train Loss: `0.2161` | Train Acc: `91.47%` | Val Loss: `0.5998` | Val Acc: `77.29%`
- **Best Validation Accuracy:** **77.29%**

---

## 6. References

1. **Luong, M.-T., Pham, H., & Manning, C. D. (2015).** *Effective Approaches to Attention-based Neural Machine Translation.* Empirical Methods in Natural Language Processing (EMNLP). [arXiv:1508.04025](https://arxiv.org/abs/1508.04025).
2. **Socher, R., et al. (2013).** *Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank.* Empirical Methods in Natural Language Processing (EMNLP) (SST-2 Dataset).
3. **PyTorch Documentation:** `torch.nn.LSTM`, `torch.nn.utils.rnn.pack_padded_sequence`, `torch.nn.BCEWithLogitsLoss`.
4. **Pydantic v2 Documentation:** Data validation and settings management using Python type hints.
