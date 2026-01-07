# EEG-to-Language Decoding  
**Word-Level Classification from EEG Signals during Natural Reading**

## Overview

This project explores whether **explicit temporal modeling** improves word-level decoding from EEG signals recorded during natural reading.

Given an EEG segment corresponding to a single word, the task is to predict **which word was read**, using a fixed vocabulary of 500 common English words. We compare four neural architectures combining CNNs, recurrent models (GRU), and attention mechanisms.

Despite strong motivation for temporal modeling, we find that **architectural complexity provides only marginal gains**, and all models plateau below the reported BELT baseline.

---

## Motivation

Reading is not instantaneous — it unfolds over time:

- Visual processing (≈50–100 ms)
- Lexical recognition (≈100–200 ms)
- Semantic processing (≈200–400 ms)
- Contextual integration (≈400–600 ms)

The BELT model combines CNNs and attention but lacks an explicit notion of *temporal progression*.  
Our core question:

> **Does adding a recurrent model (GRU), which maintains temporal memory, improve EEG-to-word decoding?**

---

## Dataset

### ZUCO 2.0

We use the **ZUCO 2.0** EEG reading dataset.

**Dataset properties**
- 18 native English speakers
- Natural reading of movie reviews and Wikipedia articles
- 105 EEG channels
- Sampling rate: 500 Hz
- ~79,509 total word instances

**Why ZUCO**
- Preprocessed EEG (artifact removal already done)
- Word-level alignment between EEG and text
- More subjects than BELT (18 vs 5)
- Naturalistic reading task

---

## Data Extraction Pipeline

The ZUCO data is stored in MATLAB `.mat` files with deeply nested HDF5 reference structures.  
Word-level EEG signals are accessed through multi-level reference chains.

### Processing steps

1. Traverse sentence → word → reference chains
2. Extract word-aligned EEG segments
3. Apply quality control:
   - discard signals shorter than 10 timesteps
   - remove malformed or corrupted entries
4. Standardize all samples to **250 timesteps (500 ms)**
   - pad shorter signals
   - crop longer signals

### Final dataset statistics

| Metric | Value |
|------|------|
| Total extracted words | 79,509 |
| Unique words | 2,865 |
| Final vocabulary | 500 most frequent |
| Final samples | 43,394 |
| Skipped samples | ~25–35% |
| Sample shape | (105 channels × 250 timesteps) |
| Disk size | 8.35 GB |

---

## Task Definition

**Input:** EEG segment (105 × 250)  
**Output:** One of 500 word classes  

This is a **500-way classification problem**, which is extremely challenging:
- Random Top-1 accuracy: **0.2%**
- Random Top-10 accuracy: **2%**

---

## Evaluation Metrics

We report:
- **Top-1 accuracy**
- **Top-5 accuracy**
- **Top-10 accuracy** (primary metric)

Top-10 accuracy is emphasized because:
- It matches BELT’s evaluation protocol
- It is more stable than Top-1
- It still represents only 2% of the class space

---

## Models

### 1. CNN-Only (Baseline)
- Learns spatial patterns across EEG channels
- No explicit temporal memory
- ~0.5M parameters

### 2. Conformer (CNN + Attention)
- Inspired by BELT
- Attention focuses on important time points
- ~1.2M parameters

### 3. CNN + GRU (Main Contribution)
- CNN extracts spatial features
- GRU models temporal evolution explicitly
- ~1.5M parameters

### 4. CNN + GRU + Attention
- Combines all components
- Highest-capacity model
- ~2.1M parameters

---

## Training Setup

| Setting | Value |
|------|------|
| Optimizer | Adam |
| Learning rate | 0.001 |
| Batch size | 32 |
| Epochs | 30 |
| Train / Validation split | 85% / 15% (stratified) |
| Total training time | ~6–7 hours |

---

## Results

### Validation Accuracy

| Model | Top-1 | Top-5 | Top-10 |
|------|------|------|------|
| CNN-Only | 6.60% | 18.35% | 27.59% |
| Conformer | 6.61% | 18.49% | 27.68% |
| CNN + GRU | **6.61%** | **18.51%** | **27.70%** |
| Full Model | 6.56% | 18.34% | 27.63% |
| BELT (reported) | – | – | **31.04%** |

---

## Key Findings

- CNNs capture most discriminative information
- GRU provides a **small but consistent improvement**
- Combining GRU and attention does not help further
- Differences between models are within statistical noise
- All models perform far above random guessing

A 0.11% difference in Top-10 accuracy corresponds to ~7 samples out of 6,510, which is **not statistically significant**.

---

## Why We Likely Didn’t Beat BELT

Possible reasons include:
- Different preprocessing or evaluation protocol
- BELT uses a full encoder–decoder with contrastive learning
- Implementation details (normalization, initialization, schedules)
- Encoder-only classification may have a lower performance ceiling

---

## Lessons Learned

- Word-level EEG decoding is extremely difficult
- Simple baselines are surprisingly strong
- Temporal modeling helps, but only marginally
- More data does not guarantee better performance
- Increased model complexity can hurt generalization

---

## Current Status

**Completed**
- Dataset extraction from all 18 subjects
- Robust preprocessing pipeline
- Four model architectures
- Training and evaluation framework
- Detailed analysis

**Needs improvement**
- Accuracy below BELT baseline
- Minimal architectural gains
- Need deeper investigation of evaluation differences

---

## Future Work

**Short-term**
- Subject-specific training
- Alternative GRU/LSTM configurations
- Improved normalization strategies
- Confusion matrix and error analysis

**Long-term**
- Full encoder–decoder architecture
- Contrastive learning
- Sentence-level decoding
- Multimodal EEG + eye-tracking
- Transformer-based approaches

---

## Final Remarks

Decoding language from EEG is both fascinating and frustratingly hard.  
While this work did not surpass the state of the art, it established a **reproducible pipeline**, clarified architectural limits, and provided valuable negative results.

> This is iteration one — not the final answer.

---

## Acknowledgments

- ZUCO dataset authors  
- BELT paper for baseline comparison  
- PyTorch community  
- Everyone who survived MATLAB `.mat` reference parsing  
