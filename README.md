# Neural Machine Translation: English → Spanish

A machine translation system implementing encoder-decoder Seq2Seq architecture with comparative analysis of attention mechanisms for limited-data translation tasks.

## Overview

This project develops a complete neural machine translation (NMT) pipeline translating English to Spanish using an **LSTM-based encoder-decoder** architecture with integrated attention mechanisms. The implementation focuses on analyzing performance trade-offs of different attention strategies under limited-data conditions.

**Dataset**: ~621 synthetic English-Spanish sentence pairs covering diverse grammatical structures and everyday vocabulary.

## Key Accomplishments

- **Encoder-Decoder Seq2Seq Architecture**: Implemented bidirectional LSTM encoder with sequential LSTM decoder for sequence-to-sequence translation
- **Attention Mechanism Comparison**: Integrated and evaluated three attention variants:
  - **Bahdanau Attention**: Additive alignment with learnable scores
  - **Luong Attention**: Multiplicative attention with global/local variants
  - **Self-Attention**: Parallel multi-head attention for diverse context representation
- **Evaluation Framework**: BLEU score measurement (corpus and sentence-level) with statistical analysis
- **Performance Analysis**: Characterized trade-offs between attention complexity, translation quality, and computational cost under synthetic data constraints
- **Visualization**: Attention heatmaps demonstrating alignment patterns between source and target sequences

## Architecture

```
English Input
     ↓
[BiLSTM Encoder] → Hidden States & Context
     ↓
[Attention Layer] → Weighted Context Vectors
     ↓
[LSTM Decoder] → Spanish Output
```

### Components

1. **Bidirectional Encoder**
   - Processes input in both forward and backward directions
   - Captures context from entire source sequence
   - Outputs hidden state vectors for attention mechanism

2. **Attention Mechanisms**
   - **Bahdanau**: Learned alignment function (additive model)
   - **Luong**: Score-based attention (multiplicative model)
   - **Multi-Head**: Parallel attention heads with scaled dot-product

3. **LSTM Decoder**
   - Sequential generation with attention context
   - Teacher forcing during training
   - Beam search inference for hypothesis generation

## Dataset

- **Size**: ~621 English-Spanish parallel sentence pairs
- **Sentence Length**: 5–15 words (tractable for LSTM training)
- **Vocabulary**: ~400 tokens per language
- **Coverage**: Diverse grammatical structures with Spanish-specific features (gender, verb conjugation, word order variations)

Synthesized as controlled test case for attention mechanism comparison without external corpus dependencies.

## Technology Stack

- **Framework**: PyTorch
- **Architecture**: Seq2Seq with Encoder-Decoder
- **Attention**: Bahdanau, Luong, Multi-Head variants
- **Evaluation**: NLTK BLEU scoring
- **Visualization**: Matplotlib, Seaborn

## Usage

1. **Setup environment**:
   ```bash
   pip install torch nltk numpy matplotlib seaborn
   ```

2. **Run notebook**:
   ```bash
   jupyter notebook Challenge_4_Neural_Machine_Translation.ipynb
   ```

3. **Execution sequence**:
   - Data preparation and tokenization
   - Vocabulary construction from parallel corpus
   - Model definition (encoder, decoder, attention variants)
   - Training loop with validation
   - Inference with beam search
   - Evaluation and attention visualization

## Model Configuration

- **Embedding Dimension**: 128
- **Hidden Dimension**: 256
- **Encoder Layers**: 2 (bidirectional)
- **Decoder Layers**: 2
- **Batch Size**: 32
- **Learning Rate**: 0.001 (Adam optimizer)
- **Epochs**: 50
- **Dropout**: 0.3
- **Beam Width**: 3

## Results & Analysis

### Evaluation Metrics
- **Quantitative**: BLEU scores (unigram to 4-gram) on held-out test set
- **Qualitative**: Attention heatmaps and alignment visualization
- **Error Analysis**: Common translation failures and grammatical error patterns

### Key Findings
- Performance trade-offs between attention mechanisms examined under synthetic data constraints
- Attention visualizations reveal meaningful alignment patterns in source-target encoding
- Limited data conditions highlight importance of attention mechanism design choices
- Comparative analysis demonstrates feasibility of different approaches at reduced scale

## Project Structure

```
NMT_es/
├── README.md
└── Neural_Machine_Translation.ipynb (main implementation)
```

## Implementation Highlights

- **Robust Tokenization**: Character-level handling for Spanish inflections
- **Flexible Architecture**: Modular attention mechanism swapping for comparative analysis
- **Comprehensive Evaluation**: BLEU scoring with statistical significance testing
- **Visualization Framework**: Attention heatmaps for interpretable model analysis
- **Inference Optimization**: Beam search implementation for hypothesis ranking

## References

- Bahdanau et al., "Neural Machine Translation by Jointly Learning to Align and Translate" (2014)
- Luong et al., "Effective Approaches to Attention-based Neural Machine Translation" (2015)
- Vaswani et al., "Attention is All You Need" (2017)
- Papineni et al., "BLEU: a Method for Automatic Evaluation of Machine Translation" (2002)
