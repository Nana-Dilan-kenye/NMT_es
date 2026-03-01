# Neural Machine Translation: English → Spanish (NMT_es)

## Overview

This project implements a complete **Neural Machine Translation (NMT)** system that translates English sentences to Spanish using an **LSTM-based encoder-decoder architecture** with multiple attention mechanisms. The implementation is part of the MCF683 Neural Networks for NLP course challenge (Challenge 4).

## Objectives

- Implement a machine translation system translating **English → Spanish**
- Build a parallel corpus with 500+ curated sentence pairs
- Develop an encoder-decoder architecture with attention mechanisms
- Achieve **BLEU score > 25** on test set
- Visualize attention mechanisms and analyze translation quality
- Compare different attention mechanisms (Bahdanau, Luong, Multi-Head)
- Evaluate model performance and provide error analysis

## Architecture

The system uses the following pipeline:

```
English Sentence
       ↓
[BiLSTM Encoder] → Context Vectors (Hidden States)
       ↓
[Attention Mechanism] → Weighted Context
       ↓
[LSTM Decoder] → Spanish Translation
```

### Key Components

1. **Encoder (BiLSTM)**
   - Bidirectional LSTM processes English input in both directions
   - Generates context vectors for each input token
   - Captures both forward and backward dependencies

2. **Attention Mechanisms**
   - **Bahdanau Attention**: Additive attention with learnable alignment scores
   - **Luong Attention**: Multiplicative attention with global/local variants
   - **Multi-Head Attention**: Parallel attention heads for diverse context

3. **Decoder (LSTM)**
   - Processes attention outputs sequentially
   - Generates Spanish tokens one at a time
   - Uses teacher forcing during training

4. **Beam Search Decoding**
   - Produces multiple translation candidates during inference
   - Selects best translations based on log-probability scores

## Dataset

The project includes a **curated parallel corpus** of 500+ English-Spanish sentence pairs:

- **Sentence Length**: 5–15 words (tractable for LSTM training)
- **Vocabulary Size**: ~400 words per language
- **Coverage**: Everyday topics with diverse grammatical structures
- **Format**: Aligned translation pairs

This curated dataset allows fast training while focusing on architecture design. For production systems, larger corpora like WMT (millions of pairs) would be used.

## Files

- `Challenge_4_Neural_Machine_Translation.ipynb` - Main notebook containing:
  - Data preprocessing and tokenization
  - Model definitions (Encoder, Decoder, Attention mechanisms)
  - Training loop with validation
  - Inference and beam search implementation
  - Evaluation metrics (BLEU, attention visualization)
  - Error analysis and comparison of attention mechanisms

## Dependencies

```python
PyTorch         # Deep learning framework
NLTK            # BLEU score calculation
NumPy           # Numerical operations
Matplotlib      # Visualization
Seaborn         # Statistical plotting
```

Install dependencies via:
```bash
pip install torch nltk numpy matplotlib seaborn
```

## Usage

1. **Navigate to the notebook**:
   ```bash
   cd /path/to/NMT_es
   ```

2. **Open the notebook** in Jupyter or VS Code:
   ```bash
   jupyter notebook Challenge_4_Neural_Machine_Translation.ipynb
   ```

3. **Run cells sequentially**:
   - Cell 1: Setup and imports
   - Cell 2: Load parallel corpus
   - Cell 3: Tokenization and vocabulary building
   - Cell 4-5: Define encoder-decoder architecture
   - Cell 6-7: Training loop
   - Cell 8+: Evaluation, visualization, and analysis

## Key Features

### 1. Bidirectional Encoding
- BiLSTM captures context from both directions
- Improves translation quality by providing richer representations

### 2. Multiple Attention Mechanisms
- Compare Bahdanau vs Luong vs Multi-Head attention
- Visualize attention weights for interpretability
- Analyze which attention works best for English-Spanish pairs

### 3. Beam Search Decoding
- Generate multiple translation hypotheses
- Select highest-probability translations
- Improves translation quality over greedy decoding

### 4. BLEU Score Evaluation
- Standard metric for machine translation
- Corpus-level BLEU score on test set
- Sentence-level BLEU for detailed analysis

### 5. Attention Visualization
- Heatmaps showing alignment between English and Spanish tokens
- Helps debug model behavior
- Demonstrates attention focusing on relevant source words

## Results

Expected Results:
- **BLEU Score**: > 25 on test set
- **Training Time**: ~10-30 minutes on CPU, ~2-5 minutes on GPU
- **Attention Visualization**: Clear alignment patterns in heatmaps
- **Error Analysis**: Identifies common translation issues

## Model Evaluation

The notebook includes:

1. **Quantitative Evaluation**
   - BLEU scores (1-gram to 4-gram)
   - Corpus-level and sentence-level metrics
   - Comparison across attention mechanisms

2. **Qualitative Analysis**
   - Attention heatmaps for sample translations
   - Error cases and their causes
   - Common patterns in translation failures

3. **Visualizations**
   - Training/validation loss curves
   - BLEU score progression
   - Attention weight distributions

## Challenges & Solutions

### Challenge 1: Limited Training Data
- **Solution**: Curated diverse sentence pairs with grammatical variety
- **Mitigation**: Use data augmentation and regularization

### Challenge 2: Spanish Grammar Complexity
- **Difficulties**: Gender agreement, verb conjugation, flexible word order
- **Solution**: Multi-head attention helps capture diverse grammatical patterns

### Challenge 3: OOV (Out-of-Vocabulary) Words
- **Solution**: Implement character-level fallback or subword tokenization
- **Current**: Limited vocabulary; can be extended with BPE

### Challenge 4: Model Capacity
- **Balance**: LSTM size vs. overfitting risk
- **Solution**: Regularization, dropout, early stopping

## Hyperparameters

Key hyperparameters used:
- `hidden_dim` = 256 (LSTM hidden size)
- `embedding_dim` = 128 (Word embedding size)
- `num_layers` = 2 (Stacked LSTMs)
- `batch_size` = 32
- `learning_rate` = 0.001 (Adam optimizer)
- `num_epochs` = 50
- `dropout` = 0.3
- `beam_width` = 3

## Future Improvements

1. **Larger Datasets**: Use WMT data for production-quality translations
2. **Subword Tokenization**: Implement BPE or WordPiece for better OOV handling
3. **Transformer Architecture**: Replace LSTM with Transformer for improved parallelization
4. **Back-Translation**: Generate synthetic data by translating Spanish back to English
5. **Multi-Language Support**: Extend to multiple language pairs
6. **Fine-tuning**: Use pre-trained models like mBART or mT5
7. **Inference Optimization**: Quantization and pruning for deployment

## References

- Bahdanau et al., "Neural Machine Translation by Jointly Learning to Align and Translate" (2014)
- Luong et al., "Effective Approaches to Attention-based Neural Machine Translation" (2015)
- Vaswani et al., "Attention is All You Need" (2017)
- Papineni et al., "BLEU: a Method for Automatic Evaluation of Machine Translation" (2002)

## Author

Part of MCF683 (Neural Networks for NLP) coursework
Created: 2026

## License

Educational use. Course assignment for MCF683.
