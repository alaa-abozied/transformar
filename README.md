# Self-Attention Encoder

Minimal NumPy implementation of a Transformer encoder's self-attention mechanism, built around the example sentence *"What are the symptoms of diabetes?"* from the RNN/LSTM & Transformer lecture (A. Prof. Noha El-Attar).

---

## What it computes

Given a sequence of tokens, the encoder:

1. Looks up a learned embedding for each token
2. Adds a sinusoidal positional encoding so position is represented
3. Projects embeddings into Q, K, V spaces via learned weight matrices
4. Computes scaled dot-product attention scores and normalises with softmax
5. Returns a weighted sum of V vectors as the contextual output

The core formula:

```
Attention(Q, K, V) = softmax(QKᵀ / √d_k) · V
```

---

## Project structure

```
self_attention.py
│
├── SelfAttention          # Scaled dot-product attention
│   ├── __init__           # Initialises W_q, W_k, W_v (d_model → d_k)
│   ├── forward(X)         # Returns (context_output, attention_weights)
│   └── _softmax(x)        # Numerically stable row-wise softmax
│
├── TransformerEncoder     # Wraps embedding + PE + attention
│   ├── __init__           # Builds random embedding matrix
│   ├── _positional_encoding(seq_len)   # Sinusoidal PE
│   └── encode(tokens)     # Full forward pass
│
└── print_attention_matrix # Pretty-prints the (seq_len × seq_len) weight matrix
```

---

## Requirements

- Python 3.10+
- NumPy

```bash
pip install numpy
```

---

## Usage

```bash
python self_attention.py
```

Sample output:

```
Sentence : "What are the symptoms of diabetes"
Tokens   : ['what', 'are', 'the', 'symptoms', 'of', 'diabetes']

Self-Attention Score Matrix (softmax weights):
                    what         are         the    symptoms          of    diabetes
------------------------------------------------------------------------------------
        what      0.1679      0.1648      0.1617      0.1666      0.1673      0.1717
         are      0.1701      0.1658      0.1606      0.1662      0.1652      0.1720
    symptoms      0.1680      0.1646      0.1619      0.1661      0.1679      0.1715
    diabetes      0.1695      0.1672      0.1630      0.1650      0.1655      0.1697
         ...

Encoder output shape : (6, 8)  (seq_len=6, d_k=8)
Row sums (must be 1) : [1. 1. 1. 1. 1. 1.]
```

To use with a different sentence or dimensions, edit the `__main__` block:

```python
sentence = "Glucose levels rise after a meal"
tokens = sentence.lower().split()

encoder = TransformerEncoder(d_model=32, d_k=16)
output, weights = encoder.encode(tokens)
```

---

## Reading the attention matrix

Each row `i` corresponds to token `i` attending over all tokens `j`. The value at `[i, j]` is how much token `i` draws from token `j`'s Value vector when building its contextual representation. Row sums are always 1.0.

With random weights the scores are near-uniform (~1/seq_len). In a trained model, semantically related pairs — e.g. `symptoms ↔ diabetes` — would have sharply higher weights, encoding that relationship directly in the output vectors.

---

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `d_model` | 16 | Token embedding dimension |
| `d_k` | 8 | Query/Key/Value projection dimension |
| `vocab_size` | 10 000 | Embedding matrix rows |

The `√d_k` scaling factor prevents the dot products from growing large in magnitude as `d_k` increases, which would push softmax into regions of near-zero gradients.

---

## Lecture context

This code corresponds to the encoder stage of the Transformer QA pipeline discussed in lecture:

```
Input question → Tokenise → Embed + PE → Self-Attention → Encoder output
                                                                   ↓
Medical passage ──────────────────────────── K, V → Enc-Dec Attention
                                                                   ↓
                                                    Decoder → Generated answer
```

The `SelfAttention` class here implements the top-left block of that diagram. Encoder-decoder (cross) attention follows the same formula — the decoder supplies Q while the encoder output supplies K and V.
