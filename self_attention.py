import numpy as np


class SelfAttention:
    def __init__(self, d_model: int, d_k: int):
        self.d_k = d_k
        rng = np.random.default_rng(42)
        self.W_q = rng.normal(0, 0.1, (d_model, d_k))
        self.W_k = rng.normal(0, 0.1, (d_model, d_k))
        self.W_v = rng.normal(0, 0.1, (d_model, d_k))

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        Q = X @ self.W_q
        K = X @ self.W_k
        V = X @ self.W_v
        raw_scores = Q @ K.T / np.sqrt(self.d_k)
        weights = self._softmax(raw_scores)
        return weights @ V, weights

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max(axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)


class TransformerEncoder:
    def __init__(self, vocab_size: int = 10_000, d_model: int = 16, d_k: int = 8):
        self.d_model = d_model
        rng = np.random.default_rng(0)
        self.embedding_matrix = rng.normal(0, 0.1, (vocab_size, d_model))
        self.attention = SelfAttention(d_model, d_k)

    def _positional_encoding(self, seq_len: int) -> np.ndarray:
        positions = np.arange(seq_len)[:, None]
        dims = np.arange(0, self.d_model, 2)[None, :]
        angles = positions / (10_000 ** (dims / self.d_model))
        PE = np.zeros((seq_len, self.d_model))
        PE[:, 0::2] = np.sin(angles)
        PE[:, 1::2] = np.cos(angles[:, : PE[:, 1::2].shape[1]])
        return PE

    def encode(self, tokens: list[str]) -> tuple[np.ndarray, np.ndarray]:
        indices = [hash(t) % self.embedding_matrix.shape[0] for t in tokens]
        X = self.embedding_matrix[indices] + self._positional_encoding(len(tokens))
        return self.attention.forward(X)


def print_attention_matrix(weights: np.ndarray, tokens: list[str]) -> None:
    col_w = 12
    header = f"{'':>{col_w}}" + "".join(f"{t:>{col_w}}" for t in tokens)
    print(header)
    print("-" * col_w * (len(tokens) + 1))
    for i, token in enumerate(tokens):
        row = f"{token:>{col_w}}" + "".join(f"{weights[i, j]:>{col_w}.4f}" for j in range(len(tokens)))
        print(row)


if __name__ == "__main__":
    sentence = "What are the symptoms of diabetes"
    tokens = sentence.lower().split()

    encoder = TransformerEncoder(d_model=16, d_k=8)
    output, attention_weights = encoder.encode(tokens)

    print(f'Sentence : "{sentence}"')
    print(f"Tokens   : {tokens}\n")
    print("Self-Attention Score Matrix (softmax weights):")
    print_attention_matrix(attention_weights, tokens)
    print(f"\nEncoder output shape : {output.shape}  (seq_len={len(tokens)}, d_k=8)")
    print(f"Row sums (must be 1) : {attention_weights.sum(axis=1).round(6)}")
