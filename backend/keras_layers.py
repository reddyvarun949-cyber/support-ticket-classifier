"""Custom Keras 3 layers for the small Transformer ticket classifier.

They live in their own file because BOTH the notebook (to build and train the model) and the Flask API (to load
models/ticket_transformer.keras) need exactly the same class definitions. Keras finds them when loading the model
because they are registered with @keras.saving.register_keras_serializable.

This file needs TensorFlow (pip install -r requirements-tf.txt). The API imports it ONLY when the Keras model file
exists, so the rest of the project works without TensorFlow.
"""
import keras
from keras import layers, ops


@keras.saving.register_keras_serializable(package="tickets")
class TokenAndPositionEmbedding(layers.Layer):
    """Token embedding + position embedding.

    Attention on its own does not know word ORDER ("not working" vs "working, not"), so we add a learned vector
    for every position 0..max_len-1 to the word vector. Token id 0 is padding: we return a mask so the layers
    after this one ignore padded positions.
    """

    def __init__(self, max_len, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=max_len, output_dim=embed_dim)

    def call(self, token_ids):
        positions = ops.arange(0, ops.shape(token_ids)[-1])          # 0, 1, 2, ... sequence length - 1
        return self.token_emb(token_ids) + self.pos_emb(positions)   # (batch, seq, embed) + (seq, embed)

    def compute_mask(self, token_ids, mask=None):
        return ops.not_equal(token_ids, 0)                           # True for real words, False for padding

    def get_config(self):
        config = super().get_config()
        config.update({"max_len": self.max_len, "vocab_size": self.vocab_size, "embed_dim": self.embed_dim})
        return config


@keras.saving.register_keras_serializable(package="tickets")
class TransformerBlock(layers.Layer):
    """One Transformer ENCODER block: multi-head self-attention + a small feed-forward network.

    Each part has a residual connection (x + f(x)) and layer normalisation, exactly like in BERT,
    only much smaller.
    """

    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads)
        self.ffn = keras.Sequential([layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim)])
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = layers.Dropout(dropout)
        self.drop2 = layers.Dropout(dropout)
        self.supports_masking = True                                 # pass the padding mask on to the next layer

    def call(self, x, mask=None, training=None):
        attention_mask = None
        if mask is not None:
            # (batch, seq) -> (batch, seq, seq): word i may look at word j only if BOTH are real words
            attention_mask = ops.logical_and(ops.expand_dims(mask, 1), ops.expand_dims(mask, 2))
        attended = self.att(query=x, value=x, key=x, attention_mask=attention_mask, training=training)
        x = self.norm1(x + self.drop1(attended, training=training))         # residual + norm
        x = self.norm2(x + self.drop2(self.ffn(x), training=training))      # residual + norm
        return x

    def get_config(self):
        config = super().get_config()
        config.update({"embed_dim": self.embed_dim, "num_heads": self.num_heads, "ff_dim": self.ff_dim,
                       "dropout": self.dropout})
        return config
