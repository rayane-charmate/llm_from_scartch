import numpy as np
import tensorflow as tf
from model.scaled_dot_product import ScaledDotProductAttention


class MultiHeadAttention(tf.keras.layers.Layer):

    def __init__(self, dk=64, dv=64, h=8, d_model=512, casual_mask=None):

        super().__init__()
        self.dk = dk
        self.dv = dv
        self.h = h
        self.d_model = d_model
        self.causal_mask = casual_mask

        self.W_Q = [tf.keras.layers.Dense(units=self.dk, use_bias=False) for _ in range(self.h)]
        self.W_K = [tf.keras.layers.Dense(units=self.dk, use_bias=False) for _ in range(self.h)]
        self.W_V = [tf.keras.layers.Dense(units=self.dv, use_bias=False) for _ in range(self.h)]
        self.W_O = tf.keras.layers.Dense(units=self.d_model, use_bias=False)
        self.attention = ScaledDotProductAttention(dk=self.dk)

    def call(self, query, key=None, value=None):

        if key is None:
            key = query

        if value is None:
            value = key

        output = [
            self.attention((
                self.W_Q[i](query),
                self.W_K[i](key),
                self.W_V[i](value)), casual_mask=self.causal_mask) for i in range(self.h)]

        return self.W_O(tf.concat(output, axis=-1))
