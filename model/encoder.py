import tensorflow as tf
from model.multi_head_attention import MultiHeadAttention
from model.positional_embedding import PositionalEmbedding


class EncoderLayer(tf.keras.layers.Layer):

    def __init__(self, d_model, dk, dv, h):

        super().__init__()
        self.d_model = d_model
        self.dk = dk
        self.dv = dv
        self.h = h

        self.attention = MultiHeadAttention(dk=self.dk, dv=self.dv, h=self.h, d_model=self.d_model)
        self.dense1 = tf.keras.layers.Dense(units=2048, activation="relu")
        self.dense2 = tf.keras.layers.Dense(units=self.d_model)
        self.norm1 = tf.keras.layers.LayerNormalization()
        self.norm2 = tf.keras.layers.LayerNormalization()

    def call(self, inputs):

        attention = self.attention(query=inputs)
        add = tf.keras.layers.Add()([inputs, attention])
        norm = self.norm1(add)

        x = self.dense1(norm)
        x = self.dense2(x)
        output = tf.keras.layers.Add()([x, norm])
        return self.norm2(output)


class Encoder(tf.keras.models.Model):

    def __init__(self, d_model, dk, dv, h, vocab_size, N):

        super().__init__()
        self.d_model = d_model
        self.h = h
        self.vocab_size = vocab_size
        self.dk = dk
        self.dv = dv
        self.N = N
        self.positional_embedding = PositionalEmbedding(d_model)
        self.encoder_layers = [
            EncoderLayer(d_model=self.d_model, dk=self.dk, dv=self.dv, h=self.h)
            for _ in range(self.N)
        ]

        self.embedding = tf.keras.layers.Embedding(self.vocab_size, self.d_model)

    def call(self, inputs):

        input_embedding = self.embedding(inputs) #* tf.math.sqrt(self.d_model)
        x = input_embedding + tf.expand_dims(self.positional_embedding(input_embedding), axis=0)

        for layer in self.encoder_layers:

            x = layer(x)

        return x
