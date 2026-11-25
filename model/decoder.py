import tensorflow as tf
from model.multi_head_attention import MultiHeadAttention
from model.positional_embedding import PositionalEmbedding


class DecoderLayer(tf.keras.layers.Layer):

    def __init__(self, d_model, dk, dv, h, mask=None):

        super().__init__()

        self.d_model = d_model
        self.dk = dk
        self.dv = dv
        self.h = h
        self.mask = mask

        # Defining layers
        self.attention = MultiHeadAttention(d_model=self.d_model, h=self.h, dk=self.dk, dv=self.dv)
        self.masked_attention = MultiHeadAttention(d_model=self.d_model, dk=self.dk, dv=self.dv, h=self.h, casual_mask=self.mask)
        self.norm1 = tf.keras.layers.LayerNormalization()
        self.norm2 = tf.keras.layers.LayerNormalization()
        self.norm3 = tf.keras.layers.LayerNormalization()

        self.dense1 = tf.keras.layers.Dense(units=2048, activation="relu")
        self.dense2 = tf.keras.layers.Dense(units=self.d_model)

    def call(self, inputs_embedding, encoder_output):

        masked_attention = self.masked_attention(query=inputs_embedding)
        add1 = tf.keras.layers.Add()([masked_attention, inputs_embedding])
        norm1 = self.norm1(add1)

        attention = self.attention(query=norm1, key=encoder_output, value=encoder_output)
        add2 = tf.keras.layers.Add()([attention, norm1])
        norm2 = self.norm2(add2)

        dense1 = self.dense1(norm2)
        dense2 = self.dense2(dense1)

        add3 = tf.keras.layers.Add()([dense2, norm2])
        return self.norm3(add3)


class Decoder(tf.keras.models.Model):

    def __init__(self, d_model, N, dk, dv, h, vocab_size, mask=None):
        
        super().__init__()

        self.d_model = d_model
        self.N = N
        self.dk = dk
        self.dv = dv
        self.h = h
        self.vocab_size = vocab_size
        self.mask = mask

        # Building layers

        self.dense1 = tf.keras.layers.Dense(units=self.vocab_size)
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=d_model)
        self.positional_embedding = PositionalEmbedding(d_model=d_model)
        self.decoder_layers = [
            DecoderLayer(d_model=self.d_model, dk=self.dk, dv=self.dv, h=self.h, mask=self.mask)
            for _ in range(self.N)
        ]

    def call(self, inputs, encoder_output):

        input_embedding = self.embedding(inputs) # * tf.math.sqrt(self.d_model)
        x = input_embedding + tf.expand_dims(self.positional_embedding(input_embedding), axis=0)

        for layer in self.decoder_layers:
            x = layer(inputs_embedding=x, encoder_output=encoder_output)

        x = self.dense1(x)
        return tf.nn.softmax(x, axis=-1)

