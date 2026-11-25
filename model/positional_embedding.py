import tensorflow as tf


class PositionalEmbedding(tf.keras.layers.Layer):

    def __init__(self, d_model=512):

        super().__init__()
        self.d_model = d_model

    def call(self, inputs):

        seq_len = tf.shape(inputs)[1]
        positions = tf.cast(
            tf.range(seq_len)[:, tf.newaxis], tf.float32)

        dimension = tf.cast(
            tf.range(self.d_model)[tf.newaxis, :], tf.float32)

        angle = positions / (tf.pow(tf.constant(10000, tf.float32), 2 * (dimension // 2) / self.d_model))

        return tf.where(
            dimension % 2 == 0,
            tf.sin(angle),
            tf.cos(angle))