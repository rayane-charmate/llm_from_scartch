import tensorflow as tf


class CasualMask(tf.keras.layers.Layer):

    def __init__(self):

        super().__init__()

    def call(self, inputs):

        seq_len = tf.shape(inputs)[-1]

        mask = tf.linalg.band_part(tf.ones([seq_len, seq_len]), 0, -1)
        mask = mask - tf.eye(seq_len)
        mask = tf.cast(mask, inputs.dtype)
        return mask * tf.cast(-1e9, inputs.dtype)