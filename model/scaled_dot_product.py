import tensorflow as tf


class MatMul(tf.keras.layers.Layer):

    def __init__(self, transpose=False):

        super().__init__()
        self.transpose = transpose

    def call(self, inputs):

        A, B = inputs
        return tf.matmul(A, B, transpose_b=self.transpose)


class Scale(tf.keras.layers.Layer):

    def __init__(self, dk):

        super().__init__()
        self.dk = dk

    def call(self, inputs):

        return inputs / tf.math.sqrt(tf.cast(self.dk, inputs.dtype))


class ScaledDotProductAttention(tf.keras.layers.Layer):

    def __init__(self, dk=64):

        super().__init__()
        self.dk = dk

    def call(self, inputs, casual_mask=None):

        Q, K, V = inputs
        input_query_key = tf.matmul(Q, K, transpose_b=True)
        scale_layer = input_query_key / tf.math.sqrt(tf.cast(self.dk, input_query_key.dtype))

        if casual_mask != None:

            scale_layer = scale_layer + tf.broadcast_to(casual_mask(scale_layer), tf.shape(scale_layer))

        softmax = tf.nn.softmax(scale_layer, axis=-1)
        return tf.matmul(softmax, V)
