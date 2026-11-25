import tensorflow as tf


class VariableLR(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, d_model, warmup_steps=4000):

        super().__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = tf.cast(warmup_steps, tf.float32)

    def __call__(self, step):

        step = tf.maximum(tf.cast(step, tf.float32), 1.0)

        return tf.math.rsqrt(self.d_model) * tf.minimum(
            tf.math.rsqrt(step), step * tf.math.pow(self.warmup_steps, -1.5))
