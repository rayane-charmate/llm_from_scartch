import tensorflow as tf
from model.encoder import Encoder
from model.decoder import Decoder
from model.casual_mask import CasualMask


class Transformer(tf.keras.models.Model):

    def __init__(self, vocab_size, dk, dv, d_model, h, N):

        super().__init__()
        self.vocab_size = vocab_size
        self.dk = dk
        self.dv = dv
        self.d_model = d_model
        self.h = h
        self.N = N
        self.casual_mask = CasualMask()

        self.encoder = Encoder(
            d_model=self.d_model,
            N=self.N,
            vocab_size=self.vocab_size,
            h=self.h,
            dk=self.dk,
            dv=self.dv)

        self.decoder = Decoder(
            d_model=self.d_model,
            N=self.N,
            dk=self.dk,
            dv=self.dv,
            h=self.h,
            vocab_size=self.vocab_size,
            mask=self.casual_mask)

    def call(self, inputs):

        encoder_input, decoder_input = inputs
        encoder_output = self.encoder(encoder_input)
        decoder_output = self.decoder(decoder_input, encoder_output)

        return decoder_output
