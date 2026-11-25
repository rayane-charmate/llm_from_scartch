import tensorflow as tf
from transformers import AutoTokenizer

class ProcessingPipeline:

    def __init__(self, dataset):

        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer.add_special_tokens({"pad_token": "<|PAD|>"})
        self.TARGET_LENGTH = 128
        self.SOURCE_LENGTH = 128
        self.dataset = dataset

    def _preprocess_data(self, text):

        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        tokens.append(self.tokenizer.eos_token_id)

        assert len(tokens) > 2

        if len(tokens) <= self.TARGET_LENGTH + self.SOURCE_LENGTH:

            split_index = len(tokens) // 2
            encoder_input = tokens[:split_index]
            decoder_output = tokens[split_index:]
            decoder_input = [self.tokenizer.bos_token_id] + tokens[split_index:-1]

            if len(encoder_input) == 0 or len(decoder_output) == 0:
                return

            yield (encoder_input, decoder_input, decoder_output)

        else:

            for i in range(0, len(tokens), self.TARGET_LENGTH):

                encoder_input = tokens[i:i + self.TARGET_LENGTH]
                decoder_output = tokens[i + self.TARGET_LENGTH:i + self.TARGET_LENGTH + self.SOURCE_LENGTH]

                if len(encoder_input) == 0 or len(decoder_output) == 0:
                    continue

                decoder_input = [self.tokenizer.bos_token_id] + decoder_output[:-1]

                yield (encoder_input, decoder_input, decoder_output)

    def _generate_dataset(self):

        for example in self.dataset:

            for enc_in, dec_in, dec_out in self._preprocess_data(example["text"]):
                yield ((enc_in, dec_in), dec_out)

    def _get_length_sequence(self, inputs, output):

        enc_in, dec_in = inputs

        return tf.maximum(
            tf.shape(enc_in)[0],
            tf.shape(dec_in)[0])

    def process_dataset(self):

        output_signature = (
            (
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32)
            ),
            tf.TensorSpec(shape=(None,), dtype=tf.int32))

        train_dataset = tf.data.Dataset.from_generator(self._generate_dataset, output_signature=output_signature)
        train_dataset = train_dataset.shuffle(buffer_size=10000)

        train_dataset = train_dataset.bucket_by_sequence_length(
            element_length_func=self._get_length_sequence,
            bucket_boundaries=[64, 128],
            bucket_batch_sizes=[64, 32, 16],
            padded_shapes=(([None], [None]), [None]),
            padding_values=((self.tokenizer.pad_token_id, self.tokenizer.pad_token_id), self.tokenizer.pad_token_id),
            drop_remainder=False
        )

        return train_dataset.prefetch(tf.data.AUTOTUNE)
