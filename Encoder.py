import tensorflow as tf

class Encoder():
    def __init__(self, vocab_size, embedding_dim, encoder_units):
        # print(vocab_size, embedding_dim, encoder_units, "##########################################ENCODER########################")
        self.encoder_units = encoder_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.encoder_units,
                                    return_sequences=True,
                                    return_state=True,
                                    recurrent_initializer='glorot_uniform')

    def __call__(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state = hidden)
        return output, state
