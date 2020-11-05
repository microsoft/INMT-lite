"""
Code for building specific models for translation and partial mode.
"""

import tensorflow as tf

class Trans_model():
    def getModel(encoder, decoder, Tx, Ty, units):
        outputs = []
        X = tf.keras.layers.Input(shape=(Tx,))
        enc_hidden = tf.keras.layers.Input(shape=(units,))
        dec_input = tf.keras.layers.Input(shape=(1,))
        
        d_i = dec_input
        e_h = enc_hidden
        X_i = X
        
        enc_output, e_h = encoder(X, enc_hidden)
        
        
        dec_hidden = e_h
    #     dec_input = tf.expand_dims([tgt_tokenizer.word_index['<start>']] * BATCH_SIZE, 1)
    #     predictions = dec_input
        for t in range(1, Ty):
        # passing enc_output to the decoder
            predictions, dec_hidden, _ = decoder(d_i, dec_hidden, enc_output)
            outputs.append(predictions)
            d_i = tf.reshape(tf.math.argmax(predictions, axis = 1), (-1, 1))
    #     outputs.append(predictions)
        
        return tf.keras.Model(inputs = [X, enc_hidden, dec_input], outputs = outputs)

class Partial_model():
    def getModel(encoder, decoder, Tx, units, tgt_vocab_length):
        outputs = []
        X = tf.keras.layers.Input(shape=(Tx,))
        partial = tf.keras.layers.Input(shape=(Tx,))
        enc_hidden = tf.keras.layers.Input(shape=(units,))
        dec_input = tf.keras.layers.Input(shape=(1,))
        mask = tf.keras.layers.Input(shape=(tgt_vocab_length,))
        
        d_i = dec_input
        X_i = X
        
        enc_output, e_h = encoder(X, enc_hidden)
        
        
        dec_hidden = e_h
        print(dec_input.shape, 'inp', dec_hidden.shape, 'dec_hidd')
        for t in range(1, Tx):
            print(t, 'tt')
        # passing enc_output to the decoder
            predictions, dec_hidden, _ = decoder(d_i, dec_hidden, enc_output)
    #         outputs.append(predictions)
            print(predictions.shape, 'pred')
            d_i = tf.reshape(partial[:, t], (-1, 1))
            print(dec_input.shape, 'dec_input')
        
        predictions, dec_hidden, _ = decoder(d_i, dec_hidden, enc_output)
        predictions = tf.multiply(predictions, mask)
        d_i = tf.squeeze(d_i)
        
        outputs.append(tf.math.top_k(predictions, 5))
        
        return tf.keras.Model(inputs = [X, enc_hidden, dec_input, partial, mask], outputs = [outputs[0][0], outputs[0][1]])