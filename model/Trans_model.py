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