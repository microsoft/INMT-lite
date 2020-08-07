import tensorflow as tf
class Partial_model():
    def getModel(encoder, decoder, Tx, units):
        outputs = []
        X = tf.keras.layers.Input(shape=(Tx,))
        partial = tf.keras.layers.Input(shape=(Tx,))
        enc_hidden = tf.keras.layers.Input(shape=(units,))
        dec_input = tf.keras.layers.Input(shape=(1,))
        
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
        d_i = tf.squeeze(d_i)
        
        outputs.append(tf.math.top_k(predictions, 5))
        
        return tf.keras.Model(inputs = [X, enc_hidden, dec_input, partial], outputs = [outputs[0][0], outputs[0][1]])
            