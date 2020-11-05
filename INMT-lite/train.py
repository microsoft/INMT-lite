"""
Code for training the models.
"""

from utils.Model_architectures import Encoder, Decoder
from utils.Model_types import Trans_model, Partial_model #TODO: Change it to better name

import numpy as np
import tensorflow as tf

import sys, getopt
import io
import json
import time
import os

def loss_function(real, pred, loss_object):
    mask = tf.math.logical_not(tf.math.equal(real, 0)) # To not account loss for padded words
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)

    return tf.reduce_mean(loss_)

@tf.function
def train_step(inp, targ, enc_hidden, optimizer, loss_object, batch_size):
    loss = 0

    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inp, enc_hidden)

        dec_hidden = enc_hidden

        dec_input = tf.expand_dims([0] * batch_size, 1)

        # Teacher forcing - feeding the target as the next input
        for t in range(1, targ.shape[1]):
            # passing enc_output to the decoder
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

            loss += loss_function(targ[:, t], predictions, loss_object)

            # using teacher forcing
            dec_input = tf.expand_dims(targ[:, t], 1)

    batch_loss = (loss / int(targ.shape[1]))

    variables = encoder.embedding.trainable_variables + encoder.gru.trainable_variables + decoder.embedding.trainable_variables + decoder.gru.trainable_variables + decoder.fc.trainable_variables + decoder.attention.W1.trainable_variables + decoder.attention.W2.trainable_variables + decoder.attention.V.trainable_variables

    gradients = tape.gradient(loss, variables)

    optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss

def predict_and_calculateLoss(inp, targ, enc_hidden, loss_object, batch_size):
    loss = 0
    enc_output, enc_hidden = encoder(inp, enc_hidden)
    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([0] * batch_size, 1)
    
    for t in range(1, targ.shape[1]):
        # passing enc_output to the decoder
        predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
        loss += loss_function(targ[:, t], predictions, loss_object)
        predictions = tf.argmax(predictions, 1)
        dec_input = tf.expand_dims(predictions, 1)

    batch_loss = (loss / int(targ.shape[1]))

    return batch_loss


def train(encoder, 
          decoder, 
          train_dataset,
          val_dataset,
          src_vocab_length,
          tgt_vocab_length,
          Tx, 
          Ty, 
          learning_rate, 
          steps_per_epoch_train,
          steps_per_epoch_val,
          train_batch_size, 
          val_batch_size,
          epochs,
          quantize_model, 
          checkpoints_steps,
          save_model_dir, 
          save_tflite_dir,
          log_steps):
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    
    trans_model = Trans_model.getModel(encoder, decoder, Tx, Ty, encoder.encoder_units)
    
    #Custom Training loop for training the model 
    for epoch in range(epochs):
        start = time.time()

        enc_hidden = tf.zeros((train_batch_size, encoder.encoder_units))
        total_loss = 0

        for (batch, (inp, targ)) in enumerate(train_dataset.take(steps_per_epoch_train)):
            batch_loss = train_step(inp, targ, enc_hidden, optimizer, loss_object, train_batch_size)
            total_loss += batch_loss

            if batch % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                            batch,
                                                            batch_loss.numpy()))
            
            
        #Validating on validation dataset after each epoch
        val_start = time.time()
        val_loss = 0
        enc_hidden = tf.zeros((val_batch_size, encoder.encoder_units))
        for(batch, (inp, targ)) in enumerate(val_dataset.take(steps_per_epoch_val)):
            val_loss += predict_and_calculateLoss(inp, targ, enc_hidden, loss_object, val_batch_size)
        
          # saving (checkpoint) the model 
        if (epoch + 1) % checkpoints_steps == 0:
            trans_model.save_weights(os.path.join(save_model_dir, "model_weight_epoch_" + str(epoch+1) + ".h5"))

        if (epoch+1)%log_steps == 0:
            print('Epoch {} Training Loss {:.4f}'.format(epoch + 1,
                                                total_loss / steps_per_epoch_train))
            print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
            print('Epoch {} Validation Loss {:.4f}'.format(epoch + 1,
                                                val_loss / steps_per_epoch_val))
            print('Time taken for validation {} sec\n'.format(time.time() - val_start))
            
    
    partial_model = Partial_model.getModel(encoder, decoder, Tx, encoder.encoder_units)
    serialise_and_save(partial_model, quantize_model, save_tflite_dir)

def serialise_and_save(partial_model, quantize, save_tflite_dir):
    converter = tf.lite.TFLiteConverter.from_keras_model(partial_model)
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    tflite_path = os.path.join(save_tflite_dir, 'tflite_model.tflite')
    print("Saving tflite model at:", tflite_path)
    
    with tf.io.gfile.GFile(tflite_path, 'wb') as f:
        f.write(tflite_model)
        
    print("Genrated tflite file at:", tflite_path)
    



def create_dataset_object(inp_tensor, tgt_tensor, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((inp_tensor, tgt_tensor)).shuffle(len(inp_tensor))
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset

def read_vector_file(file_path):
    vectors_str = io.open(file_path, encoding='utf-8').read().strip().split('\n')
    try:
        vector = [list(map(int, i.strip().split())) for i in vectors_str]
        return np.array(vector)
    except:
        raise RuntimeError("File is not preprocessed")

def read_vocab(filepath):
    with open(filepath, encoding='utf-8') as f:
        json_str = json.load(f)
    
    return json_str, json_str['num_words']

    


if __name__ == "__main__":
    
    # TODO: Implement subroutinr for training already saved model
    
    use_gpu = True
    save_model_dir = "./model" # save model in ./model if None Specified
    save_tflite_dir = "./tflite"
    quantize_model = False
    learning_rate = 0.01
    epochs = 10
    train_batch_size = 64
    val_batch_size = 64
    log_steps = 1
    src_word_vec_size = 256
    tgt_word_vec_size = 256
    reccurent_hidden = 500
    checkpoints_dir = "" # save checkpoints in ./model if None Specified
    seed = 42
    checkpoints_steps = 2
    to_path_tgt_train = ""
    to_path_src_train = ""
    to_path_src_val = ""
    to_path_tgt_val = ""
    to_path_src_vocab = ""
    to_path_tgt_vocab = ""

    #TODO: Maybe give an option to adjust Attention units ?
    
    sys_arg = sys.argv[1:]
    #Warning: Avoid using spaces while specifying directories
    try:
        opts, args = getopt.getopt(sys_arg, "", ['use_gpu=', 
                                                 'save_model_dir=', 
                                                 'save_tflite_dir=', 
                                                 'quantize_model=', 
                                                 'learning_rate=', 
                                                 'epochs=', 
                                                 'train_batch_size=', 
                                                 'val_batch_size=', 
                                                 'log_steps=', 
                                                 'src_word_vec_size=', 
                                                 'tgt_word_vec_size=',
                                                 'reccurent_hidden=',
                                                 'checkpoints_dir=', 
                                                 'seed=', 
                                                 'checkpoints_steps=',
                                                 'to_path_tgt_train=',
                                                 'to_path_src_train=',
                                                 'to_path_src_val=',
                                                 'to_path_tgt_val=',
                                                 'to_path_src_vocab=',
                                                 'to_path_tgt_vocab='])
    except Exception as e:
        print(e)
    
    
    for opt, arg in opts:
        print(opt, arg)
        if opt == '--use_gpu': use_gpu = False if arg.lower() == "false" else True
        elif opt == '--save_model_dir': save_model = arg
        elif opt == '--save_tflite': save_tflite = arg
        elif opt == '--quantize_model': quantize_model = False if arg.lower() == "false" else True
        elif opt == '--learning_rate': learning_rate = arg
        elif opt == '--epochs': epochs = int(arg)
        elif opt == '--train_batch_size': train_batch_size = int(arg)
        elif opt == '--val_batch_size': val_batch_size = int(arg)
        elif opt == '--log_steps': log_steps = int(arg)
        elif opt == '--src_word_vec_size': src_word_vec_size = int(arg)
        elif opt == '--tgt_word_vec_size': tgt_word_vec_size = int(arg)
        elif opt == '--reccurent_hidden': reccurent_hidden = int(arg)
        elif opt == '--checkpoints_dir': checkpoints_dir = arg
        elif opt == '--seed': seed = int(arg)
        elif opt == '--checkpoints_steps': checkpoints_steps = int(arg)
        elif opt == '--to_path_tgt_train': to_path_tgt_train = arg
        elif opt == '--to_path_src_train': to_path_src_train = arg
        elif opt == '--to_path_src_val': to_path_src_val = arg
        elif opt == '--to_path_tgt_val': to_path_tgt_val = arg
        elif opt == '--to_path_src_vocab': to_path_src_vocab = arg
        elif opt == '--to_path_tgt_vocab': to_path_tgt_vocab = arg
     
    if use_gpu == False:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        # pass #TODO: Disable GPU here
    
    if not(len(to_path_tgt_train)):
        raise AttributeError("Missing path to_path_tgt_train")
    if not(len(to_path_src_train)):
        raise AttributeError("Missing path to_path_src_train")
    if not(len(to_path_src_val)):
        raise AttributeError("Missing path to_path_src_val")
    if not(len(to_path_tgt_val)):
        raise AttributeError("Missing path to_path_tgt_val")
    if not(len(to_path_src_vocab)):
        raise AttributeError("Missing path to_path_src_vocab")
    if not(len(to_path_tgt_vocab)):
        raise AttributeError("Missing path to_path_tgt_vocab")
    
    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)
    if not os.path.exists(save_tflite_dir):
        os.makedirs(save_tflite_dir)
    
    
    vector_src_train = read_vector_file(to_path_src_train)
    vector_tgt_train = read_vector_file(to_path_tgt_train)
    vector_src_val = read_vector_file(to_path_src_val)
    vector_tgt_val = read_vector_file(to_path_tgt_val)
    
    src_vocab, src_vocab_length = read_vocab(to_path_src_vocab)
    tgt_vocab, tgt_vocab_length = read_vocab(to_path_tgt_vocab)
    
    encoder = Encoder(src_vocab_length, src_word_vec_size, reccurent_hidden)
    decoder = Decoder(tgt_vocab_length, tgt_word_vec_size, reccurent_hidden)
    
    steps_per_epoch_train = len(vector_src_train)//train_batch_size
    steps_per_epoch_val = len(vector_src_val)//val_batch_size
    
    train_dataset = create_dataset_object(vector_src_train, vector_tgt_train, train_batch_size)
    val_dataset = create_dataset_object(vector_src_val, vector_tgt_val, val_batch_size)
    
    Tx = vector_src_train.shape[1]
    Ty = vector_tgt_train.shape[1]
    
    # Saving model config
    model_config = {}
    model_config['Tx'] = Tx
    model_config['Ty'] = Ty
    model_config['reccurent_hidden'] = reccurent_hidden
    model_config['src_word_vec_size'] = src_word_vec_size
    model_config['tgt_word_vec_size'] = tgt_word_vec_size
    
    model_config_path = os.path.join(save_model_dir, 'model_config.json')
    with open(model_config_path, 'w', encoding='utf-8') as f:
        json.dump(model_config, f, ensure_ascii=False ) #TODO: add an argument for model config path
    
    
    train(encoder, 
          decoder, 
          train_dataset,
          val_dataset,
          src_vocab_length,
          tgt_vocab_length,
          Tx, 
          Ty, 
          learning_rate, 
          steps_per_epoch_train,
          steps_per_epoch_val,
          train_batch_size, 
          val_batch_size,
          epochs,
          quantize_model, 
          checkpoints_steps,
          save_model_dir, 
          save_tflite_dir,
          log_steps)