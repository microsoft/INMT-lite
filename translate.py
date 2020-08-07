from model.Trans_model import Trans_model
from Encoder import Encoder
from Decoder import Decoder
from preprocess import preprocess_sentence, tokenize

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

import os
import sys
import getopt
import json
import io


def test(model, model_config, src_vocab, tgt_vocab, src_word_ind, tgt_word_ind, src_tensor, tgt_tensor):
    enc_hidden = tf.zeros((len(src_tensor), model_config['reccurent_hidden']))
    dec_input =  tf.expand_dims([int(tgt_word_ind['<start>'])] * len(src_tensor), 1)
    print("predicting...")
    preds = model.predict([src_tensor, enc_hidden, dec_input])
    print("predicted")
    preds = np.array([preds[i].argmax(axis = 1) for i in range(len(preds))])
    preds = preds.swapaxes(0,1)
    
    score = 0
    for i in range(len(tgt_tensor)):
        arr1 = []
        flag = False
        for j in tgt_tensor[i]:
            if tgt_vocab[str(j)] == '<start>':
                flag = True
            elif flag:
                arr1.append(j)
            
            if tgt_vocab[str(j)] == '<end>':
                flag = False
        
        arr2 = []
        for j in preds[i]:
            arr2.append(j)
            if tgt_vocab[str(j)] == '<end>':
                break
        
        temp = abs(len(arr1)-len(arr2))
        for j in range(min(len(arr1), len(arr2))):
            if arr1[j] == arr2[j]:
                temp += 1    
        score += temp
    
    test_acc = score/len(src_tensor)
    print("The Test accuracy on the given dataset: ", test_acc)

def inference(model, model_config, src_vocab, tgt_vocab, src_word_ind, tgt_word_ind, src_tensor, tgt_path):
    enc_hidden = tf.zeros((len(src_tensor), model_config['reccurent_hidden']))
    dec_input =  tf.expand_dims([int(tgt_word_ind['<start>'])] * len(src_tensor), 1)
    
    preds = model.predict([src_tensor, enc_hidden, dec_input])
    preds = np.array([preds[i].argmax(axis = 1) for i in range(len(preds))])
    preds = preds.swapaxes(0,1)
    file_path = os.path.join(tgt_path, 'inference.txt')
    
    with open(file_path, 'w', encoding='utf-8') as f:
        for i in preds:
            f.write(','.join([tgt_vocab[str(j)] for j in i]) + '\n')
    
    print("Results successfully saved at: ", file_path)
    

def inline(model, model_config, src_vocab, tgt_vocab, src_word_ind, tgt_word_ind, sentence):
    sentence = preprocess_sentence(sentence)
    src_tensor = [[int(src_word_ind[i]) if i in src_word_ind else 0 for i in sentence.split(' ')] + [0]*(8-len(sentence.split(' ')))]
    
    src_tensor = np.array(src_tensor)
    
    enc_hidden = tf.zeros((len(src_tensor), model_config['reccurent_hidden']))
    dec_input =  tf.expand_dims([int(tgt_word_ind['<start>'])] * len(src_tensor), 1)
    
    preds = model.predict([src_tensor, enc_hidden, dec_input])
    preds = np.array([preds[i].argmax(axis = 1) for i in range(len(preds))])
    preds = preds.swapaxes(0,1)
    
    print("Predicted Sentence is: ", *[tgt_vocab[str(i)] for i in preds[0]]) # This will not contain the first start token
    
    
            

def load_data(path, src_vocab, src_vocab_len, sen_len, padding = 'post'):
    
    
    word_ind = {}
    for i in src_vocab:
        if i != 'num_words' and int(i) < src_vocab_len:
            word_ind[src_vocab[i]] = i
    
    if path:
    
        lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
        cleaned = [preprocess_sentence(i) for i in lines]
        
        tokenizer = Tokenizer(oov_token='<OOV>')
        tokenizer.word_index = word_ind
        tensor = tokenizer.texts_to_sequences(cleaned)
        padded_tensor = pad_sequences(tensor, padding=padding, truncating='post', maxlen=sen_len)
    
        return padded_tensor, word_ind

    return None, word_ind
    
    
    
    
def load_paths(model_path,
                src_path, 
                tgt_path, 
                src_vocab_path,
                tgt_vocab_path,
                model_config_path
                ):
    src_vocab = None
    tgt_vocab = None
    model_config = None
    with open(src_vocab_path, encoding='utf-8') as f:
        src_vocab = json.load(f)
    with open(tgt_vocab_path, encoding='utf-8') as f:
        tgt_vocab = json.load(f)
    with open(model_config_path, encoding='utf-8') as f:
        model_config = json.load(f)
    
    src_vocab['0'] = '<pad>'
    tgt_vocab['0'] = '<pad>'
        
    src_vocab_len = src_vocab['num_words']
    tgt_vocab_len = tgt_vocab['num_words']
    Tx = model_config['Tx']
    Ty = model_config['Ty']
    reccurent_hidden = model_config['reccurent_hidden']
    src_word_vec_size = model_config['src_word_vec_size']
    tgt_word_vec_size = model_config['tgt_word_vec_size']
    
    src_tensor = None
    tgt_tensor = None
    src_word_ind = None
    tgt_word_ind = None
    
    
    src_tensor, src_word_ind = load_data(src_path, src_vocab, src_vocab_len, Tx)
    tgt_tensor, tgt_word_ind = load_data(tgt_path, tgt_vocab, tgt_vocab_len, Ty)
    
    encoder = Encoder(src_vocab_len, src_word_vec_size, reccurent_hidden)
    decoder = Decoder(tgt_vocab_len, tgt_word_vec_size, reccurent_hidden)
    
    trans_model = Trans_model.getModel(encoder, decoder, Tx, Ty, encoder.encoder_units)
    trans_model.load_weights(model_path)
    
    to_ret = [trans_model, src_vocab, tgt_vocab, src_word_ind, tgt_word_ind, model_config]
    if src_tensor is not None:
        to_ret.append(src_tensor)
    if tgt_tensor is not None:
        to_ret.append(tgt_tensor)
    
    return to_ret




if __name__ == "__main__":
    model_path = ''
    batch_size = 64
    src_path = ''
    tgt_path = ''
    sentence = ''
    mode = ''
    src_vocab_path = ''
    tgt_vocab_path = ''
    model_config_path = ''
    use_gpu = False
    
    sys_arg = sys.argv[1:]
    
    try:
        opts, args = getopt.getopt(sys_arg, "", ['model_path=', 
                                                 'batch_size=', 
                                                 'src_path=', 
                                                 'tgt_path=', 
                                                 'sentence=',
                                                 'mode=',
                                                 'src_vocab_path=',
                                                 'tgt_vocab_path=',
                                                 'use_gpu=',
                                                 'model_config_path=',
                                                 ])
    except Exception as e:
        print(e)
    
    for opt, arg in opts:
        print(opt, arg)
        if opt == '--model_path': model_path = arg
        elif opt == '--batch_size': batch_size = int(arg)
        elif opt == '--src_path': src_path = arg
        elif opt == '--tgt_path': tgt_path = arg
        elif opt == '--sentence': sentence = arg
        elif opt == '--mode': mode = arg
        elif opt == '--src_vocab_path': src_vocab_path = arg
        elif opt == '--tgt_vocab_path': tgt_vocab_path = arg
        elif opt == '--use_gpu': use_gpu = False if arg.lower() == "false" else True
        elif opt == '--model_config_path': model_config_path = arg
    
    if use_gpu == False:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    
    if not(len(model_config_path)): raise AttributeError("Please provide path for model config file")
    if not(len(src_vocab_path)): raise AttributeError("Provide path for source language vocabulary")
    if not(len(tgt_vocab_path)): raise AttributeError("Provide path for target language vocabulary")
    if not(len(model_path)): raise AttributeError("Provide path for trained model")
    
    
    
    
        
    
    if mode.lower() == 'test':
        if not(len(src_path)): raise AttributeError("Provide path for source language data")
        if not(len(tgt_path)): raise AttributeError("Provide path for target language data")
        model, src_vocab, tgt_vocab, src_word_ind, tgt_word_ind, model_config, src_tensor, tgt_tensor = load_paths(model_path,
                                                                                                        src_path, 
                                                                                                        tgt_path, 
                                                                                                        src_vocab_path,
                                                                                                        tgt_vocab_path,
                                                                                                        model_config_path
                                                                                                        )
        test(model, model_config, src_vocab, tgt_vocab, src_word_ind, tgt_word_ind, src_tensor, tgt_tensor)
    elif mode.lower() == 'inference':
        if not(len(src_path)): raise AttributeError("Provide path for source language data")
        model, src_vocab, tgt_vocab, src_word_ind, tgt_word_ind, model_config, src_tensor = load_paths(model_path,
                                                                                            src_path, 
                                                                                            '', 
                                                                                            src_vocab_path,
                                                                                            tgt_vocab_path,
                                                                                            model_config_path
                                                                                            )
        # print(tgt_word_ind)
        inference(model, model_config, src_vocab, tgt_vocab, src_word_ind, tgt_word_ind, src_tensor, tgt_path)
    elif mode.lower() == 'inline':
        if not(sentence): raise AttributeError("Provide input sentence")
        model, src_vocab, tgt_vocab, src_word_ind, tgt_word_ind, model_config = load_paths(model_path,
                                                                                src_path, 
                                                                                tgt_path, 
                                                                                src_vocab_path,
                                                                                tgt_vocab_path,
                                                                                model_config_path
                                                                                )
        inline(model, model_config, src_vocab, tgt_vocab, src_word_ind, tgt_word_ind, sentence)
    else:
        raise AttributeError("Invalid argument for mode. Allowed arguments for mode are: test, inference and inline")
        
    
    
