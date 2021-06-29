"""
Code for performing inference/testing on the trained models.
"""

from utils.Model_architectures import Encoder, Decoder
from utils.Model_types import Trans_model, Partial_model #TODO: Change it to better name
from preprocess import preprocess_sentence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import TemplateProcessing

import tensorflow as tf
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
    print("%%%%%%%%", src_tensor[:5])
    print("%%%%%%%% PRED", preds[:5])
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
        print("$$$$$$$$\n\n ARR1", arr1)
        print("$$$$$$$$\n\n ARR2", arr2)
        temp = abs(len(arr1)-len(arr2))
        for j in range(min(len(arr1), len(arr2))):
            if arr1[j] == arr2[j]:
                temp += 1    
        score += temp/len(arr1)
    
    test_acc = score/len(src_tensor)
    print("The Test accuracy on the given dataset: ", test_acc)

def inference(model, model_config, src_vocab, tgt_vocab, src_word_ind, tgt_word_ind, src_tensor, tgt_path):
    enc_hidden = tf.zeros((len(src_tensor), model_config['reccurent_hidden']))
    dec_input =  tf.expand_dims([int(tgt_word_ind['<start>'])] * len(src_tensor), 1)
    
    preds = model.predict([src_tensor, enc_hidden, dec_input])
    preds = np.array([preds[i].argmax(axis = 1) for i in range(len(preds))])
    preds = preds.swapaxes(0,1)
    file_path = os.path.join('data', 'inference.txt')
    
    # load BBPE tokenizer
    lang_tokenizer = ByteLevelBPETokenizer(
        './data/tgt_tokenizer_bpe-vocab.json',
        './data/tgt_tokenizer_bpe-merges.txt',
    )

    with open(file_path, 'w', encoding='utf-8') as f:
        for i in preds:
            f.write(lang_tokenizer.decode(i) + '\n')
    
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
    
    
            

def load_data(path, vocab_path, merges_path, vocab_len, sent_len, padding = 'post'):

    if path:

        # load validation data
        lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
        processed_text = [preprocess_sentence(i) for i in lines]

        # load BBPE tokenizer
        lang_tokenizer = ByteLevelBPETokenizer(
            vocab_path,
            merges_path,
        )

        # post tokenization processing
        lang_tokenizer.post_processor = TemplateProcessing(
            single="<start> $A <end>",
            special_tokens=[
                ("<start>", lang_tokenizer.token_to_id("<start>")),
                ("<end>", lang_tokenizer.token_to_id("<end>")),
            ],
        )

        # encoding the input into tokens using BBPE
        tensor = lang_tokenizer.encode_batch(processed_text)
        tensor = list(map(lambda x: x.ids, tensor))

        # padd the input sentence
        tensor = pad_sequences(tensor, padding=padding, truncating='post', maxlen=sent_len)

        return tensor
    
    
    
    
def load_paths(model_path,
                src_path, 
                tgt_path, 
                src_vocab_path,
                tgt_vocab_path,
                src_merges_path, 
                tgt_merges_path,
                model_config_path
                ):
    src_vocab = None
    tgt_vocab = None
    model_config = None
    with open(src_vocab_path, encoding='utf-8') as f:
        src_word_ind = json.load(f)
        src_word_ind = {k:str(v) for k,v in src_word_ind.items()}
 
        src_vocab = {str(v):k for k,v in src_word_ind.items()}
    with open(tgt_vocab_path, encoding='utf-8') as f:
        tgt_word_ind = json.load(f)
        tgt_word_ind = {k:str(v) for k,v in tgt_word_ind.items()}

        tgt_vocab = {str(v):k for k,v in tgt_word_ind.items()}
    with open(model_config_path, encoding='utf-8') as f:
        model_config = json.load(f)
    
    src_vocab['0'] = '<pad>'
    tgt_vocab['0'] = '<pad>'
    # print("$$$$$$$$$$$\n\n TGT VOCAB", tgt_vocab)    
    # print("$$$$$$$$$$$\n\n TGT WRD IDX", tgt_word_ind)    
    src_vocab_len = len(src_vocab)
    tgt_vocab_len = len(tgt_vocab)
    Tx = model_config['Tx']
    Ty = model_config['Ty']
    reccurent_hidden = model_config['reccurent_hidden']
    src_word_vec_size = model_config['src_word_vec_size']
    tgt_word_vec_size = model_config['tgt_word_vec_size']
    
    src_tensor = None
    tgt_tensor = None
    
    
    src_tensor = load_data(src_path, src_vocab_path, src_merges_path, src_vocab_len, Tx)
    tgt_tensor = load_data(tgt_path, tgt_vocab_path, tgt_merges_path, tgt_vocab_len, Ty)
    
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
    src_merges_path = ''
    tgt_merges_path = ''
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
                                                 'src_merges_path=',
                                                 'tgt_merges_path=',
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
        elif opt == '--src_merges_path': src_merges_path = arg
        elif opt == '--tgt_merges_path': tgt_merges_path = arg
        elif opt == '--use_gpu': use_gpu = False if arg.lower() == "false" else True
        elif opt == '--model_config_path': model_config_path = arg
    
    if use_gpu == False:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    
    if not(len(model_config_path)): raise AttributeError("Please provide path for model config file")
    if not(len(src_vocab_path)): raise AttributeError("Provide path for source language vocabulary")
    if not(len(tgt_vocab_path)): raise AttributeError("Provide path for target language vocabulary")
    if not(len(src_merges_path)): raise AttributeError("Provide path for source language merge rules file of BPE")
    if not(len(tgt_merges_path)): raise AttributeError("Provide path for target language merge rules file of BPE")
    if not(len(model_path)): raise AttributeError("Provide path for trained model")
    
    
    
    
        
    
    if mode.lower() == 'test':
        if not(len(src_path)): raise AttributeError("Provide path for source language data")
        if not(len(tgt_path)): raise AttributeError("Provide path for target language data")
        model, src_vocab, tgt_vocab, src_word_ind, tgt_word_ind, model_config, src_tensor, tgt_tensor = load_paths(model_path,
                                                                                                        src_path, 
                                                                                                        tgt_path, 
                                                                                                        src_vocab_path,
                                                                                                        tgt_vocab_path,
                                                                                                        src_merges_path, 
                                                                                                        tgt_merges_path,
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
                                                                                            src_merges_path, 
                                                                                            tgt_merges_path,
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
        
    
    
