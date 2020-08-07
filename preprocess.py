from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import os
import io
import unicodedata
import re
import json 
import sys, getopt


def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s))

def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())
    
    #Add a space between word and punctuation
    w = re.sub(r"([?.!,¿|-])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
    
    w = w.strip()
    
    w = '<start> ' + w + ' <end>'
    return w

def create_dataset(path_src_train, path_tgt_train, path_src_val, path_tgt_val, seed, num_examples):
    print("Creating dataset...")
    print(path_src_train)
    lines_src_train = io.open(path_src_train, encoding='UTF-8').read().strip().split('\n')
    print(path_tgt_train)
    lines_tgt_train = io.open(path_tgt_train, encoding='UTF-8').read().strip().split('\n')
    lines_src_val = io.open(path_src_val, encoding='UTF-8').read().strip().split('\n')
    lines_tgt_val = io.open(path_tgt_val, encoding='UTF-8').read().strip().split('\n')

    
    
    #sanity check
    assert len(lines_src_train) == len(lines_tgt_train)
    assert len(lines_src_val) == len(lines_src_val)

    print("Total sentences detected for training:", len(lines_src_train))
    print("Total sentences detected for validation:", len(lines_src_val))
    
    num_examples = len(lines_src_train) if num_examples is None else num_examples
    #TODO: Pass the seed as an argument
    indexes = np.random.RandomState(seed=seed).permutation(len(lines_src_train))[:num_examples]
    indexes = [i-1 for i in indexes]
    print(len(indexes), num_examples)
    src_train_data = [preprocess_sentence(lines_src_train[i]) for i in indexes]
    tgt_train_data = [preprocess_sentence(lines_tgt_train[i]) for i in indexes]
    src_val_data = [preprocess_sentence(i) for i in lines_src_val] # We do not sample validation data
    tgt_val_data = [preprocess_sentence(i) for i in lines_tgt_val]
    
    print("Dataset Created")
    
    return src_train_data, tgt_train_data, src_val_data, tgt_val_data

def tokenize(lang, sent_len, padding='post', lang_tokenizer = None, num_words=None):
    if lang_tokenizer is None:
        if num_words:
            lang_tokenizer = Tokenizer(filters='', oov_token="<OOV>", num_words=num_words)
        else:
            lang_tokenizer = Tokenizer(filters='', oov_token="<OOV>")
        
        lang_tokenizer.fit_on_texts(lang)
        
        if num_words:
            lang_tokenizer.index_word['num_words'] = num_words+1
        else:
            lang_tokenizer.index_word['num_words'] = len(lang_tokenizer.word_index)+1 #Pad will be included
    
    tensor = lang_tokenizer.texts_to_sequences(lang)
    
    tensor = pad_sequences(tensor, padding=padding, truncating='post', maxlen=sent_len)
    
    
    return tensor, lang_tokenizer

def load_dataset(
            path_src_train, 
            path_tgt_train, 
            path_src_val, 
            path_tgt_val, 
            seed, 
            num_examples,
            length_src_vocab,
            length_tgt_vocab
            ):
    src_train_data, tgt_train_data, src_val_data, tgt_val_data = create_dataset(path_src_train,
                                                                                path_tgt_train, 
                                                                                path_src_val, 
                                                                                path_tgt_val,
                                                                                seed, 
                                                                                num_examples)
    # print(en_data[-68]) # Maybe show an example of dataset looks now ?
    # print(hi_data[-68])

    print("Started Tokenising for: src_data")
    src_train_tensor, src_tokenizer = tokenize(src_train_data, sent_len=Tx, num_words=length_src_vocab)
    print("Finished Tokenising for: src_data")
    print("Started Tokenising for: tgt_data")
    tgt_train_tensor, tgt_tokenizer = tokenize(tgt_train_data, sent_len=Ty, padding='pre', num_words=length_tgt_vocab)
    print("Finished Tokenising for: tgt_data")
    
    src_val_tensor, _ = tokenize(src_val_data, sent_len=Tx, lang_tokenizer=src_tokenizer)
    tgt_val_tensor, _ = tokenize(tgt_val_data, sent_len=Ty, lang_tokenizer=tgt_tokenizer, padding='pre')

    
    return src_train_tensor, src_tokenizer, tgt_train_tensor, tgt_tokenizer, src_val_tensor, tgt_val_tensor

def write_processed_data(dir_path, file_name, tensor):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    with open( os.path.join(dir_path, file_name) , 'w' ,encoding='utf-8') as file:
        for i in tensor:
            file.write(' '.join([str(j) for j in i]) + '\n') # Possible optimisation in single loop

def write_vocab_data(dir_path, file_name, tokenizer):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    with open( os.path.join(dir_path, file_name) , 'w' ,encoding='utf-8') as file:    
        json.dump(tokenizer.index_word , file, ensure_ascii=False )


def preprocess(
            Tx, Ty,
            training_examples_num,
            length_src_vocab,
            length_tgt_vocab,
            path_src_train, 
            path_tgt_train, 
            path_src_val, 
            path_tgt_val,
            to_path_src_train,
            to_path_tgt_train,
            to_path_src_val,
            to_path_tgt_val,
            to_path_src_vocab,
            to_path_tgt_vocab,
            seed
            ):

    # Tx = 14
    # Ty = 14
    # path_to_file_en = '/home/anurag/Projects/Machine Translation/parallel/IITB.en-hi.en'
    # path_to_file_hi = '/home/anurag/Projects/Machine Translation/parallel/IITB.en-hi.hi'

    # print(preprocess_sentence(en_sentence))
    # print(preprocess_sentence(sp_sentence))

    src_train_tensor, src_tokenizer, tgt_train_tensor, tgt_tokenizer, src_val_tensor, tgt_val_tensor = load_dataset(path_src_train,
                                                                                                                    path_tgt_train,
                                                                                                                    path_src_val,
                                                                                                                    path_tgt_val,
                                                                                                                    seed,
                                                                                                                    training_examples_num,
                                                                                                                    length_src_vocab,
                                                                                                                    length_tgt_vocab
                                                                                                                    )
    
    
    
    # Writing processed data to file
    write_processed_data(to_path_src_train, 'src_train_processed.txt', src_train_tensor)
    write_processed_data(to_path_tgt_train, 'tgt_train_processed.txt', tgt_train_tensor)
    write_processed_data(to_path_src_val, 'src_val_processed.txt', src_val_tensor)
    write_processed_data(to_path_tgt_val, 'tgt_val_processed.txt', tgt_val_tensor)
    
    # Writing Vocabulary to file
    
    write_vocab_data(to_path_src_vocab, 'src_vocab.json', src_tokenizer)
    write_vocab_data(to_path_tgt_vocab, 'tgt_vocab.json', tgt_tokenizer)
    

if __name__ == "__main__":
    
    to_path_default = './data'
    
    Tx = 14
    Ty = 14
    training_examples_num = None
    length_src_vocab = None
    length_tgt_vocab = None
    path_src_train = ''
    path_tgt_train = ''
    path_src_val = ''
    path_tgt_val = ''
    to_path_tgt_train = to_path_default
    to_path_src_train = to_path_default
    to_path_src_val = to_path_default
    to_path_tgt_val = to_path_default
    to_path_src_vocab = to_path_default
    to_path_tgt_vocab = to_path_default
    seed = 42
    
    data_dir_ele = [to_path_tgt_train, to_path_src_train, to_path_src_val, to_path_tgt_val, to_path_src_vocab, to_path_tgt_vocab]
    input_data_ele = [path_src_train, path_tgt_train, path_src_val, path_tgt_val]
    
    sys_arg = sys.argv[1:]
    
    # TODO: ADD options for vobaulary
    
    
    try:
        opts, args = getopt.getopt(sys_arg,"", ['Tx=', 
                                                'Ty=', 
                                                'training_examples_num=',
                                                'length_src_vocab=',
                                                'length_tgt_vocab=',
                                                 'path_src_train=', 
                                                 'path_tgt_train=', 
                                                 'path_src_val=', 
                                                 'path_tgt_val=', 
                                                 'to_path_src_train=', 
                                                 'to_path_tgt_train=', 
                                                 'to_path_src_val=', 
                                                 'to_path_tgt_val=', 
                                                 'to_path_src_vocab=', 
                                                 'to_path_tgt_vocab=', 
                                                 'seed='
                                                 ])
    except Exception as e:
        print(e)
        
    for opt, arg in opts:
        if opt == '--Tx': Tx = int(arg)
        elif opt == '--Ty': Ty = int(arg)
        elif opt == '--training_examples_num': training_examples_num = int(arg)
        elif opt == '--length_src_vocab': length_src_vocab = int(arg)
        elif opt == '--length_tgt_vocab': length_tgt_vocab = int(arg)
        elif opt == '--path_src_train': path_src_train = arg
        elif opt == '--path_tgt_train': path_tgt_train = arg
        elif opt == '--path_src_val': path_src_val = arg
        elif opt == '--path_tgt_val': path_tgt_val = arg
        elif opt == '--to_path_src_train': to_path_src_train = arg
        elif opt == '--to_path_tgt_train': to_path_tgt_train = arg
        elif opt == '--to_path_src_val': to_path_src_val = arg
        elif opt == '--to_path_tgt_val': to_path_tgt_val = arg
        elif opt == '--to_path_src_vocab': to_path_src_vocab = arg
        elif opt == '--to_path_tgt_vocab': to_path_tgt_vocab = arg
        elif opt == '--seed': seed = int(arg)
    
    if not(len(path_src_train)): raise AttributeError("Please provide argument to: path_src_train")
    if not(len(path_tgt_train)): raise AttributeError("Please provide argument to: path_tgt_train")
    if not(len(path_src_val)): raise AttributeError("Please provide argument to: path_src_val")
    if not(len(path_tgt_val)): raise AttributeError("Please provide argument to: path_tgt_val")    
    
    preprocess(
            Tx, Ty,
            training_examples_num,
            length_src_vocab,
            length_tgt_vocab,
            path_src_train, 
            path_tgt_train, 
            path_src_val, 
            path_tgt_val,
            to_path_src_train,
            to_path_tgt_train,
            to_path_src_val,
            to_path_tgt_val,
            to_path_src_vocab,
            to_path_tgt_vocab,
            seed
            )
    
        
    
    
    
