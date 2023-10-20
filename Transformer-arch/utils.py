from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
# from transformers import PreTrainedTokenizerFast, MarianTokenizer
# import tensorflow as tf
import tqdm 
import io
from tokenizers import decoders
import sys
import json
import logging
import numpy as np
import os
import regex as re
from io import open
# import datasets
import indicnlp
import sys
from indicnlp import common
from indicnlp import loader

# INDIC_NLP_RESOURCES= r"./indic_nlp_resources" # Add path to local package 
# common.set_resources_path(INDIC_NLP_RESOURCES)
from indicnlp.transliterate.unicode_transliterate import ItransTransliterator
# loader.load()

SRC_LANG = 'en'
TGT_LANG = 'hi'

# TGT_TOKENIZER_PATH = f"/home/t-hdiddee/INMT-lite/INMT-lite/inmt_tflite/transformer/resources/{SRC_LANG}-{TGT_LANG}/{TGT_LANG}.json"
ENCODER_MAX_LEN = 48
DECODER_MAX_LEN = 48
# tgt_tokenizer = PreTrainedTokenizerFast(tokenizer_file=TGT_TOKENIZER_PATH)
# tgt_tokenizer.add_special_tokens({'pad_token': '<pad>'})
# metric = datasets.load_metric('sacrebleu')
# tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-hi')

def shape_list(x):
    """Deal with dynamic shape in tensorflow cleanly."""
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]

def shift_tokens_right(labels, start_token_id):
    decoder_inputs = tf.concat([[start_token_id], labels[:-1]], -1)
    return decoder_inputs


def encode_data(src, tgt, src_tokenizer, tgt_tokenizer, config, encoder_max_len = ENCODER_MAX_LEN, decoder_max_len = DECODER_MAX_LEN):
    '''
    Returns the input_ids, attention_mask, labels and decoder_attention_mask for sequences during TRAINING
    '''

    input_ids_list, input_attention_lists, decoder_input_ids_lists, label_list = [], [], [], []
    for sentence in tqdm.tqdm(zip(src, tgt)):

        src_shape, tgt_shape = len(sentence[0]), len(sentence[1])
        encoder_inputs = src_tokenizer.encode(sentence[0], truncation = True, return_tensors = 'tf', padding='max_length', max_length = encoder_max_len)[0]
        label = tgt_tokenizer.encode(sentence[1], truncation = True, return_tensors = 'tf', padding='max_length', max_length = decoder_max_len)[0]
        decoder_inputs = shift_tokens_right(label, src_tokenizer.pad_token_id)
        
        
        ''' Attention_mask (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values are in ``[0, 1]``, 1 for
                tokens that are not masked, and 0 for masked tokens. If not provided, will default to a tensor the same
                shape as :obj:`input_ids` that masks the pad token. 
                Attention masks are manually generated following logic specified here - https://huggingface.co/transformers/glossary.html#attention-mask
                Special Token Dictionary is - {<s>: 0, </s>: 1, <pad>: 20000}
        '''
        # attention_mask = tf.cast(tf.math.not_equal(input_ids, pad_token_id), dtype=tf.int32) 
        input_attention_mask = [1 for input in encoder_inputs if input!=20000]
        input_attention_mask.extend([0]*(ENCODER_MAX_LEN - len(input_attention_mask)))  
        input_attention_mask = tf.convert_to_tensor(input_attention_mask)



        input_ids_list.append(encoder_inputs)
        decoder_input_ids_lists.append(decoder_inputs)
        input_attention_lists.append(input_attention_mask)
        label_list.append(label)
    print(encoder_inputs, decoder_inputs, input_attention_mask, label)
    return input_ids_list, input_attention_lists, decoder_input_ids_lists, label_list

def to_tfds(dataset):  
  columns = ['input_ids', 'attention_mask', 'decoder_input_ids', 'labels']
  # dataset.set_format(type='tensorflow', columns=columns)
  return_types = {'input_ids':tf.int32, 'attention_mask':tf.int32, 
                'decoder_input_ids':tf.int32, 'labels':tf.int32,  }
  return_shapes = {'input_ids': tf.TensorShape([None]), 'attention_mask': tf.TensorShape([None]), 
                  'decoder_input_ids': tf.TensorShape([None]), 'labels':tf.TensorShape([None])}
  ds = tf.data.Dataset.from_generator(lambda : dataset, return_types, return_shapes)
  return ds

def create_dataset(dataset, cache_path=None, batch_size=4,buffer_size= 1000, shuffling=True):    
    if cache_path is not None:
        dataset = dataset.cache(cache_path)        
    if shuffling:
        dataset = dataset.shuffle(buffer_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

def check_stats(source = "/home/t-hdiddee/INMT-lite/INMT-lite/inmt_tflite/transformer/data/hi-gondi/train_s1_s2.gondi"):
    with open(source, 'r') as source:
        source_seqs = source.read().split('\n')
    print(len(source_seqs))
    count = 0 
    for seq in source_seqs: 
        if len(seq) > 64: 
            count+=1 
    print(f'Number of exceeding sentences is {count}')

def compute_bleu(model, tf_eval_dataset, eval_batch_size):
    for batch, labels in tqdm.tqdm(tf_eval_dataset, total=len(tf_eval_dataset) // eval_batch_size ):
        generated_tokens = model.generate(input_ids = tf.convert_to_tensor((batch['input_ids'], batch['attention_mask'])))
        if isinstance(generated_tokens, tuple):
            generated_tokens = generated_tokens[0]
        decoded_preds = tgt_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        print(decoded_preds)
        labels = np.where(labels != 20000, labels, tgt_tokenizer.pad_token_id)
        print(labels)
        decoded_labels = tgt_tokenizer.decode(labels, skip_special_tokens=True)
        metric.add_batch(predictions=decoded_preds, references=decoded_labels)
    eval_metric = metric.compute()
    print(f'BLEU on validation set: {eval_metric["score"]}')


# Taken from HF :  https://huggingface.co/transformers/v2.3.0/_modules/transformers/tokenization_gpt2.html

def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a mapping to unicode strings.
    We specifically avoids mapping to whitespace/control characters the bpe code barfs on.

    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    """
    _chr = unichr if sys.version_info[0] == 2 else chr
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [_chr(n) for n in cs]
    return dict(zip(bs, cs))

byte_encoder = bytes_to_unicode()
byte_decoder = {v: k for k, v in byte_encoder.items()}

def convert_tokens_to_string(tokens):
    """ Converts a sequence of tokens (string) in a single string. """
    text = ''.join(tokens)
    text = bytearray([byte_decoder[c] for c in text]).decode('utf-8')
    return text

def shape_list(x):
    """Deal with dynamic shape in tensorflow cleanly."""
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]


def transliterate(data_path, transliterated_file_path):    
    with open(data_path, 'r', encoding = "UTF-8") as file:
        sentences = file.read()
    sentences = sentences.split('\n')
    print(len(sentences))

    transliterated_sentences = []
    for sent in tqdm.tqdm(sentences): 
        transliterated_sentences.append(ItransTransliterator.from_itrans(sent,"hi"))

    with open(transliterated_file_path, 'w', encoding = "UTF-8") as file: 
        for sent in transliterated_sentences:
            file.write(sent)
            file.write('\n')

if __name__ == '__main__':
    with open('/home/t-hdiddee/INMT-lite/transformer-arch/misc/Dialogue_Key.json', 'r') as file: 
        records = file.read().split('\n')
        lines = []
        for record in records:
            lines.append(json.load(record))
        print(len(lines))