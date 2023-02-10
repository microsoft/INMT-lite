from transformers import (PreTrainedTokenizerFast, MarianMTModel, MarianConfig, MT5ForConditionalGeneration, MarianTokenizer,
T5Tokenizer)
from datasets import load_dataset
import json
import logging
import random
import argparse
import tensorflow as tf
import tqdm
import torch
import time 
import numpy as np
import io
import os
from inference import online
from utils import read_jsonl, dump_json
import argparse
logging.basicConfig(level=logging.DEBUG)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_file", type = str)
    parser.add_argument("--model_arch", type=str, default = 'mt5')
    parser.add_argument("--model_path", type=str, default = None)
    parser.add_argument("--return_tensor", type=str, default = 'pt')
    parser.add_argument("--vocab_path", type=str, default = None)
    parser.add_argument("--source_spm", type=str, default = None)
    parser.add_argument("--target_spm", type=str, default = None)
    parser.add_argument("--dump_file_name", type=str, default = 'user-study-all-task-dump.json')
    parser.add_argument("--task_prefix", type = str, default = "")
    args = parser.parse_args()

    if "mt5" in args.model_arch: 
        tokenizer = T5Tokenizer.from_pretrained("google/mt5-small")
        model = MT5ForConditionalGeneration.from_pretrained(args.model_path)
        assert len(args.task_prefix) > 2, "Haven't passed a task prefix for mt5-type model. Please pass task prefix."
    else:
        # tokenizer = PreTrainedTokenizerFast(tokenizer_file=args.vocab_path, bos_token = "<s>", eos_token = "</s>", pad_token = "<pad>", unk_token = "<unk>")
        tokenizer = MarianTokenizer(vocab=args.vocab_path, source_spm = args.source_spm, target_spm = args.target_spm, bos_token = "<s>", eos_token = "</s>", pad_token = "<pad>", unk_token = "<unk>")
        model = MarianMTModel.from_pretrained(args.model_path)
   
    logging.debug('Tokenizer and Model loaded')    
    try: 
        src_samples = io.open(args.src_file).read().split('\n')
        selected = []
        for src in src_samples:
            if len(src.split(' ')) >= 6 and len(src.split(' ')) < 15:
                if src[-1] == '।' and not(any(chr.isdigit() for chr in src)):
                    selected.append(src)
        print(f'{len(selected)} sentences are applicable for translation.')
        src_samples = random.sample(selected, 648)
        with open('main_prompts.txt', mode = 'w') as file:
            for src_sample in src_samples:
                file.write(src_sample + '\n' )
    except:
        src_samples = read_jsonl(args.src_file)
    logging.debug('Applicable sentences chosen.')
    predictions = online(model, tokenizer, src_samples, args.task_prefix, args.return_tensor)
    if dump_json(src_samples, predictions, dump_file_name = args.dump_file_name): 
        logging.debug(f'Dump written successfully at {args.dump_file_name}')