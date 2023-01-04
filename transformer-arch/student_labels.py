from sys import int_info
from transformers import ( PreTrainedTokenizerFast, TFMarianMTModel, 
MarianTokenizer, TFMT5ForConditionalGeneration, 
T5Tokenizer, PreTrainedTokenizerFast)
from sacrebleu.metrics import BLEU, CHRF, TER
import argparse
import tensorflow as tf
import tqdm
import torch
import numpy as np
import io
import os

predictions = []
def online(model, tokenizer, src_samples, task_prefix, return_tensor):
    samples = [task_prefix + sample for sample in src_samples]
    batch = tokenizer(samples, return_tensors=return_tensor, padding=True, truncation=True, max_length = 1024)
    output = model.generate(**batch, max_new_tokens = 1024)
    predictions = tokenizer.batch_decode(output, skip_special_tokens=True)

    return predictions 

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_lang", type=str)
    parser.add_argument("--tgt_lang", type=str)
    parser.add_argument("--root", type=str, default = '/home/t-hdiddee/INMT-Lite/data/distillation/')
    parser.add_argument("--batch_size", type=int, default = 500)
    parser.add_argument("--model_arch", type=str, default = 'marian')
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--vocab_path", type=str)
    parser.add_argument("--task_prefix", type = str, default = "")
    parser.add_argument("--max_infer_samples", type = int, default = 500000)
    parser.add_argument("--src_file", type=str)

    args = parser.parse_args()

    if "mt5" in args.model_arch: 
        tokenizer = T5Tokenizer.from_pretrained("google/mt5-small")
        model = TFMT5ForConditionalGeneration.from_pretrained(args.model_path, from_pt=True)
        assert len(args.task_prefix) > 2, "Haven't passed a task prefix for mt5-type model. Please pass task prefix."
    else:
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=args.vocab_path, bos_token = "<s>", eos_token = "</s>", pad_token = "<pad>", unk_token = "<unk>")
        custom_tf_model = TFMarianMTModel.from_pretrained(args.model_path, from_pt = True)
        custom_tf_model.save_pretrained(args.model_path)
        model = TFMarianMTModel.from_pretrained(args.model_path, from_pt = True)

        
    src_samples = io.open(args.src_file, encoding='UTF-8').read().strip().split('\n')[:args.max_infer_samples]

    #Batching and Generating Predictions as a batch 
    total_samples = len(src_samples)
    print(f'Total Samples in the file are {total_samples}')
    
    current_processed = 0 
    batch_size = args.batch_size
    batch_id = 0 
    input_path, label_path = f'{args.root}Distillation_msl_{args.src_lang}_{args.tgt_lang}.inputs', f'{args.root}Distillation_msl_{args.src_lang}_{args.tgt_lang}.labels'

    print('Starting Processing!')
    while current_processed < total_samples:
        print(f'Labels generated for {batch_id} batch.')
        processing_samples = src_samples[current_processed : current_processed + batch_size]
        predictions = online(model, tokenizer, processing_samples, args.task_prefix, return_tensor = 'tf')
        assert len(processing_samples) == len(predictions)
        with open(input_path, 'a', encoding = 'UTF-8') as input_file, open(label_path, 'a', encoding='UTF-8' ) as output_file:    
            for source, pred in zip(processing_samples, predictions):
                input_file.write(source + '\n')
                output_file.write(pred + '\n')
            current_processed += batch_size
            batch_id = current_processed//batch_size
            print(f'Labels generated for {current_processed} samples.')
        input_file.close()
        output_file.close()    