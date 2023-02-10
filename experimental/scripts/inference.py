from transformers import ( PreTrainedTokenizerFast, TFMarianMTModel, MarianConfig, MT5ForConditionalGeneration, 
T5Tokenizer,MBartForConditionalGeneration, MBart50TokenizerFast, MarianTokenizer)
import argparse
import tensorflow as tf
import tqdm
import torch
import time 
import numpy as np
import io
import os


os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 

predictions = []

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels

def seq_online(model, tokenizer, src_samples, task_prefix, return_tensor):
    predictions = []
    for sample in src_samples: 
        sample = task_prefix + sample
        batch = tokenizer(sample, return_tensors=return_tensor, truncation=True, padding='max_length', max_length = 48) 
        output = model.generate(batch['input_ids'], max_new_tokens = 48)
        prediction = tokenizer.decode(output[0], skip_special_tokens=False)
        print(prediction)
        predictions.append(prediction)
    return predictions 

def online(model, tokenizer, src_samples, task_prefix, return_tensor):
    samples = [task_prefix + sample for sample in src_samples]
    batch = tokenizer(samples, return_tensors=return_tensor, truncation=True, padding='max_length', max_length = 48) 
    output = model.generate(**batch, max_new_tokens = 127)
    predictions = tokenizer.batch_decode(output, skip_special_tokens=True)
    return predictions 

def offline(encoder_interpreter_path, decoder_interpreter_path, eml, dml, src_samples, tokenizer, task_prefix):
    
    encoder_interpreter = tf.lite.Interpreter(model_path = encoder_interpreter_path)
    encoder_input_details = encoder_interpreter.get_input_details()
    encoder_output_details = encoder_interpreter.get_output_details()
    print(encoder_input_details)

    encoder_interpreter.allocate_tensors()  

    decoder_interpreter = tf.lite.Interpreter(model_path = decoder_interpreter_path)
    decoder_input_details = decoder_interpreter.get_input_details()
    decoder_output_details = decoder_interpreter.get_output_details()
    print(decoder_output_details)

    decoder_interpreter.allocate_tensors()
    
    for sample in tqdm.tqdm(src_samples):
        # print(f'Source: {sample}')
        batch = tokenizer(task_prefix + sample, return_tensors = 'tf',  truncation = True, padding='max_length', max_length = eml)
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
          
        encoder_interpreter.set_tensor(encoder_input_details[0]['index'],input_ids)
        encoder_interpreter.set_tensor(encoder_input_details[1]['index'],attention_mask) 

        encoder_interpreter.invoke()
        
        encoder_outputs = encoder_interpreter.get_tensor(encoder_output_details[0]['index']) 
    
        initial = '<pad>'
           
        decoder_input_ids = tokenizer.encode(initial, add_special_tokens=True, return_tensors="tf", truncation = True, padding='max_length', max_length = eml)


        decoder_interpreter.set_tensor(decoder_input_details[0]['index'],decoder_input_ids)  
        decoder_interpreter.set_tensor(decoder_input_details[1]['index'],encoder_outputs) 

        decoder_interpreter.invoke()        

        decoder_interpreter.set_tensor(decoder_input_details[1]['index'],encoder_outputs) 
        
        next_decoder_input_ids = -1
        
        cache = []

        while True:
            decoder_input_ids = decoder_input_ids.numpy().astype('int32') # If this is an eager tensor 
            decoder_interpreter.set_tensor(decoder_input_details[0]['index'],decoder_input_ids)  
            decoder_interpreter.invoke()  
            lm_logits = decoder_interpreter.get_tensor(decoder_output_details[0]['index'])     
            next_decoder_input_ids = torch.argmax(torch.from_numpy(lm_logits[:, -1:]), axis=-1)
            # etime = time.time()
            # print(etime - stime)
            cache.append(next_decoder_input_ids)
            decoder_input_ids = np.array([decoder_input_ids[0][1:]])  # 1: is to leave space for the previous token that would be concatenated to the decoder_input_ids             
            decoder_input_ids = torch.cat([torch.from_numpy(decoder_input_ids), next_decoder_input_ids], axis=-1)
            
            if len(cache) > 5 and next_decoder_input_ids==tokenizer.eos_token_id:
                break
            if len(cache) > eml:
                break
        print(f'Output: {tokenizer.decode(decoder_input_ids[0], skip_special_tokens=True)}.')
        predictions.append(tokenizer.decode(decoder_input_ids[0], skip_special_tokens=True))

    return predictions 
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_lang", type=str, default = None)
    parser.add_argument("--tgt_lang", type=str, default = None )
    parser.add_argument("--benchmark_path", type=str, default = './inference.txt')
    parser.add_argument("--return_tensor", type=str, default = 'tf')
    parser.add_argument("--model_arch", type=str, default = 'marian')
    parser.add_argument("--model_path", type=str, default = None)
    parser.add_argument("--vocab_path", type=str, default = None)
    parser.add_argument("--source_spm", type=str, default = None)
    parser.add_argument("--target_spm", type=str, default = None)
    parser.add_argument("--mode", type=str, default = "online")
    parser.add_argument("--task_prefix", type = str, default = "")
    parser.add_argument("--src_file", type=str)
    parser.add_argument("--encoder_interpreter_path", type=str, default = None)
    parser.add_argument("--decoder_interpreter_path", type=str, default = None)
    parser.add_argument("--eml", type=str, default = 28)
    parser.add_argument("--dml", type=str, default = 28)
        
    args = parser.parse_args()
    if "mt5" in args.model_arch: 
        tokenizer = T5Tokenizer.from_pretrained("google/mt5-small")
        if args.mode == 'online':
            model = MT5ForConditionalGeneration.from_pretrained(args.model_path)
            assert len(args.task_prefix) > 2, "Haven't passed a task prefix for mt5-type model. Please pass task prefix."
    elif "mbart" in args.model_arch:         
        tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50", src_lang=args.src_lang, tgt_lang=args.tgt_lang)
        if args.mode == 'online':
            model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50")
            args.return_tensor = 'pt'
    else:
        tokenizer = MarianTokenizer(vocab=args.vocab_path, source_spm = args.source_spm, target_spm = args.target_spm,bos_token = "<s>", eos_token = "</s>", pad_token = "<pad>", unk_token = "<unk>")
        if args.mode == 'online':
            # custom_tf_model = TFMarianMTModel.from_pretrained(args.model_path, from_pt=True)
            # custom_tf_model.save_pretrained(args.model_path)
            model =  TFMarianMTModel.from_pretrained(pretrained_model_name_or_path = args.model_path)    
        
    src_samples = io.open(args.src_file, encoding='UTF-8').read().strip().split('\n')

    if args.mode == "online":       
        print(model.config)
        predictions = online(model, tokenizer, src_samples, args.task_prefix, args.return_tensor)

    elif args.mode == "offline":
        predictions = offline(encoder_interpreter_path = args.encoder_interpreter_path, decoder_interpreter_path=args.decoder_interpreter_path, eml=args.eml, dml=args.dml, src_samples = src_samples, tokenizer=tokenizer, task_prefix = args.task_prefix)
    with open(args.benchmark_path, 'w+', encoding='UTF-8' ) as file:
        for pred in predictions:
            file.write(pred)
            file.write('\n')