from transformers import PreTrainedTokenizerFast, TFMarianMTModel, MarianConfig, MarianTokenizer
import argparse
import tensorflow as tf
import tqdm
import torch
from tensorflow.nn import softmax
import time 
import numpy as np
import io
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 

encoder_interpreter_path = '/home/t-hdiddee/INMT-lite/experimental/models/hi-gondi/tfb_hi_gondi_28_encoder.tflite'
decoder_interpreter_path =  '/home/t-hdiddee/INMT-lite/experimental/models/hi-gondi/tfb_hi_gondi_28_decoder.tflite'
# model_path = '/home/t-hdiddee/INMT-Lite/final_models/Dm/marian_hi_gondi_distilled/'

tokenizer = MarianTokenizer(vocab='/home/t-hdiddee/INMT-lite/experimental/models/hi-gondi/tokenizer/concatenated_hi_gondi_vocab.json', source_spm = '/home/t-hdiddee/INMT-lite/experimental/models/hi-gondi/tokenizer/spiece_test_hi.model', target_spm = '/home/t-hdiddee/INMT-lite/experimental/models/hi-gondi/tokenizer/spiece_test_gondi.model',  bos_token = "<s>", eos_token = "</s>", pad_token = "<pad>", unk_token = "<unk>")
eml = 28
samples =  io.open('/home/t-hdiddee/INMT-lite/experimental/data/hi-gondi/test.hi').read().strip().split('\n')

BENCHMARK_PATH = './distilled_spm_gondi_model_tflite_inference_bm.txt'

encoder_interpreter = tf.lite.Interpreter(model_path = encoder_interpreter_path)
encoder_input_details = encoder_interpreter.get_input_details()
encoder_output_details = encoder_interpreter.get_output_details()


encoder_interpreter.allocate_tensors()  

decoder_interpreter = tf.lite.Interpreter(model_path = decoder_interpreter_path)
decoder_input_details = decoder_interpreter.get_input_details()
decoder_output_details = decoder_interpreter.get_output_details()

decoder_interpreter.allocate_tensors()
predictions = []

for sample in samples: 
    k=0
    batch = tokenizer(sample, return_tensors = 'tf',  truncation = True, padding='max_length', max_length = eml)

    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
        
    encoder_interpreter.set_tensor(encoder_input_details[0]['index'],input_ids)
    encoder_interpreter.set_tensor(encoder_input_details[1]['index'],attention_mask) 

    encoder_interpreter.invoke()

    encoder_outputs = encoder_interpreter.get_tensor(encoder_output_details[0]['index']) 

    initial = '<s>'
    decoder_input_ids = tokenizer.encode(initial, return_tensors="tf", truncation = True, padding='max_length', max_length = eml, add_special_tokens=False)

    decoder_input_ids = decoder_input_ids.numpy().astype('int32')
    decoder_input_ids[0][0] = 1
    decoder_input_ids[0][1] = 0
    decoder_input_ids[0][2] = 0
    decoder_input_ids[0][3] = 0
    decoder_interpreter.set_tensor(decoder_input_details[0]['index'],decoder_input_ids)  
    decoder_interpreter.set_tensor(decoder_input_details[1]['index'],encoder_outputs) 

    decoder_interpreter.invoke()        
    lm_logits = decoder_interpreter.get_tensor(decoder_output_details[0]['index'])  

    next_decoder_input_ids = torch.argmax(torch.from_numpy(lm_logits[:,k]), axis=-1)

    next_decoder_input_ids = torch.from_numpy(np.array([next_decoder_input_ids.numpy()]))
    decoder_interpreter.set_tensor(decoder_input_details[1]['index'],encoder_outputs) 

    while next_decoder_input_ids!=tokenizer.eos_token_id:
        try: 
            decoder_input_ids = decoder_input_ids.numpy().astype('int32') # If this is an eager tensor 
            decoder_interpreter.set_tensor(decoder_input_details[0]['index'],decoder_input_ids) 
        except:  
            decoder_interpreter.set_tensor(decoder_input_details[0]['index'],decoder_input_ids)
        decoder_interpreter.invoke()  
        lm_logits = decoder_interpreter.get_tensor(decoder_output_details[0]['index'])  

        next_decoder_input_ids = torch.argmax(torch.from_numpy(lm_logits[:,k]), axis=-1)
        next_decoder_input_ids = torch.from_numpy(np.array([next_decoder_input_ids.numpy()]))
        
        k += 1 
        decoder_input_ids[0][k] = next_decoder_input_ids
        decoder_interpreter.reset_all_variables()
    predictions.append(tokenizer.decode(decoder_input_ids[0], skip_special_tokens=True))
    encoder_interpreter.reset_all_variables()
    print(f'TfLite Output: {tokenizer.decode(decoder_input_ids[0], skip_special_tokens=True)}.')
    
with open(BENCHMARK_PATH, 'w+', encoding='UTF-8' ) as file:
    for pred in predictions:
        file.write(pred)
        file.write('\n')