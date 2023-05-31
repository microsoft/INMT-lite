from transformers import TFMT5ForConditionalGeneration, T5Tokenizer
import sentencepiece as spm
import tensorflow as tf 
import os 
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 

ENCODER_MAX_LEN = 28
DECODER_MAX_LEN = 28
ENTIRE_MODEL_SAVE_PATH = "/home/t-hdiddee/INMT-Lite/final_models/mt5_small_esp_bzd/"
ENCODER_TFLITE_PATH =  "/home/t-hdiddee/INMT-Lite/tflite_models/mt5_small_esp_bzd_encoder.tflite"
DECODER_TFLITE_PATH = "/home/t-hdiddee/INMT-Lite/tflite_models/mt5_small_esp_bzd_decoder.tflite"

def shape_list(x):
    """Deal with dynamic shape in tensorflow cleanly."""
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]

def getCustomEncoder(encoder, config):
    input_ids =  tf.keras.layers.Input(shape=(ENCODER_MAX_LEN, ), dtype = tf.int32) # (1,ENCODER_MAX_LEN)
    attention_mask = tf.keras.layers.Input(shape=(ENCODER_MAX_LEN, ), dtype = tf.int32) # (1, ENCODER_MAX_LEN)
    encoded_sequence = encoder(input_ids = input_ids, attention_mask = attention_mask, output_hidden_states = config.output_hidden_states)
    return tf.keras.Model(inputs = [input_ids, attention_mask], outputs = [encoded_sequence])

def getCustomDecoder(layer, config):
    decoder = layer.decoder
    lm_head = layer.lm_head
    input_ids =  tf.keras.layers.Input(shape=(ENCODER_MAX_LEN, ), dtype = tf.int32) # (1,ENCODER_MAX_LEN)
    decoder_input_ids = tf.keras.layers.Input(shape=(DECODER_MAX_LEN, ), dtype = tf.int32) # (1,ENCODER_MAX_LEN)
    decoder_attention_mask = tf.keras.layers.Input(shape=(DECODER_MAX_LEN, ), dtype = tf.int32) # (1,ENCODER_MAX_LEN)
    encoder_outputs =  tf.keras.layers.Input(shape=(ENCODER_MAX_LEN, config.d_model), dtype = tf.float32) # (1,ENCODER_MAX_LEN, 512)
    decoder_outputs = decoder(tf.convert_to_tensor(decoder_input_ids),encoder_hidden_states=encoder_outputs)
    lm_logits = lm_head(decoder_outputs[0])
    return tf.keras.Model(inputs = [decoder_input_ids, encoder_outputs], outputs = [lm_logits])


if __name__=='__main__':

    # Loading trained model 

    model = TFMT5ForConditionalGeneration.from_pretrained(ENTIRE_MODEL_SAVE_PATH, from_pt = True)
    model.save_pretrained(ENTIRE_MODEL_SAVE_PATH)
    model = TFMT5ForConditionalGeneration.from_pretrained(ENTIRE_MODEL_SAVE_PATH)
    
    tokenizer = T5Tokenizer.from_pretrained("google/mt5-small")
    
    encoder = getCustomEncoder(encoder = model.get_encoder(), config = model.config)
    print('Converting Encoder to TFlite')
    converter = tf.lite.TFLiteConverter.from_keras_model(encoder)
    print('Optimizing Encoder Size Reduction with the Keras Model')
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [ tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS ]

    tflite_model = converter.convert()    
    print('Writing Encoder TFLite File')
    with open(ENCODER_TFLITE_PATH,'wb') as file:
        file.write(tflite_model)

    decoder = getCustomDecoder( layer = model, config = model.config)

    print('Converting Decoder to TFlite')
    converter = tf.lite.TFLiteConverter.from_keras_model(decoder)
    print('Optimizing Decoder Size Reduction with the Keras Model')
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [ tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS ]

    tflite_model = converter.convert()    
    print('Writing Decoder TFLite File')
    with open(DECODER_TFLITE_PATH,'wb') as file:
        file.write(tflite_model)
    