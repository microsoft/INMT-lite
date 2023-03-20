from transformers import PreTrainedTokenizerFast
from transformers import MarianConfig, TFMarianMTModel
import tensorflow as tf 
import os 
import numpy as np

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 

def shape_list(x):
    """Deal with dynamic shape in tensorflow cleanly."""
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]

ENTIRE_MODEL_SAVE_PATH = "/home/t-hdiddee/ACL/en-bn" # Latest
SRC_LANG = 'en'
TGT_LANG = 'bn'
ENCODER_MAX_LEN = 28
DECODER_MAX_LEN = 28


custom_tf_model = TFMarianMTModel.from_pretrained(ENTIRE_MODEL_SAVE_PATH, from_pt=True)
custom_tf_model.save_pretrained(ENTIRE_MODEL_SAVE_PATH)
ENCODER_TFLITE_PATH = ENTIRE_MODEL_SAVE_PATH + f"/tfb_{SRC_LANG}_{TGT_LANG}_{ENCODER_MAX_LEN}_encoder.tflite"
DECODER_TFLITE_PATH = ENTIRE_MODEL_SAVE_PATH + f"/tfb_{SRC_LANG}_{TGT_LANG}_{DECODER_MAX_LEN}_decoder.tflite"

def getCustomEncoder(encoder, config):
   
    input_ids =  tf.keras.layers.Input(shape=(ENCODER_MAX_LEN,), dtype = tf.int32) # (1,ENCODER_MAX_LEN)
    attention_mask = tf.keras.layers.Input(shape=(ENCODER_MAX_LEN, ), dtype = tf.int32) # (1, ENCODER_MAX_LEN)
    encoded_sequence = encoder(input_ids = input_ids, attention_mask = attention_mask, output_hidden_states = config.output_hidden_states)
    return tf.keras.Model(inputs = [input_ids, attention_mask], outputs = [encoded_sequence])

def getCustomDecoder(layer, config):
    decoder = layer.decoder
    shared = layer.shared
    input_ids =  tf.keras.layers.Input(shape=(ENCODER_MAX_LEN, ), dtype = tf.int32) # (1,ENCODER_MAX_LEN)
    decoder_input_ids = tf.keras.layers.Input(shape=(ENCODER_MAX_LEN, ), dtype = tf.int32) # (1,ENCODER_MAX_LEN)
    decoder_attention_mask = tf.keras.layers.Input(shape=(ENCODER_MAX_LEN, ), dtype = tf.int32) # (1,ENCODER_MAX_LEN)
    encoder_outputs =  tf.keras.layers.Input(shape=(ENCODER_MAX_LEN, config.d_model), dtype = tf.float32) # (1,ENCODER_MAX_LEN, 512)
    
    decoder_outputs = decoder(tf.convert_to_tensor(decoder_input_ids),encoder_hidden_states=encoder_outputs)    
    first_dims = shape_list(decoder_outputs[0])[:-1]
    x = tf.reshape(decoder_outputs[0], [-1, config.d_model])
    logits = tf.matmul(x, shared.weight.numpy(), transpose_b=True)
    lm_logits = tf.reshape(logits, first_dims + [config.vocab_size])
  
    return tf.keras.Model(inputs = [decoder_input_ids, encoder_outputs], outputs = [lm_logits])

if __name__=='__main__':

    # Loading trained model 

    model = TFMarianMTModel.from_pretrained(ENTIRE_MODEL_SAVE_PATH)

    encoder = getCustomEncoder(encoder = model.model.encoder, config = model.model.config)
    
    print('Converting Encoder to TFlite')
    converter = tf.lite.TFLiteConverter.from_keras_model(encoder)
    print('Optimizing Encoder Size Reduction with the Keras Model')
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [ tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS ]
    converter.experimental_enable_resource_variables = True
    tflite_model = converter.convert()    
    print('Writing Encoder TFLite File')
    with open(ENCODER_TFLITE_PATH,'wb') as file:
        file.write(tflite_model)

    decoder = getCustomDecoder(layer = model.model, config = model.model.config)

    print('Converting Decoder to TFlite')
    converter = tf.lite.TFLiteConverter.from_keras_model(decoder)
    print('Optimizing Decoder Size Reduction with the Keras Model')
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [ tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS ]
    converter.experimental_enable_resource_variables = True
    tflite_model = converter.convert()    
    print('Writing Decoder TFLite File')
    with open(DECODER_TFLITE_PATH,'wb') as file:
        file.write(tflite_model)