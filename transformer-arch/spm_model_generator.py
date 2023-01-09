import io 
import sentencepiece as spm
import json 
from tokenizers import SentencePieceBPETokenizer
from tokenizers.processors import TemplateProcessing

SRC_LANG = 'hi'
TGT_LANG = 'gondi'

TRAIN_SRC = f'/home/t-hdiddee/INMT-Tflite/data/hi-gondi/gold.{SRC_LANG}'  
TRAIN_TGT = f'/home/t-hdiddee/INMT-Tflite/data/hi-gondi/gold.{TGT_LANG}'

DEV_SRC = f'/home/t-hdiddee/INMT-Tflite/data/hi-gondi/test.{SRC_LANG}'
DEV_TGT = f'/home/t-hdiddee/INMT-Tflite/data/hi-gondi/test.{TGT_LANG}'

train_dataset = f'/home/t-hdiddee/INMT-Tflite/data/train_spm_test.json'
val_dataset = f'/home/t-hdiddee/INMT-Tflite/data/val_spm_test.json'
vocab_path = f'./hi_gondi_extracted_vocab.json'

source_vocab_size = 8000
target_vocab_size = 8000

def convert_data_to_jsonl(source, target, dataset_path):
    with open(source, 'r') as source:
        source_seq = [line for line in source.read().split('\n')]
    with open(target, 'r') as target:
        target_seq = [line for line in target.read().split('\n')]
    print(len(source_seq), len(target_seq))

    dataset = []
    for src, tgt in zip(source_seq, target_seq):
        item = {'translation': {SRC_LANG: src, TGT_LANG: tgt}}
        dataset.append(item)
 
    with open(dataset_path, 'w') as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii = False) + '\n')

def batch_iterator(dataset):
    for i in range(0, len(dataset), batch_size):
        yield dataset[i : i + batch_size]

def create_vocab(file, lang, vocab_size):
       spm.SentencePieceTrainer.train(f'--input={file} --vocab_size={vocab_size} --model_type=bpe --model_prefix=spiece_test_{lang} --pad_id=2 --bos_id=0 --eos_id=1 --unk_id=3')

if __name__ == "__main__":
    print('Converting Training Data into specific format')
    convert_data_to_jsonl(TRAIN_SRC, TRAIN_TGT, train_dataset)
    print('Converting Evaluation Data into specific format')
    convert_data_to_jsonl(DEV_SRC, DEV_TGT, val_dataset)
    with open(TRAIN_SRC, 'r', encoding = 'UTF-8') as file:
        train_src = [line for line in file.read().split('\n')]
    with open(TRAIN_TGT, 'r', encoding = 'UTF-8') as file:
        train_tgt = [line for line in file.read().split('\n')]
    

    create_vocab(TRAIN_SRC, SRC_LANG, source_vocab_size) 
    create_vocab(TRAIN_TGT, TGT_LANG, target_vocab_size) 
    print('Native Sentencepiece Vocab files generated. Pass these to spm_extractor.py') 
    