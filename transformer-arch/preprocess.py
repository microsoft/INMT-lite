import io 
import sentencepiece as spm
import json 
import argparse 
from tokenizers.normalizers import NFKC
from tokenizers import SentencePieceBPETokenizer

from tokenizers.processors import TemplateProcessing
from sklearn.model_selection import train_test_split

def generate_train_test_splits(src_file, tgt_file, root): 
    complete_src = io.open(src_file).read().strip().split('\n')
    complete_tgt = io.open(tgt_file).read().strip().split('\n')
    X_train, X_test, y_train, y_test = train_test_split(complete_src, complete_tgt, test_size=0.09, random_state=42)
    with open(f'{root}train_{args.src_lang}.txt', 'w') as file: 
        for sample in X_train:
            file.write(sample + '\n')
    with open(f'{root}train_{args.tgt_lang}.txt', 'w') as file: 
        for sample in y_train:
            file.write(sample + '\n')
    with open(f'{root}test_{args.src_lang}.txt', 'w') as file: 
        for sample in X_test:
            file.write(sample + '\n')
    with open(f'{root}test_{args.tgt_lang}.txt', 'w') as file: 
        for sample in y_test:
            file.write(sample + '\n')


    print("Train-Test Splits generated!")
    
    return X_train, X_test, y_train, y_test
    
def convert_data_to_jsonl(src_lang, tgt_lang, source, target, dataset_path, split = False):
    if not split: 
        with open(source, 'r', encoding = 'UTF-8') as source:
            source_seq = [line for line in source.read().split('\n')]
        with open(target, 'r', encoding = 'UTF-8') as target:
            target_seq = [line for line in target.read().split('\n')]
        print(len(source_seq), len(target_seq))
    else: 
        source_seq = source
        target_seq = target 

    dataset = []
    for src, tgt in zip(source_seq, target_seq):
        item = {'translation': {src_lang: src, tgt_lang: tgt}}
        dataset.append(item)

    with open(dataset_path, 'w') as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def batch_iterator(dataset):
    for i in range(0, len(dataset), batch_size):
        yield dataset[i : i + batch_size]

def create_vocab(file, lang, vocab_size):
    spm.SentencePieceTrainer.train(input=file, model_prefix=lang, vocab_size=vocab_size)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type = str, default = './')
    parser.add_argument("--src_lang", type=str)
    parser.add_argument("--tgt_lang", type=str)
    parser.add_argument("--train_src", type=str)
    parser.add_argument("--train_tgt", type=str)
    parser.add_argument("--test_src", type=str, default = None)
    parser.add_argument("--test_tgt", type=str, default = None)
    parser.add_argument("--vocab_size", type = int, default = 8000)
    parser.add_argument("--generate_vocab", type = bool, default = False)
    parser.add_argument("--distill", type = bool, default = False)
    parser.add_argument("--spm", type = bool, default = False)
    parser.add_argument("--split", type = bool, default = False)
    args = parser.parse_args()

    if args.distill: 
        train_file_write, val_file_write = f'{args.root}/train_distillation_{args.src_lang}_{args.tgt_lang}.json', f'{args.root}/val_distillation_{args.src_lang}_{args.tgt_lang}.json'
    else: 
        train_file_write, val_file_write = f'{args.root}/train_{args.src_lang}_{args.tgt_lang}.json', f'{args.root}/val_{args.src_lang}_{args.tgt_lang}.json'
    vocab_file_write = f'{args.root}/vocab_{args.src_lang}_{args.tgt_lang}_{args.vocab_size}.json'

    if args.split: 
        X_train, X_test, y_train, y_test = generate_train_test_splits(args.train_src, args.train_tgt, args.root)
        print('Converting Training Data into specific format')
        convert_data_to_jsonl(args.src_lang, args.tgt_lang, X_train, y_train, train_file_write, split = True)
        print('Converting Evaluation Data into specific format')
        convert_data_to_jsonl(args.src_lang, args.tgt_lang, X_test, y_test, val_file_write, split = True)
    else: 
        print('Converting Training Data into specific format')
        convert_data_to_jsonl(args.src_lang, args.tgt_lang, args.train_src, args.train_tgt, train_file_write, args.split)
        print('Converting Evaluation Data into specific format')
        convert_data_to_jsonl(args.src_lang, args.tgt_lang, args.test_src, args.test_tgt, val_file_write, args.split)
    with open(train_file_write, 'r', encoding = 'UTF-8') as file:
        train_src = [line for line in file.read().split('\n')]
    with open(val_file_write, 'r', encoding = 'UTF-8') as file:
        train_tgt = [line for line in file.read().split('\n')]
    
    #Creating a Shared vocab for the languages
    train_src.extend(train_tgt) 
    train = train_src 

    # Making a shared vocab 
    if args.generate_vocab: 
        print(f'{len(train)} samples being used to generate the vocab.')    
        tokenizer = SentencePieceBPETokenizer(unk_token="<unk>")
        # tokenizer.normalizer = NFKC() 
        tokenizer.train_from_iterator(
            train,
            vocab_size=args.vocab_size,
            show_progress=True, 
            special_tokens = ['<s>', '</s>', '<pad>']
        )
        tokenizer.post_processor = TemplateProcessing(
                single="<s> $A </s>",
                special_tokens=[
                    ("<s>", tokenizer.token_to_id("<s>")),
                    ("</s>", tokenizer.token_to_id("</s>")),
                ],
            )
        tokenizer.save(vocab_file_write)
       

    if args.spm:
        # Create Native sentencepiece vocabularies - Language specific tokenization for language-specific models. 
        create_vocab(args.train_src, args.train_src, args.vocab_size)
        create_vocab(args.train_tgt, args.train_tgt, args.vocab_size)