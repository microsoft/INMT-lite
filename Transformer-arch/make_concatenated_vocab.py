import json 

src_vocab_path = "/home/t-hdiddee/INMT-Tflite/data/vocab/spm_vocab.hi"
tgt_vocab_path = "/home/t-hdiddee/INMT-Tflite/data/vocab/spm_vocab.gondi"


concatenated_vocab_header = '{"version":"1.0","truncation":null,"padding":null,"added_tokens":[{"id":0,"special":true,"content":"<s>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false},{"id":1,"special":true,"content":"</s>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false},{"id":2,"special":true,"content":"<pad>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false}],"normalizer":{"type":"NFKC"},"pre_tokenizer":{"type":"Metaspace","replacement":"▁","add_prefix_space":true},"post_processor":{"type":"TemplateProcessing","single":[{"SpecialToken":{"id":"<s>","type_id":0}},{"Sequence":{"id":"A","type_id":0}},{"SpecialToken":{"id":"</s>","type_id":0}}],"pair":[{"Sequence":{"id":"A","type_id":0}},{"Sequence":{"id":"B","type_id":1}}],"special_tokens":{"</s>":{"id":"</s>","ids":[1],"tokens":["</s>"]},"<s>":{"id":"<s>","ids":[0],"tokens":["<s>"]}}},"decoder":{"type":"Metaspace","replacement":"▁","add_prefix_space":true},"model":{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"vocab":{},"merges":[]}}'
concatenated_vocab_path = './concatenated_hi_gondi_vocab.json'

start_idx = 8000 # Vocab_size of the source_lang: so if the vocab sizes for the spm models is 16K, 32K: The start_idx will be 16K ( Because of zero indexing )

with open(src_vocab_path, mode = 'r') as file: 
    source_vocab = json.load(file)
with open(tgt_vocab_path, mode = 'r') as file: 
    tgt_vocab = json.load(file)

shared_vocab = source_vocab
for key in tgt_vocab: 
    if key not in shared_vocab:
        shared_vocab[key]=start_idx
        start_idx += 1 

with open(concatenated_vocab_path,'w') as file: 
    json.dump(shared_vocab, file, ensure_ascii=False)
