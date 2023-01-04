import wandb
import random
import numpy as np
import datasets 
from datasets import load_dataset, load_metric
from dataclasses import dataclass, field
import os 
import transformers
from typing import Optional
from transformers import (
    EarlyStoppingCallback,
    HfArgumentParser, 
    PreTrainedTokenizerFast,  
    AutoConfig,
    MarianTokenizer,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    MarianConfig,
    MarianMTModel,
    Seq2SeqTrainingArguments,
    set_seed
)
wandb.init(project="Distillation Scaling Experiments", entity="hdiddee")

prediction_counter = 0

@dataclass 
class ModelArguments:
    src_lang: str = field(default=None)
    tgt_lang: str = field(default=None)
    model_name: str = field(default = 'vanilla')
    cache_dir: str = field(default='./cached_models')
    distil_config: str = field(default=None)
    continued_pretraining: bool = field(default = False)
    vocab_path: Optional[str] = field(default=None)
    source_prefix: str = field(default='')
    train_file: str = field(default=None)
    validation_file: str = field(default=None)
    test_file: str = field(default=None)
    max_source_length: str = field(default=1024)
    max_target_length: str = field(default=1024)
    max_train_samples: int = field(default=None) #During Hyperparameter Tuning - This value is usually set between 75K-100K 
    earlier_checkpoint: str = field(default=None)
    marian_tokenizer: bool = field(default=False)
    source_spm: str = field(default=None)
    target_spm: str = field(default=None)

def main():
    parser = HfArgumentParser((ModelArguments, Seq2SeqTrainingArguments))
    args, training_args = parser.parse_args_into_dataclasses()
    args = parser.parse_args()
    source_lang, target_lang = args.src_lang, args.tgt_lang
    

    if 'vanilla' in args.model_name: 
        if args.vocab_path is None: 
            print('You have chosen a Marian-Type model and have not specified a vocab_path. This model requires a sentencepiece BPE tokenizer file generated using a HF Spiece Interface. ')
        if args.marian_tokenizer: 
            # tokenizer = MarianTokenizer(vocab = args.vocab_path , source_spm = args.source_spm , target_spm = args.target_spm, source_lang = args.src_lang, target_lang = args.tgt_lang, bos_token = "<s>", eos_token = "</s>", pad_token = "<pad>", unk_token = "<unk>")
            tokenizer = MarianTokenizer(vocab = args.vocab_path , source_spm = args.source_spm , target_spm = args.target_spm, source_lang = args.src_lang, target_lang = args.tgt_lang, bos_token = "<s>", eos_token = "</s>", pad_token = "<pad>", unk_token = "<unk>")
        else:
            tokenizer = PreTrainedTokenizerFast(tokenizer_file=args.vocab_path, bos_token = "<s>", eos_token = "</s>", pad_token = "<pad>", unk_token = "<unk>")
        if args.distil_config == '6_4':
            enc_layers = 6
            dec_layers = 4 
        elif args.distil_config == '6_3':
            enc_layers = 6
            dec_layers = 3
        elif args.distil_config == '8_6':
            enc_layers = 8
            dec_layers = 6
        else: 
            enc_layers = 6
            dec_layers = 6
        config = MarianConfig(
                            vocab_size = len(tokenizer),
                            d_model = 512,
                            encoder_layers = enc_layers,
                            decoder_layers =  dec_layers,
                            encoder_attention_heads = 8,
                            decoder_attention_heads = 8,
                            decoder_ffn_dim = 2048,
                            encoder_ffn_dim = 2048,
                            max_length = 512,
                            max_position_embeddings = 512,
                            architectures = "MarianMTModel",
                            num_beams = 3, 
                            activation_dropout = 0.0, 
                            activation_function = "swish",
                            attention_dropout = 0.0,
                            classifier_dropout = 0.0, 
                            decoder_layerdrop = 0.0, 
                            encoder_layerdrop = 0.0, 
                            eos_token_id = tokenizer.eos_token_id, 
                            bos_token_id = tokenizer.bos_token_id, 
                            pad_token_id = tokenizer.pad_token_id, 
                            decoder_start_token_id = tokenizer.bos_token_id, 
                            add_bias_logits = False, 
                            add_final_layer_norm = False, 
                            is_encoder_decoder = True, 
                            model_type = 'marian', 
                            num_hidden_layers = 6, 
                            scale_embeddings = True, 
                            static_position_embeddings = True
                            )
        model = MarianMTModel(config)   
    else:     
        if len(args.source_prefix) < 5: 
            print('You are training MT5 without an appropriate training prefix. Please provide a prefix.')
        config = AutoConfig.from_pretrained(args.model_name,cache_dir=args.cache_dir)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name, config=config, cache_dir=args.cache_dir)
    
    prefix = args.source_prefix if args.source_prefix is not None else ""
    
    data_files = {}
    if args.train_file is not None: 
        data_files["train"] = args.train_file
    if args.validation_file is not None: 
        data_files["validation"] = args.validation_file
    if args.test_file is not None:
        data_files["test"] = args.test_file
    raw_datasets = load_dataset('json', data_files=data_files, cache_dir=args.cache_dir)
    
    tokenizer.source_lang = args.src_lang
    tokenizer.target_lang = args.tgt_lang

    def convert_to_sentinel(samples):
        sentinel_samples = []
        for sample in samples: 
            current_sentinel_id = 0
            sentinel_sample = []        
            for idx in range(len(sample)): 
                if sample[idx] == -1:
                    current_sentinel_token = f"<extra_id_{current_sentinel_id}>"
                    sample[idx] = tokenizer(current_sentinel_token)['input_ids'][0]
                    current_sentinel_id += 1
                    if sample[idx + 1] == -1: # Case of a Consecutive span 
                        continue
                    else: 
                        sentinel_sample.append(sample[idx])
                sentinel_samples.append(sentinel_sample)
        
        return sentinel_samples


    def preprocess_function(examples):         
        
        inputs = [ex[source_lang] for ex in examples["translation"]]
        targets = [ex[target_lang] for ex in examples["translation"]]
        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=args.max_source_length, padding=True, truncation=True) #return ['input_ids'] and ['attention_mask'] for the passed sequence
        labels = tokenizer(targets, max_length=args.max_target_length, padding=True, truncation=True)
        labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]
        model_inputs['labels'] = labels['input_ids']
        # print(model_inputs["input_ids"][:1])

        if args.continued_pretraining: 
            """ mT5's training objective involves corrupting randomly chosen spans of tokenized text - replaced by a set of unique sentinel tokens. """
            unmasked_inputs = model_inputs['input_ids']
            masked_inputs, masked_labels = [], []
            for unmasked_input in unmasked_inputs:
                masking_indices = random.sample(range(0, len(unmasked_input)), int(len(unmasked_input)*0.15)) #Corrupting only 15% of the the tokens
                masking_indices.sort()
                masked_input, masked_label = unmasked_input, [1]*len(unmasked_input)

                for idx in range(len(masked_input) - 1): 
                    if idx in masking_indices:
                        masked_label[idx] = masked_input[idx]
                        masked_input[idx] = -1 
                    else: 
                        masked_label[idx] = -1 

                masked_inputs.append(masked_input)
                masked_labels.append(masked_label)

            sentinel_inputs = convert_to_sentinel(masked_inputs)
            # sentinel_labels = convert_to_sentinel(masked_labels)

            # model_inputs['input_ids'] = sentinel_inputs
            # model_inputs['labels'] = sentinel_labels

        return model_inputs

    train_dataset = raw_datasets["train"]
    eval_dataset = raw_datasets["validation"]
    predict_dataset = raw_datasets["test"]

    if training_args.do_train:
        if args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(args.max_train_samples))
        train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        desc="Running tokenizer on train dataset",
        )
    if training_args.do_eval:
        eval_dataset = eval_dataset.map(
            preprocess_function,
            batched=True,
            desc="Running tokenizer on validation dataset",
        )
    if training_args.do_predict:
        predict_dataset = predict_dataset.map(
                preprocess_function,
                batched=True,
                desc="Running tokenizer on prediction dataset",
        )
    
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id = -100)
    metric = load_metric("sacrebleu")
    
    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]
        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        predictions = [pred.strip() for pred in decoded_preds]
        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        output_prediction_file = os.path.join(training_args.output_dir, f'{result["score"]}_generated_predictions.txt')
       
        with open(output_prediction_file, "w", encoding="utf-8") as writer:
            writer.write("\n".join(predictions))
        result = {"bleu": result["score"]}
        return result

    set_seed(training_args.seed)
    trainer = Seq2SeqTrainer(
        model=model, 
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None
    )

# args.earlier_checkpoint
    if training_args.do_train: 
        train_result = trainer.train()
        trainer.save_model()  # Saves the tokenizer too for easy upload
        
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    if training_args.do_eval: 

        results = {}
        max_length, num_beams = args.max_target_length, 1

        metrics = trainer.evaluate(max_length=max_length, num_beams=num_beams, metric_key_prefix="eval")
        max_eval_samples =  len(eval_dataset)
        metrics["eval_samples"] = len(eval_dataset)

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:

        predict_results = trainer.predict(
            predict_dataset, metric_key_prefix="predict", max_length=max_length, num_beams=num_beams
        )
        metrics = predict_results.metrics
        metrics["predict_samples"] = len(predict_dataset)
        predictions = tokenizer.batch_decode(
                    predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        return results

if __name__ == "__main__": 
    main()