### Usage Instructions: For Model Setup, Training, Inference and Edge Deployment (Preparing the model for Android Compatible Training) 

## Environment Information 
The environment can be setup using the provided requirements file
```
pip install -r requirements.txt
```

## Training Procedure (Generic/Not Compatible with the Android Deployment Pipeline)
```
1. Run **preprocess.py** to convert training data to HF format and generating the Tokenizer Files for the Vanilla tranformer. 
2. Run **train.py** for training and saving the best model. (monitored metric is BLEU with mt13eval tokenizer)
3. Run **split_saving_{model_architecture_type}.py** to quantize the encoder and decoder separately. 
4. Run **inference.py** (with offline = True) for offline inference on the quantized graphs.  
```
Note that for making the model Android-Compatible: We use an entirely different tokenization procedure - specified as: 
```
1. Run final_tokenizer_train.py - This create the spm models that will be used for tokenization
2. Run spm_extractor.py - This is used to create the vocab files ( required by the Hugging Face interface ) from the serialized models
3. Run make_vocab_from_extracted_files.py - To generate a concated_vocab that will be used to instantiate the tokenizer. Make sure to edit the start_idx to match the len of your source_vocab.
4. Run train.py with the required arguments (marian_tokenizer to True, and provide spm models) to start the training.
```
### Sample Commands 
```
python spm_extractor.py --provider sentencepiece --model spiece_cli_hi.model --vocab-output-path target_extracted_vocab.json --merges-output-path merges.txt
```
```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node 2 --master_addr 127.0.0.1 --master_port 29502 train.py --vocab_path /home/t-hdiddee/INMT-Tflite/scripts/concatenated_hi_gondi_vocab.json --do_train --do_eval --src_lang hi --tgt_lang gondi --train_file /home/t-hdiddee/INMT-Lite/data/distillation/train_distillation_hi_gondi.json --validation_file /home/t-hdiddee/INMT-Lite/data/distillation/val_distillation_hi_gondi.json --output_dir /home/t-hdiddee/INMT-Lite/final_models/Dm/marian_hi_gondi_distilled/ --per_device_train_batch 256 --per_device_eval_batch 256 --evaluation_strategy steps --eval_steps 400 --logging_strategy steps --marian_tokenizer True --source_spm /home/t-hdiddee/INMT-Tflite/scripts/marian/hi-gondi/spiece_test_hi.model --target_spm /home/t-hdiddee/INMT-Tflite/scripts/marian/hi-gondi/spiece_test_gondi.model --logging_steps 400 --learning_rate 5e-5 --num_train_epochs 60 --gradient_accumulation 2 --load_best_model_at_end True --metric_for_best_model bleu --save_strategy steps --save_steps 800 --save_total_limit 2 --overwrite_output_dir --warmup_steps 500 --do_predict True --eval_accumulation_steps 2 --test_file /home/t-hdiddee/INMT-Lite/data/distillation/val_distillation_hi_gondi.json --predict_with_generate True 

```

## Evaluation Signature: chrF 
```
{
 "name": "chrF2",
 "signature": "nrefs:1|case:mixed|eff:yes|nc:6|nw:0|space:no|version:2.2.0",
 "nrefs": "1",
 "case": "mixed",
 "eff": "yes",
 "nc": "6",
 "nw": "0",
 "space": "no",
 "version": "2.2.0"
}

{
 "name": "BLEU",
 "signature": "nrefs:1|case:mixed|eff:no|tok:spm-flores|smooth:exp|version:2.2.0",
 "verbose_score": "43.2/27.2/19.4/13.7 (BP = 1.000 ratio = 1.154 hyp_len = 81271 ref_len = 70443)",
 "nrefs": "1",
 "case": "mixed",
 "eff": "no",
 "tok": "spm-flores",
 "smooth": "exp",
 "version": "2.2.0"
}
```


# Directory Structure
```bash 
├── auto                  # Each of these scripts can be used for automating different parts of the pipelines w/ standard hyperparameters 
│   ├── amount_of_synthetic_data_sweeps.sh  # Sweeping across amount of data to train the distilled models for 
│   ├── compute_confidence.sh               # Computing confidence on in-domain and out-of-domain datasets
│   ├── distillation_data_generator.sh      # Generating distillation labels for all the students 
│   ├── inference_seq.sh                    # Inferencing both in online and offline modes
│   ├── student_architecture_sweep.sh       # Sweeping across different student architectures 
│   └── train_seq.sh                        # Training student models across different distillation configurations
├── confidence_estimation.py                # Monitoring online models' logits - Softmax Entropy, Top-K probabilities Dispersion
├── confidence_visualization.ipynb          # Visualization confidence of the models 
├── debug                                   # Debugging scripts 
│   ├── inference_debug.py
│   ├── tfb_inference_debug.py
│   └── train_debug.py
├── inference.py                            # Inferencing - All models [mt5, vanilla], All modes [Online, Quantization](For distillation models go to tflite_inference_distilled_models.py)
├── make_concatenated_vocab.py              # Making the vocab to train the Marian Tokenizer (Used for compatibility with Deployment goal)
├── marian                                  # Marian tokenizer models (Used for compatibility with Deployment goal)
│   ├── en-hi
│   │   ├── concatenated_vocab.json
│   │   ├── source_merges.json
│   │   ├── source_vocab.json
│   │   ├── target_merges.json
│   │   └── target_vocab.json
│   └── hi-gondi
│       ├── merges_gondi.txt
│       ├── merges_hi.txt
│       ├── spiece_test_gondi.model
│       ├── spiece_test_gondi.vocab
│       ├── spiece_test_hi.model
│       └── spiece_test_hi.vocab
├── mt5_inference.py                    # mt5-specific inferencing script 
├── preprocess.py                       # Creates train/test files in the format that is required by the dataloader + tokenizer training 
├── requirements.txt                    # package requirements for these scripts      
├── sc.py                               # For script converting before and after training for languages with unseen scripts 
├── split_saving_mt5.py                 # Converting finetuned mt5 models to offline graphs (split into encoder and decoder)
├── split_saving_tfb.py                 # Converting trained vanilla transformer models to offline graphs (split into encoder and decoder)
├── spm_extractor.py                    # Used to extract vocab/merges from the spm models (Used for compatibility with Deployment goal)
├── spm_model_generator.py              # Generating the spm models for the Marian Tokenizer (Used for compatibility with Deployment goal)
├── student_labels.py                   # Generates distillation labels in batches using source-lang monolingual data
├── supplementary                       # For supplementary checks 
│   ├── check_dedup.py                  # Checking for Train/Test duplication
│   ├── complete_model_tflite_conversion.py
│   ├── tf_model.tflite                 # Converting tf models (includes models trained with the Huggingface API) to tflite. 
│   └── tfb_inference.py                
├── sweep.yaml                           # Yaml configuration file for running sweeps on Wandb
├── tflite_inference_distilled_models.py # Sequential inferencing with the vanilla transformer models 
└── train.py                             # Training script (Supports Continued Pretraining of mt5, Marian Tokenizer Training)
```


#### Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

#### Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
