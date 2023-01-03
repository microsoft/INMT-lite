
<h1> <p align="center"> INMT-Lite</h1>
Interactive Neural Machine Translation-lite (INMT-Lite) is an assistive translation service that can be run on embedded devices like mobile phones and tablets that have low computation power, space and no internet connectivity. 

Table of Contents
=================
  * [Data](Data)
  * [Models](Models)
  * [Tranformer-Suite](Tranformer-Suite)
  * [RNN-Suite](RNN-Dev)

  
## Data
### Hindi-Gondi Parallel Corpus
INMT was developed to help expand the digital datasets for low resource languages and further support in developing other language tools for such low resource languages. The [sample model](https://microsoftapc-my.sharepoint.com/:f:/g/personal/taganu_microsoft_com/El5qZjb_teZJqsHyE5Cyh1kB2QCTMwot8E7wGYpWCi0BQA?e=zEo7bq) in this codebase is trained on  the first-ever Hindi-Gondi Parallel Corpus released by [CGNet Swara](http://cgnetswara.org/) which can be found [here](http://cgnetswara.org/hindi-gondi-corpus.html). 

## Models 

You can access all our transformer-arch based models here; 

- Gondi MT5               [Non-Compressed Model variant of 2.28GB]()
- Gondi Quantized Model   [Compressed to 400MB]()
- Gondi Distilled Model   [Compressed to 183MB]() 

## Transformer-Suite
This section delineates the instructions for Transformer Dev-Variants: For Model Setup, Training, Inference and Edge Deployment (Preparing the model for Android Compatible Training). Note that code on this repository is heavily adapted for code specified at [this](https://github.com/microsoft/Lightweight-Low-Resource-NMT) repository for generating light-weight, NMT Models. 

#### Environment Information 
The environment can be setup using the provided requirements file
```
pip install -r requirements.txt
```

#### Training Procedure (Generic/Not Compatible with the Android Deployment Pipeline)
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
#### Sample Commands 
```
python spm_extractor.py --provider sentencepiece --model spiece_cli_hi.model --vocab-output-path target_extracted_vocab.json --merges-output-path merges.txt
```
```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node 2 --master_addr 127.0.0.1 --master_port 29502 train.py --vocab_path /home/t-hdiddee/INMT-Tflite/scripts/concatenated_hi_gondi_vocab.json --do_train --do_eval --src_lang hi --tgt_lang gondi --train_file /home/t-hdiddee/INMT-Lite/data/distillation/train_distillation_hi_gondi.json --validation_file /home/t-hdiddee/INMT-Lite/data/distillation/val_distillation_hi_gondi.json --output_dir /home/t-hdiddee/INMT-Lite/final_models/Dm/marian_hi_gondi_distilled/ --per_device_train_batch 256 --per_device_eval_batch 256 --evaluation_strategy steps --eval_steps 400 --logging_strategy steps --marian_tokenizer True --source_spm /home/t-hdiddee/INMT-Tflite/scripts/marian/hi-gondi/spiece_test_hi.model --target_spm /home/t-hdiddee/INMT-Tflite/scripts/marian/hi-gondi/spiece_test_gondi.model --logging_steps 400 --learning_rate 5e-5 --num_train_epochs 60 --gradient_accumulation 2 --load_best_model_at_end True --metric_for_best_model bleu --save_strategy steps --save_steps 800 --save_total_limit 2 --overwrite_output_dir --warmup_steps 500 --do_predict True --eval_accumulation_steps 2 --test_file /home/t-hdiddee/INMT-Lite/data/distillation/val_distillation_hi_gondi.json --predict_with_generate True 

```

#### Evaluation Signature: chrF 
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
#### Directory Structure
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
├── LICENSE
├── README.md
├── SECURITY.md
└── train.py                             # Training script (Supports Continued Pretraining of mt5, Marian Tokenizer Training)

```

## RNN-Dev Usage Instructions 
```
Directory Structure: 
├── RNN
│   ├── Android/
│   ├── preprocess.py
│   ├── train.py
│   ├── translate.py
│   └── utils
│       ├── Model_architectures.py
│       └── Model_types.py
└── requirements.txt
```

All the source code and the executable files of the project are present in the `INMT-lite` sub-folder.

Here is more information about all the components that are present in `INMT-lite`:
 - **Android/** - Code and necessary files for building Android app. This can be directly imported into Android Studio.
 - **preprocess.py** - Code for preprocessing the input data for the models.
 - **train.py** - Code for training the models.
 - **translate.py** - Code for performing inference/testing on the trained models.
 - **utils/Model_architectures.py** - Code for defining the architecture of the Encoder and the Decoder blocks.
 - **utils/Model_types.py** - Code for building specific models for translation and partial mode.

Apart from the above, the project creates additional intermediate folders during the course of its run to store model snapshots, optimized tflite model etc. 

Such cases are clearly mentioned at relevant places in the documentation.

## Requirements

Create a separate environment and install the necessary packages using the following command in the root path:

```
pip install -r requirements.txt
```

Please use pip to install the required packages.

## Quickstart

The following section contains the command to quickly get started with the training and generate the tflite file. For full set of features and options, please look into the Documentation section.

### Step 1: Getting your dataset ready

The dataset should be a set of four files one pair for training and one for validation. Both training dataset and validation dataset should consist of two files each, one file consisting of source language sentences separated by an new line and tokens separated by a space and another file for target language in similar format. 

Please make sure your dataset is according to the above mentioned format before heading to further steps.
 
### Step 2: Preprocessing the data

We will be using the example dataset stored in `data/` folder. While inside the root of the project, type in:

```
python preprocess.py --path_src_train ./data/src-train.txt --path_tgt_train ./data/tgt-train.txt  --path_src_val ./data/src-val.txt --path_tgt_val ./data/tgt-val.txt
```

After the command is successfully executed, a set of six files is generated inside `data/` folder:
* `src_train_processed.txt`: Text file consisting tokenized and processed text for `src-train.txt` file.
* `tgt_train_processed.txt`: Text file consisting tokenized and processed text for `tgt-train.txt` file.
* `src_val_processed.txt`: Text file consisting tokenized and processed text for `src-val.txt` file.
* `tgt_val_processed.txt`: Text file consisting tokenized and processed text for `tgt-val.txt` file.
* `src_vocab.json`:  Consists of source language vocabulary in JSON format.
* `tgt_vocab.json`:  Consists of source language vocabulary in JSON format.

### Step 3: Training the model

To train the model on the processed dataset, run the following command in the root of the project:

```
python train.py --to_path_tgt_train ./data/tgt_train_processed.txt --to_path_src_train ./data/src_train_processed.txt --to_path_src_val ./data/src_val_processed.txt --to_path_tgt_val ./data/tgt_val_processed.txt --to_path_src_vocab ./data/src_vocab.json --to_path_tgt_vocab ./data/tgt_vocab.json
```

The command just takes the processed dataset files and json files for vocabulary of both the languages. The model has a encoder decoder architecture and uses attention. The model uses GRU with 500 units on both the encoder and decoder side and would run for 10 epochs. The model configuration such as the number of hidden units can be changed. For more information, please look into detailed documentation.

This command will generate a model config file under `model/` directory as `model/model_config.json` and a tflite file under the `tflite/` directory as `tflite/tflite_model.tflite`. Also, after every 2 epochs a checkpoint file would be saved under `model/` folder. This setting also can be changed.

By default, GPU hardware acceleration would not be used. To use the GPU, please pass in the argument: `--use_gpu true`. 

### Step 4: Testing, Inference and further

Both testing and inference are handled by `translate.py` file. It supports three modes testing, inference and inline.

#### Testing

For testing the model's performance please use the following command while in root of the project:

```
python translate.py --src_path data/src-val.txt --tgt_path data/tgt-val.txt --src_vocab_path data/src_vocab.json --tgt_vocab_path data/tgt_vocab.json --model_path model/model_weight_epoch_10.h5 --model_config_path ./model/model_config.json --mode test​​​​​​
```

After the successful run, a line would be printed reporting the test score for the model on the given dataset. The test score can be interpreted as the average words predicted correctly by the model.

The command takes in the following inputs - two text files (txt files consisting of sentences for each source and target language), a model_config file consisting information about the model, trained model's path, JSON files consisting of vocabulary of the languages and mode type for the command.

#### Inference

For getting the translations of a set of sequences written in a txt file. Use the following command while inside the root of the project.

```
python translate.py --src_path data/src-val.txt --tgt_path data/ --src_vocab_path ./data/src_vocab.json --tgt_vocab_path ./data/tgt_vocab.json --model_path ./model/model_weight_epoch_10.h5 --model_config_path ./model/model_config.json --mode inference
```

After the successful run a txt file consisting of translated sequences would be generated under `data/` folder saved as `data/inference.txt`

The command takes in the following inputs - a txt file consisting of sequences in source language, a target directory path where the inference file should be saved, the vocabulary paths of both the languages, trained model's path, a model configuration path and mode type for the command.

#### Inline

The framework also supports translating a sequence in command line itself for quick inference. For inline inference, type in the following command while in root of the project:

```
python translate.py --sentence <SENTENCE_TO_TRANSLATE> --src_vocab_path ./data/src_vocab.json --tgt_vocab_path ./data/tgt_vocab.json --model_path ./model/model_weight_epoch_10.h5 --model_config_path ./model/model_config.json --mode inline
``` 

After the successful run,  the translated sentence would be printed in the terminal itself.

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
