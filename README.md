<h1> <p align="center"> INMT-Lite </p></h1> 
Interactive Neural Machine Translation-lite (INMT-Lite) is an assistive translation service that can be run on embedded devices like mobile phones and tablets that have low computation power, space and no internet connectivity. A detailed background of the compression techniques used to drive the assistive interfaces, the model's data and evaluation and the interface design can be found at the linked works. 

<h7> <p align="center"> <a href="https://arxiv.org/abs/2211.16172"> <i> <b> Collecting Data through community-oriented channels </b> in under-resourced communities </i> </a> 
<h7> <p align="center"> <a href="https://arxiv.org/abs/2210.15184"> <em> <b> Compression of Massively Multilingual Translation Models </b> for Offline Operation </em> </a> 

<h7> <p align="center"> <a href=""> <em> Assistive Interfaces for Enhancing and Evaluating Data Collection <b> (Coming Soon!) </b> </em> </a> 


Table of Contents
=================


  * [Data](#data)
  * [Models](#models)
  * [Transformer-Suite](#transformer-suite)
  * [RNN-Suite](#rnn-suite)

  
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
#### Directory Structure
```bash 
├── confidence_estimation.py                # Monitoring online models' logits - Softmax Entropy, Top-K probabilities Dispersion
├── confidence_visualization.ipynb          # Visualization confidence of the models 
├── inference.py                            # Inferencing - All models [mt5, vanilla](For distillation models go to tflite_inference_distilled_models.py)
├── make_concatenated_vocab.py              # Making the vocab to train the Marian Tokenizer (Used for compatibility with Deployment goal)
├── mt5_inference.py                        # mt5-specific inferencing script 
├── preprocess.py                           # Creates train/test files in the format that is required by the dataloader + tokenizer training 
├── requirements.txt                        # package requirements for these scripts      
├── sc.py                                   # For script converting before and after training for languages with unseen scripts 
├── split_saving_mt5.py                     # Converting finetuned mt5 models to offline graphs (split into encoder and decoder)
├── split_saving_tfb.py                     # Converting trained vanilla transformer models to offline graphs (split into encoder and decoder)
├── spm_extractor.py                        # Used to extract vocab/merges from the spm models (Used for compatibility with Deployment goal)
├── spm_model_generator.py                  # Generating the spm models for the Marian Tokenizer (Used for compatibility with Deployment goal)
├── student_labels.py                       # Generates distillation labels in batches using source-lang monolingual data
├── sweep.yaml                              # Yaml configuration file for running sweeps on Wandb
├── tflite_inference_distilled_models.py    # Sequential inferencing with the vanilla transformer models 
├── marian                                  # Marian tokenizer models (Used for compatibility with Deployment goal, Example files are provided as output ref)
│   └── hi-gondi
│       ├── merges_gondi.txt
│       ├── merges_hi.txt
│       ├── spiece_test_gondi.model
│       ├── spiece_test_gondi.vocab
│       ├── spiece_test_hi.model
│       └── spiece_test_hi.vocab 
├── LICENSE
├── README.md
├── SECURITY.md
└── train.py                                # Training script (Supports Continued Pretraining of mt5, Marian Tokenizer Training)

```
## RNN-Suite
```
Directory Structure: 
├── RNN-Suite
│   ├── preprocess.py
│   ├── preprocess.py
│   ├── train.py
│   ├── translate.py
│   └── utils
│       ├── Model_architectures.py
│       └── Model_types.py
└── requirements.txt
```
#### Environment Information
Create a separate environment and install the necessary packages using the following command in the root path:

```
pip install -r requirements.txt
```

#### Directory Structure 
 - **preprocess.py** - Code for preprocessing the input data for the models.
 - **train.py** - Code for training the models.
 - **translate.py** - Code for performing inference/testing on the trained models.
 - **utils/Model_architectures.py** - Code for defining the architecture of the Encoder and the Decoder blocks.
 - **utils/Model_types.py** - Code for building specific models for translation and partial mode.

#### Training Procedure
Please refer to the readme in RNN root folder. 

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
