
# INMT-lite

Interactive Neural Machine Translation-lite (INMT-lite) is a framework to train and develop lite versions (.tflite) of models for neural machine translation (NMT) that can be run on embedded devices like mobile phones and tablets that have low computation power and space. The tflite models generated can be used to build the offline version of INMT mobile, a mobile version of INMT web.



Table of Contents
=================
  * [Requirements](#requirements)
  * [Quickstart](#quickstart)
  * [INMT-mobile](#INMT-Mobile)
   * [Acknowledgements](#acknowledgements)
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
We will be using the example dataset stored in `example_data/` folder. While inside the root of the project, type in:
```
python preprocess.py --path_src_train ./example_data/src-train.txt --path_tgt_train ./example_data/tgt-train.txt  --path_src_val ./example_data/src-val.txt --path_tgt_val ./example_data/tgt-val.txt
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

By default, GPU hardware acceleration would not be used. To use the GPU pass in the argument: `--use_gpu true`. 
### Step 4: Testing, Inference and further
Both testing and inference are handled by `translate.py` file. It supports three modes testing, inference and inline.

#### Testing
For testing the model's performance please use the following command while in root of the project:
```
python translate.py --src_path example_data/src-val.txt --tgt_path example_data/tgt-val.txt --src_vocab_path example_data/src_vocab.json --tgt_vocab_path example_data/tgt_vocab.json --model_path model/model_weight_epoch_10.h5 --model_config_path ./model/model_config.json --mode test​​​​​​
```
After the successful run, a line would be printed reporting the test score for the model on the given dataset. The test score can be interpreted as the average words predicted correctly by the model.

The command takes in two text files (txt files consisting of sentences for each source and target language), a model_config file consisting information about the model, trained model's path, JSON files consisting of vocabulary of the languages and mode type for the command.



#### Inference
For getting the translations of a set of sequences written in a txt file. Use the following command while inside the root of the project.

```
python translate.py --src_path example_data/src-val.txt --tgt_path example_data/ --src_vocab_path ./data/src_vocab.json --tgt_vocab_path ./data/tgt_vocab.json --model_path ./model/model_weight_epoch_10.h5 --model_config_path ./model/model_config.json --mode inference
```
After the successful run a txt file consisting of translated sequences would be generated under `data/` folder saved as `data/inference.txt`

The command takes in the txt file consisting of sequences in source language, a target directory path where the inference file should be saved, the vocabulary paths of both the languages, trained model's path, a model configuration path and mode type for the command.

#### Inline
The framework also supports translating a sequence in command line itself for quick inference. For inline inference, type in the following command while in root of the project:
```
python translate.py --sentence <SENTENCE_TO_TRANSLATE> --src_vocab_path ./data/src_vocab.json --tgt_vocab_path ./data/tgt_vocab.json --model_path ./model/model_weight_epoch_10.h5 --model_config_path ./model/model_config.json --mode inline
``` 
After the successful run,  the translated sentence would be printed in the terminal itself.

Please note that the inference in all modes does not consist of `<start>` token.
## INMT Mobile

The following section is about Interactive Neural Machine Translation (INMT) mobile, a mobile version of INMT-web. This app consists of both the offline and online mode. The offline mode uses the tflite file generated by INMT-tflite whereas the online mode uses INMT-web REST APIs to bring in the results.

### Online Mode
The Online mode of INMT mobile calls the APIs of INMT web for the partial translation provided by the user as he types. This mode supports multiple language pair translations that are supported by INMT-web and can be easily extended for new language pairs.
### Offline Mode
The Offline mode uses the tflite model as generated by tflite framework. The model when generated by the default settings is usually of the size of aroung 60 MB. The translation predictions are not as accurate as INMT-web because of the portability. However, INMT mobile offline mode doesn't need internet connection and API calls for translation predictions.




# Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.


## Acknowledgements
INMT-lite is a collaborative project developed in Microsoft Research.
Major contributors are:

 - Anurag Shukla
 - Tanuja Ganu
 - Kalika Bali
 - Sebastin Santy
 - Monojit Choudhury
 - Sandipan Dandapat
