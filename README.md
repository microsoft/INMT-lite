# Usage Instructions: For Model Training, Inference and Edge Deployment
#### Directory Structure
```
├── readme.md
├── requirements.txt
├── scripts                            # Scripts with all the variants of the commands + default hyperparameter values
│   ├── confidence_estimation.sh       # logging the confidence statistics
│   ├── inference.sh                   # inference for both architectures - online and offline graphs 
│   ├── preprocess.sh                  # preprocessing data for training and evaluation
│   ├── sweep.yaml                     # sweep yaml for hyperparameter trials 
│   └── train.sh                       # variants of training and continued pretraining
└── src                                # src files for all the experiments 
    ├── confidence_estimation.py       # logging confidence stats: average softmax entropy, standard deviation of log probabilities
    ├── continued_pretraining.py       # continued pretraining of mt5
    ├── inference.py                   # online and graph inference
    ├── preprocess.py                  # preprocessing bilingual and monolingual data + vocab and tokenizer creation 
    ├── split_saving.py                # generating the offline graphs for both model architectures 
    ├── student_labels.py              # generating the student labels for the best model architecture for the models   
    ├── train.py                       # training script for vanilla, distilled and pretrained model configuration
    └── utils.py                       # utils like script conversion, checking for deduplication
```

#### Training Procedure 
```
1. Run **preprocess.py** to convert training data to HF format and generating the Tokenizer Files for the Vanilla tranformer. 
2. Run **train.py** for training and saving the best model. (monitored metric is BLEU with mt13eval tokenizer)
3. Run **split_saving_{model_architecture_type}.py** to quantize the encoder and decoder separately. 
4. Run **inference.py** (with offline = True) for offline inference on the quantized graphs.  

Sample commands with default hyperparameter values are specified in scripts/
```


#### Evaluation Signature: BLEU and chrF
```
{
 "nrefs:1|case:mixed|eff:no|tok:spm-flores|smooth:exp|version:2.2.0",
 "verbose_score":,
 "nrefs": "1",
 "case": "mixed",
 "eff": "no",
 "tok": "spm-flores",
 "smooth": "exp",
 "version": "2.2.0"
}
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
