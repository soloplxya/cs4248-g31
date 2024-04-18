# Do You Truly Understand Your Language Models? Dissecting Translation Through Model Alterations 🤔🤔🤔
*CS4248 Group 31, mentored by Rishabh Anand*


This repository contains the source code for our group project from the natural language processing module CS4248. This markdown file contains a summary of the research that our group has done and a brief outline of the structure of this github repository. 

## About our project
Our project aims to build upon existing research by systematically exploring enhancements to translation models for English-to-Chinese tasks. We draw inspiration from seminal works in machine translation, which have proposed various approaches for improving translation quality. 

## Our Team
Our team consists of the following members: 
- [Shaine Goh Si Hui](https://www.github.com/soloplxya) (A0220084U)
- [Thia Su Mian](https://www.github.com/tsumian) (A0214460R)
- [Lin Hui Xin Tiffany](https://www.github.com/Tiffanylin21) (A0223682E)
- [Ng Xing Yu](https://www.github.com/ngxingyu) (A0234386Y)
- [Guo Bokun](https://www.github.com/bokung) (A0239830B)
- [Wang Yi Fan](https://www.github.com/pudding317) (A0275632B)

## Models 
We implemented two standard architectures commonly used in machine translation: a recurrent neural network (RNN) model and a transformer model. An outline of the key hyperparameters and model architecture is listed below:
- Baseline Recurent Neural Network model
  - Architecture adapted from [Chris2024](https://github.com/chrisvdweth/nus-cs4248x/blob/master/3-neural-nlp/Section%203.2%20-%20RNN%20Machine%20Translation.ipynb)
  - Sequence-to-sequence architecture 
  - Encoder-Decoder Framework
  - RNN/GRU/LSTM Encoder and Decoder
  - Cross-Entropy Loss / Incomplete-Trust Loss
  - Fixed or Variable teacher forcing probability
- Baseline Transformer model 
  - Architecture adapted from [Chris2024](https://github.com/chrisvdweth/nus-cs4248x/blob/master/3-neural-nlp/Section%204.2%20-%20Transformer%20Machine%20Translation.ipynb)
  - SPM BytePair Tokeniser
  - Cross-Entropy Loss / Incomplete-Trust Loss
  - trained with 32 epochs 
## Experimentation Techniques
The key experimentation techniques modified in this research project can be broken down into the following components: 
- Tokenization methods
    -  Byte-Pair Encoding 
    - Unigram Tokenization
- Encoding Methods
    - Relative Positional Encoding 
    - Absolute Positional Encoding
- Decoding Methods 
    - Greedy Decode
    - Beam Search 
- Loss Tabulation
    - Cross Entropy Loss Function
    - Incomplete Trust (In-Trust) Loss Function

## Evaluation Metrics
Due to the nature of the research, two key evaluation metrics were utilized in assessing the performance of our models. 
- COMET 
- BLEURT

Additionally, cutting edge XAI (Explainable AI) visualization tools were also utilized in our evaluation. These include: 
- Attention HeatMaps
- BertViz

## Results 
Our results show that a transformer model utilising relative positional encodings, unigram tokenization, and in-trust loss function generates the highest `COMET` and `BLEURT` scores. 
> More details about our project can be found in the pdf here: [`CS4248_Group31_Final_Report.pdf`](https://www.google.com)!!

## Structure of the Repository
This section provides an overview of the organization of the repository, outlining the purpose of each directory and highlighting important files. 

```
|-rnn/
  - README.md
  - requirements.txt
  - rnn.py
  - data.py
  - eval.py
  - rnn_model.py
  - rnn_inference.py
  - Some other utility scripts used


|-transformer/
  |- models/
  - README.md
  - all_sentences.txt
  - spm_beam_search_model.model
  - spm_beam_search_model.vocab
  - transformer-attention-heatmaps-beamsearch-decoder.ipynb
  - transformer-base-spacy-tokenizer.ipynb
  - transformer-intrust-loss.ipynb
  - transformer-relativePE.ipynb
  - transformer-spm-unigram-tokenisation.ipynb
  - transformer-visualization.ipynb
```

- `rnn/` contains all scripts for our RNN model
- `transformer/` contains all notebooks for our transformer model
- `transformer/model` contains some sample loaded models 
 

## Running Instructions

> [!IMPORTANT]
> Please ensure that you are using Python runtime version `3.10.12` when running the notebooks/python scripts locally. You can adjust this configuration through Anaconda or downloading the relevant packages on the python website.

### RNN
- To run the RNN models, follow the instructions in the [rnn/README.md](./rnn/README.md) file.
- You will require some python 3 environment with pytorch installed.

### Notebooks
Most of the code provided can be found in the form of python notebook `.ipynb` files. 

There are two ways to run python notebooks:


#### Jupyter Notebook (Locally)
> Installation: If you haven't already installed Jupyter Notebook or JupyterLab, you can do so using pip (Python's package installer). Open your terminal or command prompt and run:
```pip install notebook``` 

Once installed, you can use the command `jupyter notebook` to launch the notebook.

As our programs are quite GPU intensive due to intensive deep learning training, it is recommended that your local GPU have specs similar to that of the `NVIDIA V100 GPU`.


#### Google Colab
The set-up for colab is much simpler. You simply need to upload the notebooks onto your drive. Make sure relevant dependencies (for e.g. models to be loaded, pre-training data) is self-contained within the same directory before running any programs. 

> Each notebook contains it own set of specific instructions so it would be good to follow according to the instructions given within each notebook. 

For higher computing power, more GPU can be purchased either through a `pay-as-you-go` scheme or `Colab Pro`


## Acknowledgements
- We would like to thank our mentor, Rishabh Anand, for his guidance and support throughout the project. We would also like to thank the teaching team for their valuable feedback and suggestions.

## References
- [Chris2024](https://github.com/chrisvdweth/nus-cs4248x)
- [HuggingFace](https://huggingface.co/transformers/)
- [COMET](https://github.com/Unbabel/COMET)
- [BLEURT](https://github.com/google-research/bleurt)
- [BertViz](https://github.com/jessevig/bertviz)
- [sentencepiece](https://github.com/google/sentencepiece)
- [spacy](https://spacy.io/)
- [torchtext](https://pytorch.org/text/stable/index.html)
- [torch](https://pytorch.org/)
- [lightning](https://www.pytorchlightning.ai/)
- [datasets](https://huggingface.co/docs/datasets/)
- [Low-resource-text-translation](https://github.com/WENGSYX/Low-resource-text-translation)

## Issues 
Feel free to contact anyone of us if you notice any issues in our code! You can do this through creating an issue in github. Refer to this [link](https://docs.github.com/en/issues/tracking-your-work-with-issues/creating-an-issue) for more details on how to do so.
