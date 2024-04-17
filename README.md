# Do You Truly Understand Your Language Models? Dissecting Translation Through Model Alterations
*CS4248 Group 31, mentored by Rishabh Anand*


This repository contains the source code for our group project from the natural language processing module CS4248. This markdown file contains a summary of the research that our group has done and a brief outline of the structure of this github repository. 

## About our project
Our project aims to build upon existing research by systematically exploring enhancements to translation models for English-to-Chinese tasks. We draw inspiration from seminal works in machine translation, which have proposed various approaches for improving translation quality. 

## Our Team
Our team consists of the following members: 
- [Shaine Goh Si Hui](https://www.github.com/soloplxya) (A0220084U)
- [Thia Su Mian](https://www.github.com/tsumian) (A0214460R)
- [Lin Hui Xin Tiffany](https://www.github.com/Tiffanylin21) (A0223682E)
- [Ng Xing Yu](https://www.github.com/ngxingyu)
- [Guo Bo Kun](https://www.github.com/bokung) 
- [Wang Yi Fan](https://www.github.com/pudding317) 

## Models 
We implemented two standard architectures commonly used in machine translation: a recurrent neural network (RNN) model and a transformer model. An outline of the key hyperparameters and model architecture is listed below:
- Baseline Recurent Neural Network model
  - Sequence-to-sequence architecture 
  - Encoder-Decoder Framework
- Baseline Transformer model 
  - Architecture adapted from [Chris2024](https://github.com/chrisvdweth/nus-cs4248x/blob/master/3-neural-nlp/Section%204.2%20-%20Transformer%20Machine%20Translation.ipynb)
  - SPM BytePair Tokeniser
  - Cross-Entropy Loss 
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
Our results show that a transformer model utilising relative positional encodings, unigram tokenization, an in-trust loss function generates the highest `COMET` and `BLEURT` scores. 
More details can be found in our paper [here]().

## Structure of the Repository
To be added 



