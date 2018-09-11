# Bias-from-rationale-analysis
Identifying bias by using rationale model

## Code
### Baseline model

The baseline model is used for calssification. It is a standard model constructed by several layers of Bi-LSTM by using allennlp. The input is 300-dimensional word embeddings from GloVe-300.To use the model, setup allennlp first. 

 Check `./lstm_experiments.py` for changing:
- Data path
- Output path
- Training or testing

 And check `./training_config/lstm.json` for changing:
- Class number
- Input/output dimension
- Dropout
- Batchsize
- Training config: epoch/ patience/ CUDA device/ optimizer type and learning rate


### Rationale model

The model is an Pytorch version implimentation of the papaer ```Rationalizing Neural Predictions``` by Tao Lei.
The main implimentation of the code is finished by [Yala/text_nn](https://github.com/yala/text_nn "悬停显示").

#### Requirements
The input is 300-dimensional word embedding from GloVe-300.

#### Usage
