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

The model has a generator and an encoder:
- Generator: picking up rationale, the key words. Having two parameters, selection_lambda and continuity_lambda.
- Encoder: a text-CNN model for classification only on picked rationales.

#### Requirements
The input is 300-dimensional word embedding from GloVe-300.

#### Usage
##### Running example:

```
CUDA_VISIBLE_DEVICES=0 python -u scripts/main.py  --batch_size 64 --cuda --dataset news_group --embedding glove --dropout 0.05 
--weight_decay 5e-08 --num_layers 1 --model_form cnn --hidden_dim 100 --epochs 50 --init_lr 0.001 --num_workers 0 --objective
cross_entropy --patience 10 --save_dir snapshot --train --test --results_path logs/demo_run.pkl  --gumbel_decay 1e-5 
--get_rationales --selection_lambda .001 --continuity_lambda 0
```
You should change:
- result_path: path and name of the file
- dataset: the data you use, register it before use it.
- selection_λ: 

The model is sensitive, and recommended λ:
- 0: pick the whole sentence as rationale and do prediction on it, a kind of baseline as text-CNN.
- 0.01
- 0.005
- 0.001
- 0.0005
- 0.0001

##### Checking results:

Using `./analysis/analysis1.py` to get rationales. You can get:
- text_golds/ text_preds
- original text
- rationale

In case of some characters cannot be coding, using `.replace('/xxx','')` if there is error-reporting.

Using `./analysis/frequency.py` to get rationale occurence. There are several functions:
- count_word_keyseq: count word and print in keys sequence.
- count_word_valueseq: count word with and print in the values sequence.
- count_word_valueseq_two: count rationale difference of two groups and print in the value sequence.


##### Adding new dataset:
- Add a pytorch Dataset object to `./rationale_net/datasets` and register it to the dataset factory. See the news_group, beer_review, gender, gender_sentiment datasets for an example.
- Add the corresponding import to `./rationale_net/datasets/__init__.py`
- Take `./rationale_net/datasets/gender.py` as an example, you can change datapath, train/val/test data amount, class number, maxlength of a sentence.

