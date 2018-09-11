import  pickle
import  torch
import os
import sys
import io
from rationale_net.datasets.gender_sentiment import AbstractDataset
from pandas import read_csv
import operator

import torch._utils
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')

#in_file = 'E:\\NLP\\dataset\\long_gender\\sentiment\\sentiment_female_1_rationale.txt'
in_file_1 = 'E:\\NLP\\dataset\\yelp_culture\\culture_1_r.txt'
in_file_2 = 'E:\\NLP\\dataset\\long_gender\\sentiment\\sentiment_male_1_rationale.txt'
#out_file = 'E:\\NLP\\dataset\\long_gender\\sentiment\\sentiment_female_1_rationale_count_valuesequence.txt'
out_file = 'E:\\NLP\\dataset\\yelp_culture\\culture_1_r_wo.txt'

def count_word_keyseq(in_file,out_file):
    word_count={}#统计词频的字典
    total = 0
    for line in open(in_file):
        words = line.strip().split(" ")
        for word in words:
            if word in word_count:
                word_count[word]+=1
            else:
                word_count[word]=1
    #out = open(out_file,'w')
    for word in sorted(word_count.keys()):#按单词的顺序遍历字典的每个元素
        total = total + word_count[word]
        #print(total)
    total = total - word_count['_']
    print(total)
        #out.write('%s:%d' % (word, word_count.get(word)))
        #out.write('\n')
    #out.close()

def count_word_valueseq(in_file,out_file):
    word_count={}#统计词频的字典
    for line in open(in_file):
        words = line.strip().split(" ")
        for word in words:
            if word in word_count:
                word_count[word]+=1
            else:
                word_count[word]=1
    out = open(out_file,'w')
    sorted(word_count.items(), key=lambda item:item[1])
    for word in word_count.keys():#按单词的顺序遍历字典的每个元素
        if word_count[word] > 0:
            print(word,word_count[word])
            out.write('%s:%d' % (word, word_count.get(word)))
            out.write('\n')
    out.close()

def count_word_valueseq_two(in_file_1,in_file_2,out_file):
    word_count_1={}#统计词频的字典
    word_count_2={}
    word_count={}
    for line in open(in_file_1):
        words_1 = line.strip().split(" ")
        for word_1 in words_1:
            if word_1 in word_count_1:
                word_count_1[word_1]+=1
            else:
                word_count_1[word_1]=1
    for line in open(in_file_2):
        words_2 = line.strip().split(" ")
        for word_2 in words_2:
            if word_2 in word_count_2:
                word_count_2[word_2]+=1
            else:
                word_count_2[word_2]=1
    out = open(out_file,'w')

    for key in word_count_1:
        if word_count_2.get(key):
            word_count[key] = word_count_1[key] - word_count_2[key]
        else:
            word_count[key] = word_count_1[key]
    for key in word_count_2:
        if word_count_1.get(key):
            pass
        else:
            word_count[key] = word_count_2[key]

    sorted(word_count.items(), key=lambda item:item[1])
    for word in word_count.keys():#按单词的顺序遍历字典的每个元素
        print(word,word_count[word])
        out.write('%s:%d' % (word, word_count.get(word)))
        out.write('\n')
    out.close()

#count_word_valueseq_two(in_file_1, in_file_2,out_file)
count_word_valueseq(in_file_1,out_file)
