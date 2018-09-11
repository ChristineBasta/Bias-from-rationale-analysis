import  pickle
import  torch
import os
import sys
import io
from rationale_net.datasets.gender_sentiment import AbstractDataset
from pandas import read_csv


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

file = pickle.load(open('E:\\NLP\\dataset\\yelp_culture\\culture1.pkl','rb'))
file = pickle.load(open('E:\\NLP\\dataset\\Mturk_3.pkl','rb'))
text_text = file['test_data']
text_preds = file['test_stats']['preds']
text_golds = file['test_stats']['golds']
rationales = file['test_stats']['rationales']


file1 = open('E:\\NLP\\dataset\\yelp_culture\\culture_1_r.txt','w+')
file1 = open('E:\\NLP\\dataset\\Mturk_3.txt','w+')
for i in range(len(rationales)):
    #a = text_golds[i]-text_preds[i]
    #file1.write(str(a)+'\n')
    file1.write(str(text_golds[i])+' '+str(text_preds[i]) + '\n')
    #file1.write(str(text_golds[i])+'\t'+str(text_preds[i])+'\n')
    #file1.write(text_text[i]['text']+'\n')
    text_text[i]['text'] = text_text[i]['text'].replace('\x9c','').replace('\xc3',' ').replace('\x8d','').replace('\x99','').replace('\xe4','').replace('\xa3','').replace('\xb8','').replace('\x89','').replace('\xa1','').replace('\xa2','').replace('\xbc','').replace('\xaf','').replace('\xb3','').replace('\xad','').replace('\xbb','').replace('\xc3','').replace('\xbd','').replace('\xa9','').replace('\xc2','')+'\n'
    text_text[i]['text'] = text_text[i]['text'].replace('\x8e','').replace('\x95','').replace('\x8b','').replace('\xc5','').replace('\xa5','').replace('\xb9','').replace('\xb5','').replace('\xb6','').replace('\x8c','').replace('\xba','').replace('\x98','').replace('\x88','').replace('\x9a','').replace('\x97','').replace('\xe2','').replace('\xb4','').replace('\x87','').replace('\xac','').replace('\xa6','').replace('\x9f','').replace('\xbf','').replace('\xb2','').replace('\xe7','').replace('\x9e','').replace('\xe6','').replace('\x9d','').replace('\x83','').replace('\x8f','').replace('\xab','').replace('\xe5','').replace('\xab','').replace('\xef','').replace('\xbe','').replace('\x84','').replace('\x96','').replace('\x94','').replace('\xaa','').replace('\xae','').replace('\xc4','').replace('\x81','').replace('\x80','').replace('\xe3','').replace('\x8a','').replace('\x82','').replace('\x86','').replace('\x93','').replace('\x92','').replace('\xf1','').replace('\xc9','')
    text_text[i]['text'] = text_text[i]['text'].replace('\U0001f610','').replace('\U0001f611','').replace('\u0d6b','').replace('\U0001f62d','').replace('\u2122','').replace('\U0001f612','').replace('\u2713','').replace('\U0001f692','').replace('\u0336','').replace('\uff61','').replace('\u21bc','').replace('\u203f','').replace('\u25d5','').replace('\u2611','').replace('\u1d39','').replace('\u1d40','').replace('\u2665','').replace('\xf8','').replace('\xfb','').replace('\xeb','').replace('\xc7','').replace('\xc8','').replace('\xf5','').replace('\xdc','').replace('\xd6','').replace('\xc1','').replace('\xc0','').replace('\xee','').replace('\xf6','').replace('\xdf','').replace('\xf4','').replace('\x90','').replace('\x91','').replace('\x9b','').replace('\xfb','').replace('\xb3','')
    text_text[i]['text'] = text_text[i]['text'].replace('\U0001f60d', '').replace('\U0001f44c', '').replace('\U0001f481', '').replace('\U0001f612', '').replace('\ufe0f', '').replace('\u0142', '').replace('\u0119', '').replace('\U0001f60a', '').replace('\u200b', '').replace('\u2024', '').replace('\u035e', '').replace('\u0332', '').replace('\u0353', '').replace('\u0326', '').replace('\u0356', '').replace('\u035a', '').replace('\u031b', '').replace('\U0001f34d', '').replace('\U0001f334', '').replace('\u011f', '').replace('\u0131', '').replace('\u201e', '').replace('\U0001f60f', '').replace('\u261e', '').replace('\u261c', '').replace('\U0001f63b', '').replace('\U0001f624', '').replace('\u20a6', '').replace('\u0e3f', '').replace('\u2130', '').replace('\u270c', '').replace('\u2620', '').replace('\u23b3', '').replace('\u279c', '').replace('\u2011', '').replace('\u0296', '').replace('\u0361', '').replace('\u1557', '').replace('\u0f3d', '').replace('\u035c', '').replace('\u0644', '').replace('\u0e88', '').replace('\u0f3c', '').replace('\u1559', '').replace('\u2022', '').replace('\u0153', '').replace('\u1f10', '').replace('\u03c2', '').replace('\u03ac', '').replace('\U0001f60e', '').replace('\u2764', '').replace('\U0001f602', '').replace('\U0001f609', '').replace('\U0001f495', '').replace('\U0001f633', '').replace('\u2661', '').replace('\u266b', '').replace('\uad6d', '').replace('\ud55c', '').replace('\uad6c', '').replace('\ub300', '').replace('\U0001f440', '').replace('\u263a', '').replace('\U0001f44d', '').replace('\U0001f604', '').replace('\u26bd', '').replace('\U0001f608', '').replace('\u25e1', '').replace('\u273f', '')
    file1.write(text_text[i]['text'])
    file1.write(rationales[i].replace('\U0001f612', '').replace('\U0001f334', '').replace('\xe5', '').replace('\u2022', '').replace('\U0001f495','').replace('\xeb','').replace('\u266b','').replace('\xe4','').replace('\u2665','').replace('\xf6','').replace('\xe2','').replace('\xfb','').replace('\xe7','').replace('\xf1','').replace('\xb4','').replace('\x93','').replace('\xc5','').replace('\xb3','').replace('\xab','').replace('\x89','').replace('\xad','').replace('\x81','').replace('\xc4','').replace('\xbc','').replace('\xae','').replace('\x80','').replace('\xc3','').replace('\xaa','').replace('\xa9','').replace('\xc2','').replace('\xbb','')+'\n')
    file1.write('____________'+'\n')
    file1.flush()
file1.close()
