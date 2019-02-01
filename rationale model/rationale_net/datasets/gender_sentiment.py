import gzip
import tqdm
from rationale_net.utils.embedding import get_indices_tensor
from rationale_net.datasets.factory import RegisterDataset
from rationale_net.datasets.abstract_dataset import AbstractDataset
import re
import random
random.seed(0)

SMALL_TRAIN_SIZE = 800

@RegisterDataset('gender_sentiment')
class LongGenderDataset(AbstractDataset):

    def __init__(self, args, word_to_indx, mode, max_length=800, stem='raw_data/gender_sentiment/all_sent_shuffle_'):
        #aspect = args.aspect
        self.args= args
        self.name = mode
        self.objective = args.objective
        self.dataset = []
        self.word_to_indx  = word_to_indx
        self.max_length = max_length
        #self.aspects_to_num = {'appearance':0, 'aroma':1, 'palate':2,'taste':3, 'overall':4}# 5 kind of labels
        #self.class_map = {0: 0, 1:0, 2:0, 3:0, 4:1, 5:1, 6:1, 7:1, 8:2, 9:2, 10:2}# mapping to 3 class
        self.name_to_key = {'train':'obftrain', 'dev':'obftrain', 'test':'test'}
        self.class_balance = {}
        #with gzip.open(stem+str(self.aspects_to_num[aspect])+'.'+self.name_to_key[self.name]+'.txt.gz') as gfile:
        with gzip.open(stem + self.name_to_key[self.name] + '.txt.gz') as gfile:
            lines = gfile.readlines()
            lines = list(zip(range(len(lines)), lines))
            random.shuffle(lines)
            if args.debug_mode:
                lines = lines[:SMALL_TRAIN_SIZE]
            elif self.name == 'dev':
                lines = lines[300000:305000]
            elif self.name == 'test':
                lines = lines[:50000]
            elif self.name == 'train':
                lines = lines[:300000]

            for indx, line in tqdm.tqdm(enumerate(lines)):
                uid, line_content = line
                #sample = self.processLine(line_content, self.aspects_to_num[aspect], indx)
                sample = self.processLine(line_content, indx)

                if not sample['y'] in self.class_balance:
                    self.class_balance[ sample['y'] ] = 0
                self.class_balance[ sample['y'] ] += 1
                sample['uid'] = uid
                self.dataset.append(sample)
            gfile.close()
        print ("Class balance", self.class_balance)

        if args.class_balance:
            raise NotImplementedError("sentiment gender dataset doesn't support balanced sampling!")

    ## Convert one line from beer dataset to {Text, Tensor, Labels}
    def processLine(self, line, i):
        if isinstance(line, bytes):
            line = line.decode()
        label_t = [ float(v) for v in line.split('\t')[0] ]
        #label = int(label_t[0]-1)
        label = int(label_t[0])

        text_list = line.split('\t')[-1].split()[:self.max_length]
        text = " ".join(text_list)
        self.args.num_class = 5
        x = get_indices_tensor(text_list, self.word_to_indx, self.max_length)
        sample = {'text':text,'x':x, 'y':label, 'i':i}
        return sample

