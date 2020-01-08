
# coding: utf-8

import os
import sys
import gzip
import random
import numpy as np
from colored import fg, attr, bg
import json

from dataset import SentenceClassification, BeerDatasetBinary, SentenceClassificationSet

class Sst2Dataset(SentenceClassification):
    def __init__(self, data_dir, truncate_num=300, freq_threshold=1, aspect=0, score_threshold=0.5, split_ratio=0.15):
        """
        This function initialize a data set from SST 2:
        Inputs:
            data_dir -- the directory containing the data
            aspect -- an integer of an aspect from 0-4
            truncate_num -- max length of the review text to use
        """
        # self.aspect = aspect
        # self.score_threshold = score_threshold
        # self.aspect_names = ['apperance', 'aroma', 'palate', 'taste']
        # self.split_ratio = split_ratio
        
        super(Sst2Dataset, self).__init__(data_dir, truncate_num, freq_threshold)
        
        self.truncated_word_dict = None
        self.truncated_vocab = None
        
        
    def _init_lm_output_vocab(self, truncate_freq=4):
        word_freq_dict = self._get_word_freq({'train':self.data_sets['train']})
        print('size of the raw vocabulary on training: %d'% len(word_freq_dict))
        print('size of the raw vocabulary on training: %d'% len(self.idx_2_word))

        self.truncated_word_dict = {0:0, 1:1, 2:2, 3:3}
        self.truncated_vocab = ['<PAD>', '<START>', '<END>', '<UNK>']

        for wid, word in self.idx_2_word.items():
            if wid < 4:
                continue
            elif word not in word_freq_dict:
                self.truncated_word_dict[wid] = self.truncated_word_dict[self.word_vocab['<UNK>']]
            else:
                if word_freq_dict[word] >= truncate_freq:
                    self.truncated_word_dict[wid] = len(self.truncated_vocab)
                    self.truncated_vocab.append(word)
                else:
                    self.truncated_word_dict[wid] = self.truncated_word_dict[self.word_vocab['<UNK>']]

        tmp_dict = {}
        for word, wid in self.truncated_word_dict.items():
            if wid not in tmp_dict:
                tmp_dict[wid] = 1
        print('size of the truncated vocabulary on training: %d'%len(tmp_dict))
        print('size of the truncated vocabulary on training: %d'%len(self.truncated_vocab))
        
        
    def load_dataset(self):
        
        # filein = open(os.path.join(self.data_dir, 'sec_name_dict.json'), 'r')
        # self.filtered_name_dict = json.load(filein)
        # filein.close()
        
        self.data_sets = {}
        self.label_vocab = {0:0, 1:1}

        # print('splitting with %.2f'%self.split_ratio)
        self.data_sets['train'] = self._load_data_set(os.path.join(self.data_dir, 'stsa.binary.train'))
        self.data_sets['dev'] = self._load_data_set(os.path.join(self.data_dir, 'stsa.binary.dev'))
        
        # load dev
        self.data_sets['test'] = self._load_data_set(os.path.join(self.data_dir, 'stsa.binary.test'))
    
        # build vocab
        self._build_vocab()
        
        self.idx2label = {val: key for key, val in self.label_vocab.items()}
        
        
    def _load_data_set(self, fpath, with_dev=False):
        """
        Inputs: 
            fpath -- the path of the file. 
        Outputs:
            positive_pairs -- a list of positive question-passage pairs
            negative_pairs -- a list of negative question-passage pairs
        """
        
        if with_dev:
            data_set = SentenceClassificationSetSubSampling()
        else:
            data_set = SentenceClassificationSet()
        
        # section_name_dict = {}
        
        with open(os.path.join(fpath), 'r') as f:
            for idx, line in enumerate(f):

                label_idx = 0
                sentence_start = 2
                label = int(line[label_idx])
                sentence = line[sentence_start:self.truncate_num + sentence_start]

                data_set.add_one(sentence.split(" "), label)
            
        data_set.print_info()

        return data_set

if __name__ == "__main__":
    test_case = 'sst2'
    
    if test_case == 'sst2': 
        data_dir = os.path.join("..", "datasets", "sst2")

        sst2_data = Sst2Dataset(data_dir)

        x_mat, y_vec, x_mask = sst2_data.get_batch('dev', range(2), sort=False)
        print(y_vec)
        print(x_mat[1])

        print("label vocab: ", sst2_data.label_vocab)
        sst2_data.display_sentence(x_mat[0])
        sst2_data.display_sentence(x_mat[1])

